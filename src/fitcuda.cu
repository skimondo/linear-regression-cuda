#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "cudacheck.h"
#include "fitcuda.h"
#include "fitserial.h"

#define COARSE_FACTOR 8

// Patron de réduction vu en classe
__global__ void kernel_sum_coarse(double* input, double* result, int n) {
  // mémoire partagée par le warp
  // doit être allouée au lancement
  // on obtient en pratique le début de cet espace
  extern __shared__ double input_s[];

  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;

  // Somme des éléments jusqu'à obtenir un seul bloc
  double sum_local = (i < n) ? input[i] : 0.0;
  for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; tile++) {
    unsigned int src = i + tile * blockDim.x;
    if (src < n) {
      sum_local += input[src];
    }
  }

  input_s[t] = sum_local;

  // Réduction en mémoire partagée (et non en mémoire globale)
  // si blockDim = 32, stride = 16, 8, 4, 2, 1 (5 itérations)
  for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }

  // Ajouter le résultat du bloc
  if (t == 0) {
    atomicAdd(result, input_s[0]);
  }
}

__global__ void square_kernel(double* input, double* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] * input[idx];
  }
}

__global__ void product_kernel(double* input1, double* input2, double* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input1[idx] * input2[idx];
  }
}

FitCuda::FitCuda() {
  // test pour vérifier que nous avons bel et bien un périphérique
  int deviceId;
  cudaCheck(cudaGetDevice(&deviceId));
}

FitCuda::~FitCuda() {}

void FitCuda::fit(double* x, double* y, int n, FitResult& res) {
  // Allocate device memory
  double *x_dev, *y_dev;
  cudaMalloc(&x_dev, n * sizeof(double));
  cudaMalloc(&y_dev, n * sizeof(double));

          // Copy input data to device
  cudaMemcpy(x_dev, x, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_dev, y, n * sizeof(double), cudaMemcpyHostToDevice);

          // Allocate memory for reduction results
  double *result_dev;
  cudaMalloc(&result_dev, 5 * sizeof(double));  // sx, sy, sx2, sxy, sy2

          // Host memory to retrieve results
  double result_host[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

          // Compute sums (sx, sy), sums of squares (sx2, sy2), and product sum (sxy)
          // `reduction` method calculates the sum over the array
  double sx = reduction(x_dev, n);
  double sy = reduction(y_dev, n);

          // Square the values and calculate reductions
  double *x_squared, *y_squared, *xy_product;
  cudaMalloc(&x_squared, n * sizeof(double));
  cudaMalloc(&y_squared, n * sizeof(double));
  cudaMalloc(&xy_product, n * sizeof(double));

          // Launch kernels to compute squares and products
  int blockDim = 1024;
  int gridDim = (n + blockDim - 1) / blockDim;

  square_kernel<<<gridDim, blockDim>>>(x_dev, x_squared, n);  // Compute x^2
  square_kernel<<<gridDim, blockDim>>>(y_dev, y_squared, n);  // Compute y^2
  product_kernel<<<gridDim, blockDim>>>(x_dev, y_dev, xy_product, n);  // Compute x*y
  cudaDeviceSynchronize();

  double sx2 = reduction(x_squared, n);
  double sy2 = reduction(y_squared, n);
  double sxy = reduction(xy_product, n);

          // Free temporary memory
  cudaFree(x_squared);
  cudaFree(y_squared);
  cudaFree(xy_product);

          // Calculate mean values
  double xmean = sx / n;
  double ymean = sy / n;

          // Calculate slope (b) and intercept (a)
  double denominator = sx2 - sx * sx / n;
  double b = (sxy - sx * sy / n) / denominator;
  double a = (sy - b * sx) / n;

          // Calculate the coefficient of determination R
  double ss_tot = sy2 - sy * sy / n;  // Total sum of squares
  double ss_res = ss_tot - b * sxy;  // Residual sum of squares
  double r = 1 - (ss_res / ss_tot);

          // Store results
  res.a = a;
  res.b = b;
  res.r = r;
  res.xmean = xmean;
  res.ymean = ymean;

          // Free device memory
  cudaFree(x_dev);
  cudaFree(y_dev);
  cudaFree(result_dev);
}


double FitCuda::reduction(double* v, int n) {
  double res;
  double* v_dev;
  double* res_dev;
  cudaMalloc(&v_dev, n * sizeof(double));
  cudaMalloc(&res_dev, sizeof(double));

  cudaMemcpy(v_dev, v, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(res_dev, 0, sizeof(double));

  // Pour la fraction de GPU V100
  int blockDim = 1024;
  int gridDim = 80;
  int sharedSize = blockDim * sizeof(double);  // taille du tableau extern __shared__ double input_s[]
  kernel_sum_coarse<<<gridDim, blockDim, sharedSize>>>(v_dev, res_dev, n);
  cudaCheck(cudaDeviceSynchronize());

  cudaMemcpy(&res, res_dev, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
}
