#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "cudacheck.h"
#include "fitcuda.h"
#include "fitserial.h"

#define COARSE_FACTOR 8

// Patron de réduction vu en classe
__global__ void kernel_sum_coarse(double* input, double* result, int size) {
  // mémoire partagée par le warp
  // doit être allouée au lancement
  // on obtient en pratique le début de cet espace
  extern __shared__ double input_s[];

  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;

  // Somme des éléments jusqu'à obtenir un seul bloc
  double sum_local = (i < size) ? input[i] : 0.0;
  for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; tile++) {
    unsigned int src = i + tile * blockDim.x;
    if (src < size) {
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

__global__ void kernel_sum_coarse_one_dep(double* input, double* mean ,double* result, int size) {
  // mémoire partagée par le warp
  // doit être allouée au lancement
  // on obtient en pratique le début de cet espace
  extern __shared__ double input_s[];

  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;

  // double sum_local = (i < size) ? input[i] - *mean : 0.0;
  double sum_local = 0.0;

  for (unsigned int tile = 0; tile < COARSE_FACTOR * 2; tile++) {
    unsigned int src = i + tile * blockDim.x;
    if (src < size) {
      // sum_local = sum_local + (input[src] - *mean) * (input[src] - *mean);
      double diff = input[src] - *mean;
      sum_local += diff * diff;
    }
  }

  input_s[t] = sum_local;

  // Réduction en mémoire partagée (et non en mémoire globale)
  // si blockDim = 32, stride = 16, 8, 4, 2, 1 (5 itérations)
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (t < stride) {
      // input_s[t] = input_s[t] + (input_s[t + stride] - *mean) * (input_s[t + stride] - *mean);
      input_s[t] += input_s[t + stride];
    }
  }

  // Ajouter le résultat du bloc
  if (t == 0) {
    atomicAdd(result, input_s[0]);
  }
}

FitCuda::FitCuda() {
  // test pour vérifier que nous avons bel et bien un périphérique
  int deviceId;
  cudaCheck(cudaGetDevice(&deviceId));
}

FitCuda::~FitCuda() {}

void FitCuda::fit(double* x, double* y, int n, FitResult& res) {
  double a = 0;
  double b = 0;
  double r = 0;
  double sx = 0.0;
  double sy = 0.0;
  double ss = n;
  double xmean = 0.0;
  double ymean = 0.0;
  double ssxym = 0.0;
  double ssxm = 0.0;
  double ssym = 0.0;
  double t = 0.0;
  double u = 0.0;

  sx = reduction(x, n);
  sy = reduction(y, n);

  xmean = sx / n;
  ymean = sy / n;

  ssxm = reduction_one_dep(x, xmean, n);
  ssym = reduction_one_dep(y, ymean, n);
  std::cout << "\nssxm: " << ssxm << std::endl;
  std::cout << "ssym: " << ssym << std::endl;


  for (int i = 0; i < n; i++) {
    t = x[i] - xmean;
    u = y[i] - ymean;
    // ssxm += t * t;
    // ssym += u * u;
    ssxym += t * u;
    b += t * y[i];
  }

  b = b / ssxm;
  a = (sy - sx * b) / ss;

  ssxm = ssxm / n;
  ssym = ssym / n;
  ssxym = ssxym / n;

  // Coefficient R
  if ((ssxm != 0.0) && (ssym != 0.0)) {
    r = ssxym / sqrt(ssxm * ssym);
    if (r > 1.0) {
      r = 1.0;
    } else if (r < -1.0) {
      r = -1.0;
    }
  }

  res.a = a;
  res.b = b;
  res.r = r;
  res.xmean = xmean;
  res.ymean = ymean;
}


double FitCuda::reduction(double* array, int size) {
  double result;
  double* array_d;
  double* result_d;
  cudaMalloc(&array_d, size * sizeof(double));
  cudaMalloc(&result_d, sizeof(double));

  cudaMemcpy(array_d, array, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(result_d, 0, sizeof(double));

  // Pour la fraction de GPU V100
  int blockDim = 1024;
  int gridDim = 80;
  int sharedSize = blockDim * sizeof(double);  // taille du tableau extern __shared__ double input_s[]
  kernel_sum_coarse<<<gridDim, blockDim, sharedSize>>>(array_d, result_d, size);
  cudaCheck(cudaDeviceSynchronize());

  cudaMemcpy(&result, result_d, sizeof(double), cudaMemcpyDeviceToHost);
  return result;
}


double FitCuda::reduction_one_dep(double* xArray, double xmean, int size) {
  double result;
  double* xArray_d;
  double* xmean_d;
  double* result_d;
  cudaMalloc(&xArray_d, size * sizeof(double));
  cudaMalloc(&xmean_d, sizeof(double));
  cudaMalloc(&result_d, sizeof(double));

  cudaMemcpy(xArray_d, xArray, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(xmean_d, &xmean, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(result_d, 0, sizeof(double));

  // Pour la fraction de GPU V100
  int blockDim = 1024;
  int gridDim = 80;
  int sharedSize = blockDim * sizeof(double);  // taille du tableau extern __shared__ double input_s[]
  kernel_sum_coarse_one_dep<<<gridDim, blockDim, sharedSize>>>(xArray_d, xmean_d, result_d, size);
  cudaCheck(cudaDeviceSynchronize());

  cudaMemcpy(&result, result_d, sizeof(double), cudaMemcpyDeviceToHost);
  return result;
}

