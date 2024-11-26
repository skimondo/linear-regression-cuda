#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "cudacheck.h"
#include "fitcuda.h"
#include "fitserial.h"

#define COARSE_FACTOR 8
#define THREADS_PER_BLOCK 1024
#define GRID_DIMENSION 80

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

__global__ void kernel_sum_coarse_two_dep(double* xArray, double* xmean, double* yArray, double* ymean, double* result, int size) {
  // mémoire partagée par le warp
  // doit être allouée au lancement
  // on obtient en pratique le début de cet espace
  extern __shared__ double input_s[];

  unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x;

  double sum_local = 0.0;

  for (unsigned int tile = 0; tile < COARSE_FACTOR * 2; tile++) {
    unsigned int src = i + tile * blockDim.x;
    if (src < size) {
      double diff_x = xArray[src] - *xmean;
      double diff_y = yArray[src] - *ymean;
      sum_local += diff_x * diff_y;
    }
  }

  input_s[t] = sum_local;

  // Réduction en mémoire partagée (et non en mémoire globale)
  // si blockDim = 32, stride = 16, 8, 4, 2, 1 (5 itérations)
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
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

FitCuda::FitCuda() {
  // test pour vérifier que nous avons bel et bien un périphérique
  int deviceId;
  cudaCheck(cudaGetDevice(&deviceId));
}

FitCuda::~FitCuda() {}

void FitCuda::fit(double* xarray, double* yarray, int size, FitResult& res) {
  double a = 0;
  double b = 0;
  double r = 0;
  double sx = 0.0;
  double sy = 0.0;
  double ss = size;
  double xmean = 0.0;
  double ymean = 0.0;
  double ssxym = 0.0;
  double ssxm = 0.0;
  double ssym = 0.0;

/******************************************************************/
/* sum(x) and sum(y)
/******************************************************************/

  // double result;
  double* xarray_d;
  double* yarray_d;
  double* xmean_d;
  double* ymean_d;
  double* result_d;

  cudaMalloc(&xarray_d, size * sizeof(double));
  cudaMalloc(&yarray_d, size * sizeof(double));
  cudaMalloc(&xmean_d, sizeof(double));
  cudaMalloc(&ymean_d, sizeof(double));

  cudaMalloc(&result_d, sizeof(double));

  cudaMemcpy(xarray_d, xarray, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(yarray_d, yarray, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(result_d, 0, sizeof(double));

  // Pour la fraction de GPU V100
  int blockDim = THREADS_PER_BLOCK;
  int gridDim = (size + blockDim * COARSE_FACTOR * 2 - 1) / (blockDim * COARSE_FACTOR * 2);
  // int gridDim = GRID_DIMENSION;
  int sharedSize = blockDim * sizeof(double);  // taille du tableau extern __shared__ double input_s[]
  kernel_sum_coarse<<<gridDim, blockDim, sharedSize>>>(xarray_d, result_d, size);
  cudaCheck(cudaDeviceSynchronize());
  cudaMemcpy(&sx, result_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemset(result_d, 0, sizeof(double));
  kernel_sum_coarse<<<gridDim, blockDim, sharedSize>>>(yarray_d, result_d, size);
  cudaCheck(cudaDeviceSynchronize());
  cudaMemcpy(&sy, result_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemset(result_d, 0, sizeof(double));


/******************************************************************/
/* sxxm, ssym, ssxym
/******************************************************************/

  xmean = sx / size;
  ymean = sy / size;

  cudaMemcpy(xmean_d, &xmean, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(ymean_d, &ymean, sizeof(double), cudaMemcpyHostToDevice);

  // ssxm = reduction_one_dep(xarray, xmean, size);
  kernel_sum_coarse_one_dep<<<gridDim, blockDim, sharedSize>>>(xarray_d, xmean_d, result_d, size);
  cudaCheck(cudaDeviceSynchronize());
  cudaMemcpy(&ssxm, result_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemset(result_d, 0, sizeof(double));

  // ssym = reduction_one_dep(yarray, ymean, size);
  kernel_sum_coarse_one_dep<<<gridDim, blockDim, sharedSize>>>(yarray_d, ymean_d, result_d, size);
  cudaCheck(cudaDeviceSynchronize());
  cudaMemcpy(&ssym, result_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemset(result_d, 0, sizeof(double));

  // ssxym = reduction_two_dep(xarray, xmean, yarray, ymean, size);
  kernel_sum_coarse_two_dep<<<gridDim, blockDim, 2*sharedSize>>>(xarray_d, xmean_d, yarray_d, ymean_d, result_d, size);
  cudaCheck(cudaDeviceSynchronize());
  cudaMemcpy(&ssxym, result_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemset(result_d, 0, sizeof(double));

  std::cout << "\nssxm: " << ssxm << std::endl;
  std::cout << "ssym: " << ssym << std::endl;
  std::cout << "ssxym: " << ssxym << std::endl;

  b = ssxym / ssxm;
  a = (sy - sx * b) / ss;

  ssxm = ssxm / size;
  ssym = ssym / size;
  ssxym = ssxym / size;

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

  cudaFree(xarray_d);
  cudaFree(yarray_d);
  cudaFree(xmean_d);
  cudaFree(ymean_d);
  cudaFree(result_d);
}

// Only kept to pass the tests, not used in the fit function
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


// double FitCuda::reduction_one_dep(double* xArray, double xmean, int size) {
//   double result;
//   double* xArray_d;
//   double* xmean_d;
//   double* result_d;
//   cudaMalloc(&xArray_d, size * sizeof(double));
//   cudaMalloc(&xmean_d, sizeof(double));
//   cudaMalloc(&result_d, sizeof(double));

//   cudaMemcpy(xArray_d, xArray, size * sizeof(double), cudaMemcpyHostToDevice);
//   cudaMemcpy(xmean_d, &xmean, sizeof(double), cudaMemcpyHostToDevice);
//   cudaMemset(result_d, 0, sizeof(double));

//   // Pour la fraction de GPU V100
//   int blockDim = 1024;
//   int gridDim = 80;
//   int sharedSize = blockDim * sizeof(double);  // taille du tableau extern __shared__ double input_s[]
//   kernel_sum_coarse_one_dep<<<gridDim, blockDim, sharedSize>>>(xArray_d, xmean_d, result_d, size);
//   cudaCheck(cudaDeviceSynchronize());

//   cudaMemcpy(&result, result_d, sizeof(double), cudaMemcpyDeviceToHost);
//   return result;
// }

