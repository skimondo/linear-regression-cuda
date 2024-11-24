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
  double xmean = 0.0;
  double ymean = 0.0;

  // À COMPLÉTER

  res.a = a;
  res.b = b;
  res.r = r;
  res.xmean = xmean;
  res.ymean = ymean;
  return;
}

double FitCuda::reduction(double* v, int n) {
  double res;
  double* v_d;
  double* res_d;
  cudaMalloc(&v_d, n * sizeof(double));
  cudaMalloc(&res_d, sizeof(double));

  cudaMemcpy(v_d, v, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(res_d, 0, sizeof(double));

  // Pour la fraction de GPU V100
  int blockDim = 1024;
  int gridDim = 80;
  int sharedSize = blockDim * sizeof(double);  // taille du tableau extern __shared__ double input_s[]
  kernel_sum_coarse<<<gridDim, blockDim, sharedSize>>>(v_d, res_d, n);
  cudaCheck(cudaDeviceSynchronize());

  cudaMemcpy(&res, res_d, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
}
