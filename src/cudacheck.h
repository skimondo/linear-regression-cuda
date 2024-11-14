#pragma once

#include <cuda_runtime.h>

// Macro utile pour simplifier la gestion d'erreur
#define cudaCheck(call)                                                                                            \
  {                                                                                                                \
    cudaError_t err = call;                                                                                        \
    if (err != cudaSuccess) {                                                                                      \
      std::cerr << "CUDA error in file " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) \
                << " (" << err << ")" << std::endl;                                                                \
      exit(1);                                                                                                     \
    }                                                                                                              \
  }
