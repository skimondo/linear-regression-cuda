#pragma once

#include "fitserial.h"

class FitCuda : public IFitBase {
public:
  FitCuda();
  ~FitCuda();
  void fit(double* x, double* y, int n, FitResult& res);
  double reduction(double* v, int n);
};
