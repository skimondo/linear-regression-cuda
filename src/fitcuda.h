#pragma once

#include "fitserial.h"

class FitCuda : public IFitBase {
public:
  FitCuda();
  ~FitCuda();
  void fit(double* x, double* y, int n, FitResult& res);
  double reduction(double* v, int n);
  double reduction_one_dep(double* xArray, double xmean, int size);
  double reduction_two_dep(double* xArray, double xmean, double* yArray, double ymean, int size);
};
