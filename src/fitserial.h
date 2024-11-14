#pragma once

#include <queue>
#include <random>
#include <vector>

struct FitResult {
  FitResult() : a(0), b(0), r(0), xmean(0), ymean(0) {}
  double a;
  double b;
  double r;
  double xmean;
  double ymean;
};

class IFitBase {
public:
  IFitBase(){};
  virtual ~IFitBase(){};
  virtual void fit(double* x, double* y, int n, FitResult& res) = 0;
};

class FitSerial : public IFitBase {
public:
  FitSerial();
  ~FitSerial();
  void fit(double* x, double* y, int n, FitResult& res);
};
