#include "experiments.h"

#include <random>

void experiment_basic(std::vector<double>& x,  //
                      std::vector<double>& y,  //
                      int n,                   //
                      double x1,               //
                      double x2,               //
                      double a,                //
                      double b                 //
) {
  x.resize(n);
  y.resize(n);
  double step = (x2 - x1) / (n - 1);
  for (int i = 0; i < n; i++) {
    x[i] = i * step;
    y[i] = b * x[i] + a;
  }
}

void experiment_noisy(std::vector<double>& x,  //
                      std::vector<double>& y,  //
                      int n,                   //
                      double x1,               //
                      double x2,               //
                      double a,                //
                      double b                 //
) {
  std::default_random_engine rng(0);
  std::normal_distribution<double> noise(-1.0, 1.0);

  x.resize(n);
  y.resize(n);
  double step = (x2 - x1) / (n - 1);
  for (int i = 0; i < n; i++) {
    x[i] = i * step;
    y[i] = a * x[i] + b + noise(rng);
    // y[i] = b * x[i] + a + noise(rng);

  }
}
