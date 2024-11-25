#include <experiments.h>
#include <fitcuda.h>
#include <uqam/tp.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>

#include "testutils.h"
using namespace Catch;

static const double tol = 1.0e-6;

TEST_CASE("FitCudaValidation") {
  int n = 100;
  double a = 10;  // ordonnée à l'origine
  double b = 2;   // pente
  std::vector<double> data_x;
  std::vector<double> data_y;
  experiment_basic(data_x, data_y, n, 0, 1, a, b);

  FitResult res;
  FitCuda fitter;
  fitter.fit(data_x.data(), data_y.data(), n, res);
  CHECK_THAT(res.a, Catch::Matchers::WithinAbs(a, tol));
  CHECK_THAT(res.b, Catch::Matchers::WithinAbs(b, tol));
}

TEST_CASE("FitCudaReduction") {
  // La réduction d'un vecteur de 1 de taille n devrait donner la somme n*1
  int n = 1E6;
  std::vector<double> v(n, 1.0);
  double ref = std::accumulate(v.begin(), v.end(), 0);
  FitCuda fitter;
  double res = fitter.reduction(v.data(), v.size());
  CHECK_THAT(res, Catch::Matchers::WithinAbs(ref, tol));
}

TEST_CASE("FitCudaValidation2") {
  int n = 10000;
  double a = 10;  // ordonnée à l'origine
  double b = 2;   // pente
  std::vector<double> data_x;
  std::vector<double> data_y;
  experiment_basic(data_x, data_y, n, 0, 1, a, b);

  FitResult res;
  FitCuda fitter;
  fitter.fit(data_x.data(), data_y.data(), n, res);
  CHECK_THAT(res.a, Catch::Matchers::WithinAbs(a, tol));
  CHECK_THAT(res.b, Catch::Matchers::WithinAbs(b, tol));
}

TEST_CASE("FitCudaValidation3") {
  int n = 100000;
  double a = 10;  // ordonnée à l'origine
  double b = 2;   // pente
  std::vector<double> data_x;
  std::vector<double> data_y;
  experiment_basic(data_x, data_y, n, 0, 1, a, b);

  FitResult res;
  FitCuda fitter;
  fitter.fit(data_x.data(), data_y.data(), n, res);
  CHECK_THAT(res.a, Catch::Matchers::WithinAbs(a, tol));
  CHECK_THAT(res.b, Catch::Matchers::WithinAbs(b, tol));
}

TEST_CASE("FitCudaValidation4") {
  int n = 1000000;
  double a = 10;  // ordonnée à l'origine
  double b = 2;   // pente
  std::vector<double> data_x;
  std::vector<double> data_y;
  experiment_basic(data_x, data_y, n, 0, 1, a, b);

  FitResult res;
  FitCuda fitter;
  fitter.fit(data_x.data(), data_y.data(), n, res);
  CHECK_THAT(res.a, Catch::Matchers::WithinAbs(a, tol));
  CHECK_THAT(res.b, Catch::Matchers::WithinAbs(b, tol));
}

TEST_CASE("FitCudaValidation5") {
  int n = 10000000;
  double a = 10;  // ordonnée à l'origine
  double b = 2;   // pente
  std::vector<double> data_x;
  std::vector<double> data_y;
  experiment_basic(data_x, data_y, n, 0, 1, a, b);

  FitResult res;
  FitCuda fitter;
  fitter.fit(data_x.data(), data_y.data(), n, res);
  CHECK_THAT(res.a, Catch::Matchers::WithinAbs(a, tol));
  CHECK_THAT(res.b, Catch::Matchers::WithinAbs(b, tol));
}