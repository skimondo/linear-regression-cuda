#include <experiments.h>
#include <fitserial.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <vector>

#include "testutils.h"

static const double tol = 1.0e-6;

TEST_CASE("Serial") {
  int n = 100;
  double a = 10;  // ordonnée à l'origine
  double b = 2;   // pente
  std::vector<double> data_x;
  std::vector<double> data_y;
  experiment_basic(data_x, data_y, n, 0, 1, a, b);

  FitResult res;
  FitSerial fitter;
  fitter.fit(data_x.data(), data_y.data(), n, res);
  CHECK_THAT(res.a, Catch::Matchers::WithinAbs(a, tol));
  CHECK_THAT(res.b, Catch::Matchers::WithinAbs(b, tol));
}
