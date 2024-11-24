#include "fitserial.h"

#include <cmath>
#include <iostream>

/*
 * Basé sur la classe Fitab
 * Modifié pour calculer le coefficient de détermination R
 * Numerical Recipies, The Art of Scientific Computing, Third Edition, Chap. 15
 */

FitSerial::FitSerial() {}
FitSerial::~FitSerial() {}

void FitSerial::fit(double* x, double* y, int n, FitResult& res) {
  double a = 0;
  double b = 0;
  double sx = 0;
  double sy = 0;
  for (int i = 0; i < n; i++) {
    sx += x[i];
    sy += y[i];
  }

  double ss = n;
  double xmean = sx / n;
  double ymean = sy / n;
  double ssxym = 0;
  double ssxm = 0;
  double ssym = 0;
  double t, u;

  for (int i = 0; i < n; i++) {
    t = x[i] - xmean;
    u = y[i] - ymean;
    ssxm += t * t;
    ssym += u * u;
    ssxym += t * u;
    b += t * y[i];
  }

  // Résolution de l'équation linéaire
  // Très simple pour un polynome de degré 1
  b = b / ssxm;
  a = (sy - sx * b) / ss;

  ssxm = ssxm / n;
  ssym = ssym / n;
  ssxym = ssxym / n;

  // Coefficient R
  double r = 0;
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

  // fin
  return;
}
