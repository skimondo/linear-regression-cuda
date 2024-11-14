#include <uqam/tp.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "experiments.h"
#include "fitcuda.h"
#include "fitserial.h"
#include "optparser.hpp"

int main(int argc, char** argv) {
  int method = 0;
  int num_points = 10;
  std::string output = "";

  OptionsParser args(argc, argv);
  args.AddOption(&method, "-p", "--parallel", "parallel (0: serial, 1: cuda)");
  args.AddOption(&num_points, "-n", "--num-points", "number of points");
  args.AddOption(&output, "-o", "--output", "points output file");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    return 1;
  }
  args.PrintOptions(std::cout);

  // Générer des données
  std::vector<double> data_x;
  std::vector<double> data_y;
  experiment_basic(data_x, data_y, num_points, 0, 1, 10, 2);

  // Sauvegarder les points
  if (output.length() > 0){
    std::ofstream ofs(output);
    for (int i = 0; i < num_points; i++) {
      ofs << data_x[i] << " " << data_y[i] << "\n";
    }
  }

  // Construire l'arbre VP
  IFitBase* fitter;
  if (method == 0) {
    fitter = new FitSerial();
  } else if (method == 1) {
    fitter = new FitCuda();
  } else {
    printf("oups, mauvaise options pour la méthode\n");
    return 1;
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "régression linéaire... " << std::flush;
  FitResult res;
  fitter->fit(data_x.data(), data_y.data(), num_points, res);
  std::cout << "terminé!" << std::endl;

  auto t2 = std::chrono::high_resolution_clock::now();
  double dt = std::chrono::duration<double>(t2 - t1).count();
  double pps = 1000.0 * num_points / dt;
  std::cout << "temps d'exécution: " << dt << " s\n";
  std::cout << "vitesse          : " << pps << " pps\n";

  std::cout << "a:    " << res.a << "\n";
  std::cout << "b:    " << res.b << "\n";
  std::cout << "r:    " << res.r << "\n";
  std::cout << "xmean:" << res.xmean << "\n";
  std::cout << "ymean:" << res.ymean << "\n";

  delete fitter;
  std::cout << "Fin normale du programme" << std::endl;
  return 0;
}
