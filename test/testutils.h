#pragma once

#include <fitserial.h>

#include <catch2/catch_all.hpp>

namespace Catch {

// Permet d'afficher le résultat de la comparaison dans un test
template <>
struct StringMaker<FitResult> {
  static std::string convert(const FitResult& obj) {
    return "(" + std::to_string(obj.a) + ", " + std::to_string(obj.b) + ", " + std::to_string(obj.r) + ")";
  }
};

}  // namespace Catch
