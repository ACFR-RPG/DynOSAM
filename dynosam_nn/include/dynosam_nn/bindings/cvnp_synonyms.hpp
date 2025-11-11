#pragma once
#include <pybind11/numpy.h>

#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace cvnp {
struct TypeSynonyms {
  int cv_depth = -1;
  std::string cv_depth_name;
  std::string np_format;
  std::string np_format_long;

  pybind11::dtype dtype() { return pybind11::dtype(np_format); }
  std::string str() const;
};

extern std::vector<TypeSynonyms> sTypeSynonyms;

std::vector<TypeSynonyms> list_types_synonyms();
void print_types_synonyms();
}  // namespace cvnp
