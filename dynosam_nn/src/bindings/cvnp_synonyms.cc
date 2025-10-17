#include "dynosam_nn/bindings/cvnp_synonyms.hpp"

#include <iostream>
#include <string>

namespace cvnp {
std::vector<TypeSynonyms> sTypeSynonyms{
    {CV_8U, "CV_8U", pybind11::format_descriptor<uint8_t>::format(),
     "np.uint8"},
    {CV_8S, "CV_8S", pybind11::format_descriptor<int8_t>::format(), "np.int8"},
    {CV_16U, "CV_16U", pybind11::format_descriptor<uint16_t>::format(),
     "np.uint16"},
    {CV_16S, "CV_16S", pybind11::format_descriptor<int16_t>::format(),
     "np.int16"},
    {CV_32S, "CV_32S", pybind11::format_descriptor<int32_t>::format(),
     "np.int32"},
    {CV_32F, "CV_32F", pybind11::format_descriptor<float>::format(), "float"},
    {CV_64F, "CV_64F", pybind11::format_descriptor<double>::format(),
     "np.float64"},

    // Note: this format needs adaptations
    //#if (CV_MAJOR_VERSION >= 4)
    //        { CV_16F, "CV_16F",
    //        pybind11::format_descriptor<cv::float16_t>::format() },
    //#endif
};

static int sColumnWidth = 12;

static std::string align_center(const std::string& s) {
  int nb_spaces = s.size() < sColumnWidth ? sColumnWidth - s.size() : 0;
  int nb_spaces_left = nb_spaces / 2;
  int nb_spaces_right = sColumnWidth - s.size() - nb_spaces_left;
  if (nb_spaces_right < 0) nb_spaces_right = 0;
  return std::string((size_t)nb_spaces_left, ' ') + s +
         std::string((size_t)nb_spaces_right, ' ');
}
static std::string align_center(const int v) {
  return align_center(std::to_string(v));
}

std::string TypeSynonyms::str() const {
  return align_center(cv_depth) + align_center(cv_depth_name) +
         align_center(np_format) + align_center(np_format_long);
}

std::string _print_types_synonyms_str() {
  std::string title = align_center("cv_depth") + align_center("cv_depth_name") +
                      align_center("np_format") +
                      align_center("np_format_long");
  ;

  std::string r;
  r = title + "\n";
  for (const auto& format : sTypeSynonyms) r = r + format.str() + "\n";
  return r;
}

std::vector<TypeSynonyms> list_types_synonyms() { return sTypeSynonyms; }

void print_types_synonyms() { std::cout << _print_types_synonyms_str(); }
}  // namespace cvnp
