#include "dynosam_common/utils/FileSystem.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace dyno::utils {

void throwExceptionIfPathInvalid(const std::string& image_path) {
  namespace fs = std::filesystem;
  if (!fs::exists(image_path)) {
    throw std::runtime_error("Path does not exist: " + image_path);
  }
}

std::vector<std::filesystem::path> getAllFilesInDir(
    const std::string& folder_path) {
  std::vector<std::filesystem::path> files_in_directory;
  std::copy(std::filesystem::directory_iterator(folder_path),
            std::filesystem::directory_iterator(),
            std::back_inserter(files_in_directory));
  std::sort(files_in_directory.begin(), files_in_directory.end());
  return files_in_directory;
}

void loadPathsInDirectory(
    std::vector<std::string>& file_paths, const std::string& folder_path,
    const std::function<bool(const std::string&)>& condition) {
  std::function<bool(const std::string&)> impl_condition;
  if (condition) {
    impl_condition = condition;
  } else {
    // if no condition is provided, set condition to always return true; adding
    // all the files found
    impl_condition = [](const std::string&) -> bool { return true; };
  }

  auto files_in_directory = getAllFilesInDir(folder_path);
  for (const std::string file_path : files_in_directory) {
    throwExceptionIfPathInvalid(file_path);

    // if condition is true, add
    if (impl_condition(file_path)) file_paths.push_back(file_path);
  }
}

std::vector<std::string> trimAndSplit(const std::string& input,
                                      const std::string& delimiter) {
  std::string trim_input = boost::algorithm::trim_right_copy(input);
  std::vector<std::string> split_line;
  boost::algorithm::split(split_line, trim_input, boost::is_any_of(delimiter));
  return split_line;
}

bool getLine(std::ifstream& fstream, std::vector<std::string>& split_lines) {
  std::string line;
  std::getline(fstream, line);

  split_lines.clear();

  if (line.empty()) return false;

  split_lines = trimAndSplit(line);
  return true;
}

}  // namespace dyno::utils
