#include "dynosam_common/utils/FileSystem.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {
namespace utils {

void throwExceptionIfPathInvalid(const std::string& image_path) {
  namespace fs = std::filesystem;
  if (!fs::exists(image_path)) {
    throw std::runtime_error("Path does not exist: " + image_path);
  }
}

void loadRGB(const std::string& image_path, cv::Mat& img) {
  throwExceptionIfPathInvalid(image_path);
  img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
}

void loadFlow(const std::string& image_path, cv::Mat& img) {
  throwExceptionIfPathInvalid(image_path);
  img = readOpticalFlow(image_path);
}

void loadDepth(const std::string& image_path, cv::Mat& img) {
  throwExceptionIfPathInvalid(image_path);
  img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
  img.convertTo(img, CV_64F);
}

void loadSemanticMask(const std::string& image_path, const cv::Size& size,
                      cv::Mat& mask) {
  throwExceptionIfPathInvalid(image_path);
  CHECK(!size.empty());

  mask = cv::Mat(size, CV_32SC1);

  std::ifstream file_mask;
  file_mask.open(image_path.c_str());

  int count = 0;
  while (!file_mask.eof()) {
    std::string s;
    getline(file_mask, s);
    if (!s.empty()) {
      std::stringstream ss;
      ss << s;
      int tmp;
      for (int i = 0; i < mask.cols; ++i) {
        ss >> tmp;
        if (tmp != 0) {
          mask.at<int>(count, i) = tmp;
        } else {
          mask.at<int>(count, i) = 0;
        }
      }
      count++;
    }
  }

  file_mask.close();
}

void loadMask(const std::string& image_path, cv::Mat& mask) {
  throwExceptionIfPathInvalid(image_path);
  mask = cv::imread(image_path, cv::IMREAD_UNCHANGED);
  mask.convertTo(mask, CV_32SC1);
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
  getline(fstream, line);

  split_lines.clear();

  if (line.empty()) return false;

  split_lines = trimAndSplit(line);
  return true;
}

}  // namespace utils
}  // namespace dyno
