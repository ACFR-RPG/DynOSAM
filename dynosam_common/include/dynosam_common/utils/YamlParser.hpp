/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include <glog/logging.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <string>

#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/utils/FileSystem.hpp"
#include "dynosam_common/utils/Macros.hpp"

namespace dyno {

class YamlParser {
 public:
  DYNO_POINTER_TYPEDEFS(YamlParser)

  YamlParser(const std::string& filepath) : fs_(), filepath_(filepath) {
    utils::throwExceptionIfPathInvalid(filepath_);
    openFile(filepath, &fs_);
  }
  ~YamlParser() { closeFile(&fs_); }

  template <class T>
  bool getYamlParam(const std::string& id, T* output) const {
    CHECK(!id.empty());
    CHECK_NOTNULL(output);
    const cv::FileNode& file_handle = fs_[id];
    CHECK_NE(file_handle.type(), cv::FileNode::NONE)
        << "Missing parameter: " << id.c_str()
        << " in file: " << filepath_.c_str();
    file_handle >> *output;
    return true;
  }

  template <class T>
  bool getYamlParam(const std::string& id, T* output, T default_value) const {
    CHECK(!id.empty());
    CHECK_NOTNULL(output);
    const cv::FileNode& file_handle = fs_[id];
    if (file_handle.type() == cv::FileNode::NONE) {
      *output = default_value;
      return false;
    } else {
      file_handle >> *output;
      return true;
    }
  }

  template <class T>
  void getNestedYamlParam(const std::string& id, const std::string& id_2,
                          T* output) const {
    CHECK(!id.empty());
    CHECK(!id_2.empty());
    const cv::FileNode& file_handle = fs_[id];
    CHECK_NE(file_handle.type(), cv::FileNode::NONE)
        << "Missing parameter: " << id.c_str()
        << " in file: " << filepath_.c_str();
    const cv::FileNode& file_handle_2 = file_handle[id_2];
    CHECK_NE(file_handle_2.type(), cv::FileNode::NONE)
        << "Missing nested parameter: " << id_2.c_str() << " inside "
        << id.c_str() << '\n'
        << " in file: " << filepath_.c_str();
    CHECK(file_handle.isMap())
        << "I think that if this is not a map, we can't use >>";
    file_handle_2 >> *CHECK_NOTNULL(output);
  }

  template <class T>
  void getNestedYamlParam(const std::string& id, const std::string& id_2,
                          T* output, T default_value) const {
    CHECK(!id.empty());
    CHECK(!id_2.empty());
    const cv::FileNode& file_handle = fs_[id];
    CHECK_NE(file_handle.type(), cv::FileNode::NONE)
        << "Missing parameter: " << id.c_str()
        << " in file: " << filepath_.c_str();
    const cv::FileNode& file_handle_2 = file_handle[id_2];
    if (file_handle_2.type() == cv::FileNode::NONE) {
      *output = default_value;
    } else {
      T v;
      file_handle_2 >> v;
      file_handle_2 >> *output;
    }
  }

 private:
  void openFile(const std::string& filepath, cv::FileStorage* fs) const {
    CHECK(!filepath.empty()) << "Empty filepath!";
    try {
      CHECK_NOTNULL(fs)->open(filepath, cv::FileStorage::READ);
    } catch (cv::Exception& e) {
      DYNO_THROW_MSG(DynosamException)
          << "Cannot open file: " << filepath << "\n"
          << "OpenCV error code: " + e.msg
          << " (remember that the first line should be: %YAML:1.0)!";
    }
    LOG_IF(FATAL, !fs->isOpened())
        << "Cannot open file in parseYAML: " << filepath
        << " (remember that the first line should be: %YAML:1.0)";
  }

  inline void closeFile(cv::FileStorage* fs) const {
    CHECK_NOTNULL(fs)->release();
  }

 private:
  cv::FileStorage fs_;
  std::string filepath_;
};

}  // namespace dyno
