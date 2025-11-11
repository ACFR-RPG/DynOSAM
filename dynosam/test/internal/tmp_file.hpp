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
#include <cstdio>
#include <cstdlib>
#include <iostream>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

namespace dyno_testing {
static constexpr std::string_view endl = "\n";

class TempFile {
 public:
  TempFile();
  ~TempFile();

  /**
   * @brief Get the absolute file path
   *
   * @return std::string
   */
  inline std::string getFilePath() const { return absolute_path.native(); }

  std::string readLine() const;
  void write(const std::string& data);

 private:
  boost::filesystem::path relative_path;
  boost::filesystem::path absolute_path;
  mutable std::ofstream write_stream;
  mutable std::ifstream read_stream;

  std::FILE* tmpf;
  mutable std::mutex io_mutex;
  mutable long current_pos = 0;
};

inline std::ostream& operator<<(std::ostream& os, const TempFile& tmp_file) {
  os << tmp_file.readLine();
  return os;
}

inline std::istream& operator>>(std::istream& is, TempFile& tmp_file) {
  std::string input_line;
  std::getline(is, input_line);
  tmp_file.write(input_line);
  return is;
}

inline TempFile& operator<<(TempFile& tmp_file, const std::string& data) {
  tmp_file.write(data);
  return tmp_file;
}

}  // namespace dyno_testing
