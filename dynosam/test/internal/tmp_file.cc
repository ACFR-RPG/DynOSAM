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

#include "tmp_file.hpp"

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS

#include <glog/logging.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>

namespace dyno_testing {

TempFile::TempFile() {
  relative_path = boost::filesystem::unique_path("%%%%_%%%%_%%%%_%%%%");
  // absolute_path = boost::filesystem::temp_directory_path() / relative_path;
  // //opening temp file does not actually create a file..?
  absolute_path = boost::filesystem::current_path() / relative_path;

  write_stream.open(getFilePath());

  write_stream.close();
}

TempFile::~TempFile() {
  if (write_stream.is_open()) {
    write_stream.close();
  }

  if (read_stream.is_open()) {
    read_stream.close();
  }

  boost::filesystem::remove(absolute_path);
}

std::string TempFile::readLine() const {
  const std::lock_guard<std::mutex> lg(io_mutex);
  read_stream.open(getFilePath());

  // update the read ptr to the last position
  // TODO:(jesse) ensure the buf is no greater than the length of the file
  std::filebuf* pbuf = read_stream.rdbuf();
  pbuf->pubseekpos(current_pos);

  std::string line;
  std::getline(read_stream, line);

  // set current position to the pos in the file for next read
  current_pos = pbuf->pubseekoff(0, read_stream.cur);

  // if(static_cast<uint8_t>(read_stream.peek()) != 0) {
  //     line += "\n";
  // }

  read_stream.close();
  return line;
}
void TempFile::write(const std::string& data) {
  const std::lock_guard<std::mutex> lg(io_mutex);

  write_stream.open(getFilePath(), std::ios::app);

  write_stream << data << std::endl;
  write_stream.close();
}

}  // namespace dyno_testing
