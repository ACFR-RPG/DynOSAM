/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_common/logger/Logger.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <filesystem>

#include "dynosam_common/utils/Statistics.hpp"

DEFINE_string(output_path, "./", "Path where to store dynosam's log output.");

namespace dyno {

namespace fs = std::filesystem;

std::string getOutputFilePath(const std::string& file_name) {
  fs::path out_path(FLAGS_output_path);
  return out_path / file_name;
}

void writeStatisticsSamplesToFile(const std::string& file_name) {
  utils::Statistics::WriteAllSamplesToCsvFile(getOutputFilePath(file_name));
}

void writeStatisticsSummaryToFile(const std::string& file_name) {
  utils::Statistics::WriteSummaryToCsvFile(getOutputFilePath(file_name));
}

void writeStatisticsModuleSummariesToFile() {
  utils::Statistics::WritePerModuleSummariesToCsvFile(getOutputFilePath(""));
}

bool createDirectory(const std::string& path) {
  std::filesystem::path dir_path(path);

  // Check if directory already exists
  if (std::filesystem::exists(dir_path)) {
    if (std::filesystem::is_directory(dir_path)) {
      VLOG(10) << "Directory already exists: " << path;
      return true;
    } else {
      VLOG(10) << "Path exists but is not a directory: " << path;
      return false;
    }
  }

  // Create directory if it doesn't exist
  if (std::filesystem::create_directories(dir_path)) {
    VLOG(10) << "Directory created: " << path;
    return true;
  } else {
    LOG(WARNING) << "Failed to create directory: " << path;
    return false;
  }
}

// This constructor will directly open the log file when called.
OfstreamWrapper::OfstreamWrapper(const std::string& filename,
                                 const bool& open_file_in_append_mode)
    : OfstreamWrapper(filename, FLAGS_output_path, open_file_in_append_mode) {}

OfstreamWrapper::OfstreamWrapper(const std::string& filename,
                                 const std::string& output_path,
                                 const bool& open_file_in_append_mode)
    : filename_(filename), output_path_(output_path) {
  openLogFile(open_file_in_append_mode);
}

// This destructor will directly close the log file when the wrapper is
// destructed. So no need to explicitly call .close();
OfstreamWrapper::~OfstreamWrapper() {
  VLOG(20) << "Closing output file: " << filename_.c_str();
  ofstream_.flush();
  ofstream_.close();
}

void OfstreamWrapper::closeAndOpenLogFile() {
  ofstream_.flush();
  ofstream_.close();
  CHECK(!filename_.empty());
  OpenFile(output_path_ + '/' + filename_, &ofstream_, false);
}

bool OfstreamWrapper::WriteOutCsvWriter(const CsvWriter& csv,
                                        const std::string& filename) {
  // set append mode to false as we never want to write over the top of a csv
  // file as this will upset the header
  OfstreamWrapper ofsw(filename, false);
  return csv.write(ofsw.ofstream_);
}

void OfstreamWrapper::openLogFile(bool open_file_in_append_mode) {
  CHECK(!filename_.empty());
  CHECK(!output_path_.empty());
  LOG(INFO) << "Opening output file: " << filename_.c_str();

  try {
    const std::string full_file_path = this->getFilePath();
    OpenFile(full_file_path, &ofstream_, open_file_in_append_mode);
  } catch (const DynosamException& e) {
    // log or fail silently
    if (VLOG_IS_ON(10)) {
      LOG(ERROR) << e.what();
    }
  }
}

fs::path OfstreamWrapper::getFilePath() const {
  fs::path fs_out_path(output_path_);
  if (!fs::exists(fs_out_path)) {
    DYNO_THROW_MSG(DynosamException)
        << "Cannot log to file " << filename_ << " as output file path "
        << fs_out_path << " does not exist!";
    throw;
  }
  return fs_out_path / fs::path(filename_);
}

CsvWriterToFile::CsvWriterToFile(const std::string& file,
                                 const CsvHeader& header,
                                 const std::string& seperator)
    : CsvWriter(header, seperator), file_(file) {}

CsvWriterToFile::~CsvWriterToFile() {
  OfstreamWrapper::WriteOutCsvWriter(*this, file_);
}

}  // namespace dyno
