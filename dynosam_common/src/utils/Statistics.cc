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

/********************************************************************************
 Copyright 2017 Autonomous Systems Lab, ETH Zurich, Switzerland

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*********************************************************************************/

/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   Statistics.cpp
 * @brief  For logging statistics in a thread-safe manner.
 * @author Antoni Rosinol
 */

#include "dynosam_common/utils/Statistics.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>

#include "dynosam_common/utils/CsvParser.hpp"

namespace dyno {

namespace fs = std::filesystem;

namespace utils {

Statistics& Statistics::Instance() {
  static Statistics instance;
  return instance;
}

Statistics::Statistics() : max_tag_length_(0) {}

Statistics::~Statistics() {}

// Static functions to query the stats collectors:
size_t Statistics::GetHandle(std::string const& tag) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  // Search for an existing tag.
  map_t::iterator i = Instance().tag_map_.find(tag);
  if (i == Instance().tag_map_.end()) {
    // If it is not there, create a tag.
    size_t handle = Instance().stats_collectors_.size();
    Instance().tag_map_[tag] = handle;
    Instance().stats_collectors_.push_back(StatisticsMapValue());
    // Track the maximum tag length to help printing a table of values later.
    Instance().max_tag_length_ =
        std::max(Instance().max_tag_length_, tag.size());
    return handle;
  } else {
    return i->second;
  }
}

// Return true if a handle has been initialized for a specific tag.
// In contrast to GetHandle(), this allows testing for existence without
// modifying the tag/handle map.
bool Statistics::HasHandle(std::string const& tag) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  bool tag_found = Instance().tag_map_.count(tag);
  return tag_found;
}

std::string Statistics::GetTag(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  std::string tag;
  // Perform a linear search for the tag.
  for (typename map_t::value_type current_tag : Instance().tag_map_) {
    if (current_tag.second == handle) {
      return current_tag.first;
    }
  }
  return tag;
}

StatsCollectorImpl::StatsCollectorImpl(size_t handle) : handle_(handle) {}

StatsCollectorImpl::StatsCollectorImpl(std::string const& tag)
    : handle_(Statistics::GetHandle(tag)) {}

size_t StatsCollectorImpl::GetHandle() const { return handle_; }
void StatsCollectorImpl::AddSample(double sample) const {
  Statistics::Instance().AddSample(handle_, sample);
}
void StatsCollectorImpl::IncrementOne() const {
  Statistics::Instance().AddSample(handle_, 1.0);
}

std::vector<std::string> Statistics::getTagByModule(
    std::string const& query_module) {
  std::vector<std::string> modules;
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  for (const auto& [tag, _] : GetStatsCollectors()) {
    std::optional<std::string> module = getModuleNameFromTag(tag);

    // if query module is empty match with those tags in the global namespace
    if (query_module.empty() && !module) {
      modules.push_back(tag);
    } else if (module && *module == query_module) {
      modules.push_back(tag);
    }
  }
  return modules;
}

void Statistics::AddSample(size_t handle, double seconds) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  stats_collectors_[handle].AddValue(seconds);
}
double Statistics::GetLastValue(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].GetLastValue();
}
double Statistics::GetLastValue(std::string const& tag) {
  return GetLastValue(GetHandle(tag));
}
double Statistics::GetTotal(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].Sum();
}
double Statistics::GetTotal(std::string const& tag) {
  return GetTotal(GetHandle(tag));
}
double Statistics::GetMean(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].Mean();
}
double Statistics::GetMean(std::string const& tag) {
  return GetMean(GetHandle(tag));
}
size_t Statistics::GetNumSamples(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].TotalSamples();
}
size_t Statistics::GetNumSamples(std::string const& tag) {
  return GetNumSamples(GetHandle(tag));
}
std::vector<double> Statistics::GetAllSamples(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].GetAllValues();
}
std::vector<double> Statistics::GetAllSamples(std::string const& tag) {
  return GetAllSamples(GetHandle(tag));
}
double Statistics::GetVariance(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].LazyVariance();
}
double Statistics::GetVariance(std::string const& tag) {
  return GetVariance(GetHandle(tag));
}
double Statistics::GetMin(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].Min();
}
double Statistics::GetMin(std::string const& tag) {
  return GetMin(GetHandle(tag));
}
double Statistics::GetMax(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].Max();
}
double Statistics::GetMax(std::string const& tag) {
  return GetMax(GetHandle(tag));
}
double Statistics::GetMedian(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].Median();
}
double Statistics::GetMedian(std::string const& tag) {
  return GetMedian(GetHandle(tag));
}
double Statistics::GetQ1(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].Q1();
}
double Statistics::GetQ1(std::string const& tag) {
  return GetQ1(GetHandle(tag));
}
double Statistics::GetQ3(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].Q1();
}
double Statistics::GetQ3(std::string const& tag) {
  return GetQ3(GetHandle(tag));
}
double Statistics::GetHz(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].MeanCallsPerSec();
}
double Statistics::GetHz(std::string const& tag) {
  return GetHz(GetHandle(tag));
}

// Delta time statistics.
double Statistics::GetMeanDeltaTime(std::string const& tag) {
  return GetMeanDeltaTime(GetHandle(tag));
}
double Statistics::GetMeanDeltaTime(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].MeanDeltaTime();
}
double Statistics::GetMaxDeltaTime(std::string const& tag) {
  return GetMaxDeltaTime(GetHandle(tag));
}
double Statistics::GetMaxDeltaTime(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].MaxDeltaTime();
}
double Statistics::GetMinDeltaTime(std::string const& tag) {
  return GetMinDeltaTime(GetHandle(tag));
}
double Statistics::GetMinDeltaTime(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].MinDeltaTime();
}
double Statistics::GetLastDeltaTime(std::string const& tag) {
  return GetLastDeltaTime(GetHandle(tag));
}
double Statistics::GetLastDeltaTime(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].GetLastDeltaTime();
}
double Statistics::GetVarianceDeltaTime(std::string const& tag) {
  return GetVarianceDeltaTime(GetHandle(tag));
}
double Statistics::GetVarianceDeltaTime(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].LazyVarianceDeltaTime();
}

std::string Statistics::SecondsToTimeString(double seconds) {
  double secs = fmod(seconds, 60);
  int minutes = (seconds / 60);
  int hours = (seconds / 3600);
  minutes = minutes - (hours * 60);

  char buffer[256];
  snprintf(buffer, sizeof(buffer),
#ifdef SM_TIMING_SHOW_HOURS
           "%02d:"
#endif
#ifdef SM_TIMING_SHOW_MINUTES
           "%02d:"
#endif
           "%09.6f",
#ifdef SM_TIMING_SHOW_HOURS
           hours,
#endif
#ifdef SM_TIMING_SHOW_MINUTES
           minutes,
#endif
           secs);
  return buffer;
}

void Statistics::Print(std::ostream& out) {  // NOLINT
  const map_t& tag_map = Instance().tag_map_;

  if (tag_map.empty()) {
    return;
  }

  out << "Statistics\n";

  out.width((std::streamsize)Instance().max_tag_length_);
  out.setf(std::ios::left, std::ios::adjustfield);
  out << "-----------";
  out.width(7);
  out.setf(std::ios::right, std::ios::adjustfield);
  out << "#\t";
  out << "Log Hz\t";
  out << "{avg     +- std    }\t";
  out << "[min,max]\n";

  for (const typename map_t::value_type& t : tag_map) {
    size_t i = t.second;
    out.width((std::streamsize)Instance().max_tag_length_);
    out.setf(std::ios::left, std::ios::adjustfield);
    // Print Name of tag
    out << t.first << "\t";

    out.setf(std::ios::right, std::ios::adjustfield);
    // Print #
    out << std::setw(5) << GetNumSamples(i) << "\t";
    if (GetNumSamples(i) > 0) {
      out << std::showpoint << GetHz(i) << "\t";
      double mean = GetMean(i);
      double stddev = sqrt(GetVariance(i));
      out << "{" << std::showpoint << mean;
      out << " +- ";
      out << std::showpoint << stddev << "}\t";

      double min_value = GetMin(i);
      double max_value = GetMax(i);

      // out.width(5);
      out << std::noshowpoint << "[" << min_value << "," << max_value << "]";
    }
    out << std::endl;
  }
}

void Statistics::WriteAllSamplesToCsvFile(const std::string& path) {
  const map_t& tag_map = Instance().tag_map_;
  if (tag_map.empty()) {
    return;
  }

  static const CsvHeader header("label", "samples");
  CsvWriter writer(header);

  VLOG(1) << "Writing statistics to file: " << path;
  for (const map_t::value_type& tag : tag_map) {
    const size_t& index = tag.second;
    if (GetNumSamples(index) > 0) {
      const std::string& label = tag.first;

      // // Add header to csv file. tag.first is the stats label.
      writer << label;

      // Each row has all samples.
      std::stringstream ss;
      const std::vector<double>& samples = GetAllSamples(index);
      for (const auto& sample : samples) {
        // make space separated so not to confuse the CsvWriter
        // treat all samples as a single column
        ss << ' ' << sample;
      }
      writer << ss.str();
    }
  }

  writer.write(path);
}

void Statistics::WriteSummaryToCsvFile(const std::string& path) {
  const map_t& tag_map = Instance().tag_map_;
  if (tag_map.empty()) {
    return;
  }

  static const CsvHeader header("label", "num samples", "log Hz", "mean",
                                "stddev", "min", "max");
  CsvWriter writer(header);

  for (const map_t::value_type& tag : tag_map) {
    SummaryWriterHelper(writer, tag);
  }

  writer.write(path);
}

void Statistics::WriteToYamlFile(const std::string& path) {
  std::ofstream output_file(path);

  if (!output_file) {
    LOG(ERROR) << "Could not write statistics: Unable to open file: " << path;
    return;
  }

  const map_t& tag_map = Instance().tag_map_;
  if (tag_map.empty()) {
    return;
  }

  VLOG(1) << "Writing statistics to file: " << path;
  for (const map_t::value_type& tag : tag_map) {
    const size_t index = tag.second;

    if (GetNumSamples(index) > 0) {
      std::string label = tag.first;

      // We do not want colons or hashes in a label, as they might interfere
      // with reading the yaml later.
      std::replace(label.begin(), label.end(), ':', '_');
      std::replace(label.begin(), label.end(), '#', '_');

      output_file << label << ":\n";
      output_file << "  samples: " << GetNumSamples(index) << "\n";
      output_file << "  mean: " << GetMean(index) << "\n";
      output_file << "  stddev: " << sqrt(GetVariance(index)) << "\n";
      output_file << "  min: " << GetMin(index) << "\n";
      output_file << "  max: " << GetMax(index) << "\n";
      output_file << "  median: " << GetMedian(index) << "\n";
      output_file << "  q1: " << GetQ1(index) << "\n";
      output_file << "  q3: " << GetQ3(index) << "\n";
    }
    output_file << "\n";
  }
}

void Statistics::WritePerModuleSummariesToCsvFile(
    const std::string& folder_path) {
  const map_t& tag_map = Instance().tag_map_;
  if (tag_map.empty()) {
    return;
  }

  static const CsvHeader header("label", "num samples", "log Hz", "mean",
                                "stddev", "min", "max");
  std::map<std::string, CsvWriter::Ptr> module_to_writer;

  CsvWriter global_writer(header);

  for (const map_t::value_type& tag : tag_map) {
    std::optional<std::string> module_name = getModuleNameFromTag(tag.first);

    // writer to use
    CsvWriter* writer;
    if (module_name) {
      // tag is not in global namespace so use the specificied namespace
      // if we dont have a writer for this module, make one
      if (module_to_writer.find(*module_name) == module_to_writer.end()) {
        module_to_writer[*module_name] = std::make_shared<CsvWriter>(header);
      }

      // set writer to use
      writer = module_to_writer.at(*module_name).get();
    } else {
      // no module name so use the global namespace
      writer = &global_writer;
    }
    CHECK_NOTNULL(writer);
    SummaryWriterHelper(*writer, tag);
  }

  // write out all summaries
  for (auto& [module_name, writer] : module_to_writer) {
    std::string full_path =
        fs::path(folder_path) / std::string("stats_" + module_name + ".csv");
    LOG(INFO) << "Writing out csv stats module: " << full_path;
    writer->write(full_path);
  }

  {
    std::string full_path = fs::path(folder_path) / "stats_global.csv";
    LOG(INFO) << "Writing out csv stats module: " << full_path;
    global_writer.write(full_path);
  }
}

std::string Statistics::Print() {
  std::stringstream ss;
  Print(ss);
  return ss.str();
}

void Statistics::Reset() {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  Instance().tag_map_.clear();
}

std::optional<std::string> Statistics::getModuleNameFromTag(
    const std::string& tag) {
  size_t pos = tag.find('.');
  if (pos != std::string::npos) {
    return tag.substr(0, pos);
  }
  return {};
}

}  // namespace utils

}  // namespace dyno
