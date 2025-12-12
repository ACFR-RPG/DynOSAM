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

#include "dynosam_common/utils/TimingStats.hpp"

#include <glog/vlog_is_on.h>

namespace dyno {
namespace utils {

// std::ostream& operator<<(std::ostream& os, const TimingStatsNamespace&
// stats_namespace) {
//     os << (std::string)stats_namespace;
//     return os;
// }

TimingStatsCollector::TimingStatsCollector(const std::string& tag,
                                           int glog_level,
                                           bool construct_stopped)
    : tag_(tag + " [ms]"),
      glog_level_(glog_level),
      tic_time_(Timer::tic()),
      is_timing_(false) {
  if (!construct_stopped) {
    start();
  }
}

TimingStatsCollector::~TimingStatsCollector() { stop(); }

void TimingStatsCollector::start() {
  tic_time_ = Timer::tic();
  is_timing_ = true;
}

void TimingStatsCollector::stop() {
  if (isTiming() && shouldGlog()) {
    log();
  }

  tic_time_ = Timer::tic();
  is_timing_ = false;
}

bool TimingStatsCollector::isTiming() const { return is_timing_; }

bool TimingStatsCollector::shouldGlog() const {
  if (glog_level_ == 0) {
    return true;
  }

  return VLOG_IS_ON(glog_level_);
}

void TimingStatsCollector::discardTiming() { is_timing_ = false; }

double TimingStatsCollector::delta() const {
  const auto toc = Timer::toc<std::chrono::nanoseconds>(tic_time_);
  return static_cast<double>(toc.count());
}

void TimingStatsCollector::log() {
  const double nanoseconds = delta();
  const double milliseconds = nanoseconds / 1e6;

  if (!collector_) {
    collector_ = std::make_unique<StatsCollector>(tag_);
  }
  collector_->AddSample(milliseconds);
}

}  // namespace utils
}  // namespace dyno
