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

namespace dyno {
namespace utils {

// std::ostream& operator<<(std::ostream& os, const TimingStatsNamespace&
// stats_namespace) {
//     os << (std::string)stats_namespace;
//     return os;
// }

TimingStatsCollector::TimingStatsCollector(const std::string& tag)
    : tic_time_(Timer::tic()), collector_(tag + " [ms]") {}

TimingStatsCollector::~TimingStatsCollector() { tocAndLog(); }

void TimingStatsCollector::reset() {
  tic_time_ = Timer::tic();
  is_valid_ = true;
}

bool TimingStatsCollector::isValid() const { return is_valid_; }

void TimingStatsCollector::tocAndLog() {
  if (is_valid_) {
    auto toc = Timer::toc<std::chrono::nanoseconds>(tic_time_);
    auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(toc);
    collector_.AddSample(static_cast<double>(milliseconds.count()));
    is_valid_ = false;
  }
}

}  // namespace utils
}  // namespace dyno
