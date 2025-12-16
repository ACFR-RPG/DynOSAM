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

void ChronoTimeGenerator::onStart() { tic_time_ = Timer::tic(); }
void ChronoTimeGenerator::onStop() { tic_time_ = Timer::tic(); }

double ChronoTimeGenerator::calcDelta() const {
  const auto toc = Timer::toc<std::chrono::nanoseconds>(tic_time_);
  return static_cast<double>(toc.count());
}

ChronoTimingStats::ChronoTimingStats(const std::string& tag, int glog_level,
                                     bool construct_stopped)
    : This(ChronoTimeGenerator{}, tag, glog_level, construct_stopped) {}

}  // namespace utils
}  // namespace dyno
