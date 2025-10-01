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

#include <atomic>

#include "dynosam_common/utils/Statistics.hpp"
#include "dynosam_common/utils/Timing.hpp"

namespace dyno {
namespace utils {

// class TimingStatsNamespace {
// public:
//     template <typename... Args>
//     TimingStatsNamespace(const Args&... args) : result_(joinStrings(args...))
//     {}

//     operator std::string() const {
//         return result_;
//     }

//     TimingStatsNamespace& operator+=(const TimingStatsNamespace& rhs) {
//         //combine as namespace
//         result_ = joinStrings(result_, (std::string)rhs);
//         return *this;
//     }

//     TimingStatsNamespace& operator+=(const std::string& rhs) {
//         //combine as suffix
//         result_ += rhs;
//         return *this;
//     }

//     friend TimingStatsNamespace operator+(TimingStatsNamespace lhs, const
//     TimingStatsNamespace& rhs) {
//         lhs += rhs;
//         return lhs;
//     }

//     friend TimingStatsNamespace operator+(std::string lhs, const
//     TimingStatsNamespace& rhs) {
//         lhs += rhs;
//         return lhs;
//     }

// private:
//     template <typename T>
//     static std::string toString(const T& value) {
//         std::ostringstream oss;
//         oss << value;
//         return oss.str();
//     }

//     template <typename First, typename... Rest>
//     static std::string joinStrings(const First& first, const Rest&... rest) {
//         std::ostringstream oss;
//         oss << toString(first);

//         // Only add '.' if rest is not empty
//         if constexpr (sizeof...(rest) > 0) {
//             ((oss << '.' << toString(rest)), ...); // Fold expression for
//             parameter pack
//         }
//         return oss.str();
//     }

// private:
//     std::string result_;

// };

// std::ostream& operator<<(std::ostream& os, const TimingStatsNamespace&
// stats_namespace);

class TimingStatsCollector {
 public:
  DYNO_POINTER_TYPEDEFS(TimingStatsCollector)

  TimingStatsCollector(const std::string& tag);
  ~TimingStatsCollector();

  /**
   * @brief Resets the current tic_time and sets is_valid = true,
   * such that when the collector tries to toc, the tic (comparison time) will
   * be valid
   *
   *
   */
  void reset();

  bool isValid() const;

 private:
  /**
   * @brief Creates a toc time to compare against the latest tic time and logs
   * the diff as sample to the collector
   *
   * Only logs if is_valid_ == true, after which is_valid will be set to false.
   * The collector then needs to be reset to be used again
   *
   */
  void tocAndLog();

 private:
  Timer::TimePoint
      tic_time_;  //! Time on creation (the time to compare against)
  StatsCollector collector_;

  std::atomic_bool is_valid_{
      true};  //! Indiactes validity of the tic_time and if the collector has
              //! been reset allowing a new toc to be made
};

}  // namespace utils
}  // namespace dyno

#define TIMING_STATS(tag) \
  dyno::utils::TimingStatsCollector timing_stats_##tag(tag);
