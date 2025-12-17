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

/**
 * @brief Timing logger that allows for scoped timing and manual (ie.
 * start/stop) and access (ie. get the delta without logging) functionalities.
 *
 * Class uses a TimingGenerator to generate the start and stop comparison times
 * as this may be different depending on what is being timed (ie. CPU vs GPU
 * timing). TimingGenerator class must implement void onStart(); void onStop();
 *  double calcDelta() const;
 *
 * Functions where calcDelta() returns the dt in nanoseconds.
 *
 * @tparam TIMING_GENERATOR
 */
template <typename TIMING_GENERATOR>
class BaseTimingStatsCollector {
 public:
  using TimingGenerator = TIMING_GENERATOR;

  DYNO_POINTER_TYPEDEFS(BaseTimingStatsCollector)

 protected:
  BaseTimingStatsCollector(const TimingGenerator& timing_generator,
                           const std::string& tag, int glog_level,
                           bool construct_stopped)
      : timing_generator_(timing_generator),
        tag_(tag + " [ms]"),
        glog_level_(glog_level),
        is_timing_(false) {
    if (!construct_stopped) {
      start();
    }
  }

 public:
  ~BaseTimingStatsCollector() { stop(); }

  void start() {
    timing_generator_.onStart();
    is_timing_ = true;
  }

  void stop() {
    if (isTiming() && shouldGlog()) {
      log();
    }

    timing_generator_.onStop();
    is_timing_ = false;
  }

  bool isTiming() const { return is_timing_; }
  void discardTiming() { is_timing_ = false; }

  // Time delta (dt) in nano seconds
  inline double delta() const { return timing_generator_.calcDelta(); }

  // not that it will log to glog, but that the glog verbosity level is set
  // such that it will log to the collector
  bool shouldGlog() const {
    if (glog_level_ == 0) {
      return true;
    }
    return VLOG_IS_ON(glog_level_);
  }

 private:
  /**
   * @brief Creates a toc time to compare against the latest tic time and logs
   * the diff as sample to the collector
   *
   * Only logs if is_valid_ == true, after which is_valid will be set to false.
   * The collector then needs to be reset to be used again
   *
   */
  void log() {
    const double nanoseconds = delta();
    const double milliseconds = nanoseconds / 1e6;

    if (!collector_) {
      collector_ = std::make_unique<StatsCollector>(tag_);
    }
    collector_->AddSample(milliseconds);
  }

 private:
  TimingGenerator timing_generator_;
  const std::string tag_;
  const int glog_level_;
  //! Timing state
  std::atomic_bool is_timing_;
  //! Internal logger.
  //! Created only first time logging ocurs to ensure the tag only appears if
  //! the timing actually logs and not just if it is instantiated.
  std::unique_ptr<StatsCollector> collector_;
};

struct ChronoTimeGenerator {
  //! Comparison time
  Timer::TimePoint tic_time_;
  void onStart();
  void onStop();

  double calcDelta() const;
};

class ChronoTimingStats : public BaseTimingStatsCollector<ChronoTimeGenerator> {
 public:
  using This = BaseTimingStatsCollector<ChronoTimeGenerator>;
  ChronoTimingStats(const std::string& tag, int glog_level = 0,
                    bool construct_stopped = false);
};

}  // namespace utils
}  // namespace dyno
