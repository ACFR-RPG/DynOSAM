/*
 *   Copyright (c) 2023 Jesse Morris (jesse.morris@sydney.edu.au)
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

#include <chrono>
#include <memory>

namespace dyno {
namespace utils {
class Timer {
 public:
  /// @brief Alias for high resolution clock
  using Clock = std::chrono::high_resolution_clock;
  /// @brief Alias for time point as defined by Timer::Clock
  using TimePoint = Clock::time_point;

  static TimePoint tic() { return Clock::now(); }

  // Stop timer and report duration in given time.
  // Returns duration in milliseconds by default.
  // call .count() on returned duration to have number of ticks.
  template <typename T = std::chrono::milliseconds>
  static T toc(const TimePoint& start) {
    return std::chrono::duration_cast<T>(Clock::now() - start);
  }

  /**
   * @brief Time in seconds since epoch. Uses
   * std::chrono::high_resolution_clock::now().
   *
   * @return double Time in seconds
   */
  static double now() {
    auto t_now = Clock::now().time_since_epoch();
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(t_now)
                   .count()) /
           1e9;
  }

  template <typename U, typename T = std::chrono::milliseconds>
  static double toUnits(const T& time) {
    const auto unit = std::chrono::duration_cast<U>(time);
    return static_cast<double>(unit.count());
  }

  template <typename T = std::chrono::milliseconds>
  static double toSeconds(const T& time) {
    return toUnits<std::chrono::seconds>(time);
  }
};

// Usage: measure<>::execution(function, arguments)
template <typename T = std::chrono::milliseconds>
struct Measure {
  template <typename F, typename... Args>
  static typename T::rep execution(F&& func, Args&&... args) {
    auto start = std::chrono::steady_clock::now();
    std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
    auto duration =
        std::chrono::duration_cast<T>(std::chrono::steady_clock::now() - start);
    return duration.count();
  }
};

}  // namespace utils
}  // namespace dyno
