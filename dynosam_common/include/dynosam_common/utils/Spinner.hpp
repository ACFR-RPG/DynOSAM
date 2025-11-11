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
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "dynosam_common/utils/Macros.hpp"

namespace dyno {

/**
 * @brief Class to handle the calling of a function within a thread.
 * The spinner launches its own thread for each instance and is non-blocking.
 *
 * This version of the class expects the Evoke function to be within a while
 * loop itself as the class will simply bind this to a thread and execute it.
 * See LoopingSpinner to for a spinOnce style evoke function call.
 *
 */
class Spinner {
 public:
  DYNO_POINTER_TYPEDEFS(Spinner)
  using Evoke = std::function<void()>;

  /**
   * @brief Construct a new Spinner given a function which is blocking (ie. has
   * a loop within the function method itself)
   *
   * Name is just used as a logging identifier. If auso start is true, the
   * function will be envoked.
   *
   * @param func Evoke
   * @param name const std::string&
   * @param auto_start bool. Defaults to true
   */
  explicit Spinner(Evoke func, const std::string& name, bool auto_start = true);
  virtual ~Spinner();

  /**
   * @brief Starts the function in a thread. If the spinner has already been
   * started (i.e from auto_start), this function will do nothing. Non-blocking
   * call
   *
   */
  void start();

  /**
   * @brief Shutdown the spinner - once stopped cannot be restarted. Used to
   * ensure the thread closes safely. Be careful to avoid a race condition where
   * you try and shutdown the spinner but your functional loop has not exited
   * yet (depending on its condition) - in this case the shutdown function will
   * block indefinitely.
   *
   */
  virtual void shutdown();

  /**
   * @brief Checks if the thread is running.
   * As much thread safety has been used to ensure that this bool is accurate.
   *
   * @return true
   * @return false
   */
  inline bool isRunning() const { return is_running_; }

 private:
  Evoke func_;
  const std::string name_;

  std::unique_ptr<std::thread> spin_thread_{nullptr};
  std::atomic<bool> is_running_{false};
};

/**
 * @brief A spinner that will take a non-blocking function (spinOnce) and puts
 * it in a loop with a given delay. Similar to timer functionality but
 * guarantees that the function is called within a loop. Unlike a timer, the
 * class does not take into account how long the function takes to run as the
 * delay period will be used regardless
 *
 */
class LoopingSpinner : public Spinner {
 public:
  /**
   * @brief Construct a new Looping Spinner object with a spinOnce style
   * fucntion, name and period (i.e delay). The period is templated on
   * std::chrono::duration<DurationRepT, DurationT> so that chrono literals can
   * be used to specify the delay in timing.
   *
   * E.g to create a spinner that has a 1 second delay
   * constexpr static std::chrono::second kSecondDelay = 1s;
   * LoopingSpinner spinner(func, "spinner", kSecondDelay);
   *
   * @tparam DurationRepT Rep, an arithmetic type representing the number of
   * ticks. Defaults to std::int64_t.
   * @tparam DurationT Period (until C++17)typename Period::type (since C++17),
   * a std::ratio representing the tick period (i.e. the number of second's
   * fractions per tick). Defaults to std::milli.
   * @param func Evoke spinOnce style fucntion
   * @param name const std::string&
   * @param period std::chrono::duration<DurationRepT, DurationT> used to define
   * the delay
   */
  template <typename DurationRepT = std::int64_t,
            typename DurationT = std::milli>
  explicit LoopingSpinner(Evoke func, const std::string& name,
                          std::chrono::duration<DurationRepT, DurationT> period)
      : Spinner(std::bind(&LoopingSpinner::spin, this), name, false),
        spinOnce_(func),
        delay_(period) {
    start();
  }

  /**
   * @brief Shutsdown the thread and exists the loop.
   * In this case, as the class manages the loop, the function should exit
   * almost immediately.
   *
   */
  void shutdown() override;

 private:
  void spin();
  Evoke spinOnce_;
  std::atomic<bool> is_shutdown_{false};

  const std::chrono::milliseconds delay_;  //! constructed from the period
};

}  // namespace dyno
