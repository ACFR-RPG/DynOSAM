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

#include "dynosam_common/utils/Spinner.hpp"

#include <glog/logging.h>

#include <chrono>

using namespace std::chrono_literals;

namespace dyno {

Spinner::Spinner(Evoke func, const std::string& name, bool auto_start)
    : func_(func), name_(name) {
  VLOG(10) << "Creating spinner - " << name;

  if (auto_start) {
    VLOG(10) << "Auto starting spinner - " << name;
    start();
  }
}

Spinner::~Spinner() { shutdown(); }

void Spinner::start() {
  if (!is_running_) {
    spin_thread_ = std::make_unique<std::thread>(func_);
    is_running_ = true;
  }
}

void Spinner::shutdown() {
  if (spin_thread_ && spin_thread_->joinable()) {
    VLOG(10) << "Spinner " << name_ << " shutting down";
    spin_thread_->join();
    is_running_ = false;
  } else {
    // VLOG(10) << "Spinner " << name_ << "already shutdown";
  }
}

void LoopingSpinner::shutdown() {
  is_shutdown_ = true;
  std::this_thread::sleep_for(100ms);
  Spinner::shutdown();
}

void LoopingSpinner::spin() {
  while (!is_shutdown_) {
    spinOnce_();
    std::this_thread::sleep_for(delay_);
  }
}

}  // namespace dyno
