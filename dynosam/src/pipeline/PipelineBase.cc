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

#include "dynosam/pipeline/PipelineBase.hpp"

#include <glog/logging.h>

#include <chrono>  // std::chrono::
#include <memory>
#include <thread>  // std::this_thread::sleep_for

#include "dynosam_common/utils/Timing.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno {

PipelineBase::TimingNaming::TimingNaming(const std::string& module_name)
    : module(module_name),
      input_packet_wrapped("get_input"),
      process_wrapped("process"),
      output_timing_wrapped("push_output") {}

utils::ChronoTimingStats PipelineBase::TimingNaming::inputPacketTimer() const {
  return utils::ChronoTimingStats{module + "." + input_packet_wrapped,
                                  intermediate_timing_glog_verbosity};
}
utils::ChronoTimingStats PipelineBase::TimingNaming::processTimer() const {
  return utils::ChronoTimingStats{module + "." + process_wrapped,
                                  intermediate_timing_glog_verbosity};
}
utils::ChronoTimingStats PipelineBase::TimingNaming::outputTiming() const {
  return utils::ChronoTimingStats{module + "." + output_timing_wrapped,
                                  intermediate_timing_glog_verbosity};
}

PipelineBase::SpinTimingStats PipelineBase::TimingNaming::constructTimingStats()
    const {
  // helper lambda to get and set statistics value
  auto set_mean_statistic =
      [](const std::string& tag) -> std::optional<double> {
    std::string used_tag = tag + " [ms]";
    std::optional<double> value;
    if (utils::Statistics::HasHandle(used_tag)) {
      value.emplace(utils::Statistics::GetMean(used_tag));
    }
    return value;
  };

  // helper lambda to get and set statistics value
  auto set_hz_statistic = [](const std::string& tag) -> std::optional<double> {
    std::string used_tag = tag + " [ms]";
    std::optional<double> value;
    if (utils::Statistics::HasHandle(used_tag)) {
      value.emplace(utils::Statistics::GetHz(used_tag));
    }
    return value;
  };

  SpinTimingStats timing_stats;
  timing_stats.spin_ms = set_mean_statistic(this->module).value_or(0.0);
  timing_stats.spin_hz = set_hz_statistic(this->module).value_or(0.0);

  timing_stats.get_input_packet_ms =
      set_mean_statistic(module + "." + input_packet_wrapped);
  timing_stats.process_ms = set_mean_statistic(module + "." + process_wrapped);
  timing_stats.push_packet_ms =
      set_mean_statistic(module + "." + output_timing_wrapped);

  return timing_stats;
}

bool PipelineBase::spin() {
  LOG(INFO) << "Starting module " << this->moduleName();
  while (!isShutdown()) {
    spinOnce();
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(5ns);  // give CPU thread some sleep
    //   time... //TODO: only if threaded?
  }
  return true;
}

void PipelineBase::shutdown() {
  LOG(INFO) << "Shutting down module " << this->moduleName();
  is_shutdown_ = true;
  shutdownQueues();
}

bool PipelineBase::isWorking() const { return is_thread_working_ || hasWork(); }

void PipelineBase::notifyFailures(const PipelineBase::ReturnCode& result) {
  for (const auto& failure_callbacks : on_failure_callbacks_) {
    if (failure_callbacks) {
      failure_callbacks(result);
    } else {
      LOG(ERROR) << "Invalid OnFailureCallback for module: "
                 << this->moduleName();
    }
  }
}

void PipelineBase::registerOnFailureCallback(
    const OnPipelineFailureCallback& callback_) {
  CHECK(callback_);
  on_failure_callbacks_.push_back(callback_);
}

}  // namespace dyno
