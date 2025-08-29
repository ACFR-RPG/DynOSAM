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

#include "dynosam/pipeline/PipelineBase.hpp"
#include "dynosam/utils/Timing.hpp"
#include "dynosam/utils/TimingStats.hpp"

namespace dyno {

template <typename INPUT, typename OUTPUT>
PipelineBase::ReturnCode PipelineModule<INPUT, OUTPUT>::spinOnce() {
  using ReturnCode = PipelineBase::ReturnCode;

  if (isShutdown()) {
    return ReturnCode::IS_SHUTDOWN;
  }

  ReturnCode return_code;
  utils::TimingStatsCollector timing_stats(module_name_);

  InputConstSharedPtr input = nullptr;
  is_thread_working_ = false;
  input = getInputPacket();
  is_thread_working_ = true;

  if (input) {
    // Transfer the ownership of input to the actual pipeline module.
    // From this point on, you cannot use input, since process owns it.
    OutputConstSharedPtr output = nullptr;
    output = process(input);

    if (output) {
      // Received a valid output, send to output queue
      if (!pushOutputPacket(output)) {
        LOG_EVERY_N(WARNING, 100)
            << "Module: " << module_name_ << " - Output push failed.";
        is_thread_working_ = false;
        return_code = ReturnCode::OUTPUT_PUSH_FAILURE;
      } else {
        emitProcessCallbacks(input, output);
        is_thread_working_ = false;
        return_code = ReturnCode::SUCCESS;
      }
    } else {
      return_code = ReturnCode::PROCESSING_FAILURE;
      is_thread_working_ = false;
    }
  } else {
    is_thread_working_ = false;
    return_code = ReturnCode::GET_PACKET_FAILURE;
  }

  return return_code;
}

template <typename INPUT, typename OUTPUT>
void PipelineModule<INPUT, OUTPUT>::registerOnProcessCallback(
    const OnProcessCallback& callback) {
  CHECK(callback);
  on_process_callbacks_.push_back(callback);
}

template <typename INPUT, typename OUTPUT>
void PipelineModule<INPUT, OUTPUT>::emitProcessCallbacks(
    const InputConstSharedPtr& input_packet,
    const OutputConstSharedPtr& output_packet) {
  for (OnProcessCallback callbacks : on_process_callbacks_) {
    callbacks(input_packet, output_packet);
  }
}

// MIMOPipelineModule/////////////////

template <typename INPUT, typename OUTPUT>
void MIMOPipelineModule<INPUT, OUTPUT>::registerOutputQueue(
    OutputQueue* output_queue) {
  CHECK(output_queue);
  output_queues_.push_back(output_queue);
}

template <typename INPUT, typename OUTPUT>
bool MIMOPipelineModule<INPUT, OUTPUT>::pushOutputPacket(
    const typename Base::OutputConstSharedPtr& output_packet) const {
  auto tic_callbacks = utils::Timer::tic();
  //! We need to make our packet shared in order to send it to multiple
  //! other modules.
  //! Call all callbacks
  for (OutputQueue* queue : output_queues_) {
    CHECK(queue);
    queue->push(output_packet);
  }
  static constexpr auto kTimeLimitCallbacks = std::chrono::milliseconds(10);
  auto callbacks_duration = utils::Timer::toc(tic_callbacks);
  LOG_IF(WARNING, callbacks_duration > kTimeLimitCallbacks)
      << "Pushing output packet to queues for module: " << this->module_name_
      << " are taking very long! Current latency: "
      << callbacks_duration.count() << " ms.";
  return true;
}

// SIMOPipelineModule///////////////

template <typename INPUT, typename OUTPUT>
typename SIMOPipelineModule<INPUT, OUTPUT>::InputConstSharedPtr
SIMOPipelineModule<INPUT, OUTPUT>::getInputPacket() {
  typename Base::InputConstSharedPtr input = nullptr;

  bool queue_state;
  if (parallel_run_) {
    queue_state = input_queue->popBlocking(input);
  } else {
    queue_state = input_queue->pop(input);
  }

  if (queue_state) {
    return input;
  } else {
    // LOG_IF_EVERY_N(WARNING, !Base::isShutdown() && !hasWork(), 100)
    //     << "Module: " << Base::module_name_ << " - didn't return an output.";
    return nullptr;
  }
}

template <typename INPUT, typename OUTPUT>
void SIMOPipelineModule<INPUT, OUTPUT>::shutdownQueues() {
  input_queue->shutdown();
}

template <typename INPUT, typename OUTPUT>
bool SIMOPipelineModule<INPUT, OUTPUT>::hasWork() const {
  return !input_queue->isShutdown() && !input_queue->empty();
}

}  // namespace dyno
