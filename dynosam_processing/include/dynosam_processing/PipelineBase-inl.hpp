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

#include "dynosam_common/utils/Timing.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_processing/PipelineBase.hpp"

namespace dyno {

template <typename INPUT, typename OUTPUT>
PipelineBase::ReturnCode PipelineModule<INPUT, OUTPUT>::spinOnce() {
  using ReturnCode = PipelineBase::ReturnCode;

  if (isShutdown()) {
    return ReturnCode::IS_SHUTDOWN;
  }

  is_thread_working_ = false;
  InputConstSharedPtr input = getInputPacket();
  is_thread_working_ = true;

  if (!input) {
    is_thread_working_ = false;
    return ReturnCode::GET_PACKET_FAILURE;
  }

  OutputConstSharedPtr output{nullptr};
  try {
    output = process(input);
  } catch (const std::exception& e) {
    VLOG(10) << "Module " << this->moduleName()
             << " caught exception during processing. Shutting down!";
    this->shutdown();
    this->emitOnExceptionCallbacks(
        ExceptionShutdownContext{e, this->moduleName()});
    return ReturnCode::PROCESSING_FAILURE;
  }

  if (!output) {
    is_thread_working_ = false;
    return ReturnCode::PROCESSING_FAILURE;
  }

  if (!pushOutputPacket(output)) {
    LOG_EVERY_N(WARNING, 100)
        << "Module: " << this->moduleName() << " - Output push failed.";

    is_thread_working_ = false;
    return ReturnCode::OUTPUT_PUSH_FAILURE;
  }

  emitProcessCallbacks(input, output);

  is_thread_working_ = false;
  return ReturnCode::SUCCESS;
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
void MIMOPipelineModule<INPUT, OUTPUT>::registerOutputCallback(
    OutputCallback output_callback) {
  CHECK(output_callback);
  output_callbacks_.push_back(output_callback);
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
  for (OutputCallback cb : output_callbacks_) {
    cb(output_packet);
  }

  static constexpr auto kTimeLimitCallbacks = std::chrono::milliseconds(10);
  auto callbacks_duration = utils::Timer::toc(tic_callbacks);
  LOG_IF(WARNING, callbacks_duration > kTimeLimitCallbacks)
      << "Pushing output packet to queues for module: " << this->moduleName()
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
    LOG_IF_EVERY_N(WARNING, !Base::isShutdown() && !hasWork() && VLOG_IS_ON(50),
                   100)
        << "Module: " << Base::moduleName() << " - didn't return an output.";
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
