#include "dynosam/backend/PoseChangeBackendPipeline.hpp"

#include "dynosam_common/utils/SafeCast.hpp"

namespace dyno {

PoseChangeBackendPipeline::PoseChangeBackendPipeline(
    InputQueue* input_queue, PoseChangeVIBackendModule::Ptr module)
    : Base("pose-change-vi-pipeline", input_queue, module) {}

PoseChangeInput::ConstPtr PoseChangeBackendPipeline::getInputPacket() {
  // arguable all backends should be batched....
  std::vector<PoseChangeInput::ConstPtr> batch_input;
  bool queue_state;
  if (parallel_run_) {
    queue_state = input_queue->popAllBlocking(batch_input);
  } else {
    queue_state = input_queue->popAll(batch_input);
  }

  if (queue_state) {
    BatchPoseChangeInput::Builder builder(batch_input.size());
    for (const auto& input : batch_input) {
      // expect all input from the frontend to be SinglePoseChangeInput
      SinglePoseChangeInput::ConstPtr single_input =
          safeCast<PoseChangeInput, SinglePoseChangeInput>(input);
      CHECK_NOTNULL(single_input);
      builder.add(single_input);
    }

    return builder.finalise();
  } else {
    return nullptr;
  }
}

}  // namespace dyno
