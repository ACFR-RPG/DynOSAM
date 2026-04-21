#include "dynosam/backend/PoseChangeBackendModule.hpp"
#include "dynosam/pipeline/PipelineModuleProcessor.hpp"

namespace dyno {

class PoseChangeBackendPipeline
    : public PipelineModuleProcessor<PoseChangeInput, DynoState> {
 public:
  typedef PipelineModuleProcessor<PoseChangeInput, DynoState> Base;
  typedef Base::InputQueue InputQueue;

  PoseChangeBackendPipeline(InputQueue* input_queue,
                            PoseChangeVIBackendModule::Ptr module)
      : Base("pose-change-vi-pipeline", input_queue, module) {}

  PoseChangeInput::ConstPtr getInputPacket() override {
    // arguable all backends should be batched....
    std::vector<PoseChangeInput::ConstPtr> batch_input;
    bool queue_state;
    if (parallel_run_) {
      queue_state = input_queue->popAllBlocking(batch_input);
    } else {
      queue_state = input_queue->popAll(batch_input);
    }

    if (queue_state) {
      // TODO: get frame with latest timestamp
      PoseChangeInput::Ptr latest_pc_input =
          std::const_pointer_cast<PoseChangeInput>(batch_input.back());
      latest_pc_input->starting_frame_id = batch_input.front()->frame_id;

      // LOG(INFO) << "Constructing batch PC input from "
      //           << batch_input.front()->frame_id << " to "
      //           << latest_pc_input->frame_id;

      for (auto it = batch_input.begin(); it != batch_input.end() - 1; ++it) {
        auto pc_input = std::const_pointer_cast<PoseChangeInput>(*it);

        // ideally this should just an insert to ensure there are no duplicate
        // values
        latest_pc_input->new_values.insert_or_assign(pc_input->new_values);
        latest_pc_input->new_factors += pc_input->new_factors;
      }
      return latest_pc_input;

    } else {
      return nullptr;
    }
  };
};

}  // namespace dyno
