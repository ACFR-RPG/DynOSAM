#pragma once

#include "dynosam/backend/PoseChangeBackendModule.hpp"
#include "dynosam/pipeline/PipelineModuleProcessor.hpp"

namespace dyno {

/**
 * @brief Pipeline specifc to the PoseChangeBackendModule.
 *
 * Takes a batch of PoseChangeInput::ConstPtr from the input pipeline which are
 * all expected to be of derived SinglePoseChangeInput type and converts
 * them to a BatchPoseChangeInput which will then be processed
 * by the PoseChangeBackendModule.
 *
 * We take a batch of data so that the backend can incrementally process
 * all the data since the last update.
 *
 */
class PoseChangeBackendPipeline
    : public PipelineModuleProcessor<PoseChangeInput, DynoState> {
 public:
  typedef PipelineModuleProcessor<PoseChangeInput, DynoState> Base;
  typedef Base::InputQueue InputQueue;

  PoseChangeBackendPipeline(InputQueue* input_queue,
                            PoseChangeVIBackendModule::Ptr module);

 protected:
  PoseChangeInput::ConstPtr getInputPacket() override;
};

}  // namespace dyno
