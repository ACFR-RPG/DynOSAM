#pragma once

#include "dynosam2/DynamicSlamVisualizer.hpp"
#include "dynosam2/common/DynamicSlamInterface.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_processing/PipelineBase.hpp"

namespace dyno2 {

typedef dyno::SIMOPipelineModule<PublicationData, dyno::EmptyPayload>
    PublicationDataPipeline;
typedef PublicationDataPipeline::InputQueue PublicationDataQueue;

/// @brief Both the visualzier itself AND the Pipeline to recieve the data
class DynamicSlamVisualizerRosBase : public DynamicSlamVisualizer,
                                     public PublicationDataPipeline {
 public:
  DYNO_POINTER_TYPEDEFS(DynamicSlamVisualizerRosBase)
  DynamicSlamVisualizerRosBase(InputQueue* input_queue)
      : PublicationDataPipeline("dymosam2-viz", input_queue) {}

 private:
  inline dyno::EmptyPayload::ConstPtr process(
      const PublicationData::ConstPtr& input) override {
    // perform publishing
    this->publishEstimatorUpdate(*input);

    static const auto empty_payload = std::make_shared<dyno::EmptyPayload>();
    return empty_payload;
  }
};

class DynamicSlamVisualizerRos : public DynamicSlamVisualizerRosBase {
 public:
  DYNO_POINTER_TYPEDEFS(DynamicSlamVisualizerRos)
  DynamicSlamVisualizerRos(InputQueue* input_queue)
      : DynamicSlamVisualizerRosBase(input_queue) {}

 private:
  virtual void publishEstimatorUpdate(
      const PublicationData& publication_data) override {
    LOG(INFO) << "Here";
  }
};

}  // namespace dyno2
