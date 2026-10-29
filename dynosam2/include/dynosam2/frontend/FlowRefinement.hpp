#pragma once

#include "dynosam2/common/Parameters.hpp"
#include "dynosam2/common/Types.hpp"
#include "dynosam2/frontend/FeatureBlockContainer.hpp"
#include "dynosam2/frontend/Frame.hpp"
#include "dynosam2/frontend/MatchingAdaptor.hpp"
#include "dynosam_common/SensorModels.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/ImageContainer.hpp"

namespace dyno2 {

struct FlowRefinementParams {
  double flow_sigma{10.0};
  double flow_prior_sigma{3.33};
  double k_huber{0.001};
  bool use_robust{true};
};

struct FlowRefinementResult {};

// TODO: draw pre and post flow!
class FlowRefinement {
 public:
  FlowRefinement(const Camera::Ptr camera, const ImageContainer& images,
                 FrameAbsoluteAdaptor& adaptor,
                 const FlowRefinementParams& params = {});
  ~FlowRefinement();

  // will update the pixel values in the local geometry but NOT the 3d geometry
  // will mark values in features as outliers but not update the memory space!
  void refine(const gtsam::Pose3& pose_in, dyno::Pose3Measurement& pose_out);

 private:
  ImageContainer images_;
  FrameAbsoluteAdaptor& adaptor_;
  const FlowRefinementParams params_;

  struct ImplOptimizer;
  std::unique_ptr<ImplOptimizer> impl_;
};

}  // namespace dyno2
