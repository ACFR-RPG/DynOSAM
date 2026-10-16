#pragma once

#include <string>

namespace dyno2 {

struct EstimationParameters {};

struct FrontendParameters {
  bool prefer_provided_object_detection{true};
  std::string detection_model{"yolov8n-seg.pt"};

  struct Tracker {
    //! Maximum number of keypoints to detect
    size_t max_nr_kepoints_extaction;
    //! Desired number of features to track
    size_t max_nr_desired_tracks;
    //! Minimum number of tracks before redetection
    size_t min_nr_tracks;
    size_t min_distance_btw_tracked_and_detected;
  };

  Tracker static_tracks{800, 500, 400, 8};
  // TODO: most of this should really be a function of the object SIZE
  //  since we dont want to try and re-detect ALL the time if the object is too
  //  small to even get 300 features (this)
  Tracker dynamic_tracks{350, 300, 10, 3};

  // struct GeometricSolve {
  //   bool refine_with_flow{true};
  //   bool use_robust_loss{true};
  // };

  // GeometricSolve static_geometric_solve;
  // GeometricSolve dynamic_geometric_solve;
};

struct PipelineParameters {
  FrontendParameters frontend;
};

}  // namespace dyno2
