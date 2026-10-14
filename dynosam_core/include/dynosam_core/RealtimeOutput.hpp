#pragma once

#include "dynosam_common/Types.hpp"
#include "dynosam_core/DynoState.hpp"
#include "dynosam_core/PointCloudProcess.hpp"

namespace dyno {

/**
 * @brief Struct containing debug imagery from the frontend that (optionally) is
 * included in the frontend output
 *
 */
struct DebugImagery {
  DYNO_POINTER_TYPEDEFS(DebugImagery)

  // TODO: make const!!
  cv::Mat tracking_image;
  // TODO: for now!
  cv::Mat rgb_viz;
  cv::Mat flow_viz;
  cv::Mat depth_viz;
  cv::Mat mask_viz;
};

struct RealtimeOutput {
  DYNO_POINTER_TYPEDEFS(RealtimeOutput)
  //! Current state data
  DynoState state;
  //! Possible dense point cloud (with label and RGB) in camera frame
  PointCloudLabelRGB::Ptr dense_labelled_cloud{nullptr};
  //! Debug/visualiation imagery for this frame. Internal data may be empty
  DebugImagery debug_imagery;
  //! Optionally proviced ground truth for this frame
  //! Mostly required for visualisation
  GroundTruthInputPacket::Optional ground_truth;
};

}  // namespace dyno
