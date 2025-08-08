/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/utils/JsonUtils.hpp"

#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/VisionImuOutputPacket.hpp"

#define DYNO_FRONTEND_OUTPUT_PACKET_BASE_TO_JSON(jason, input)       \
  jason["frontend_type"] = input.frontend_type_;                     \
  jason["static_keypoints"] = input.static_keypoint_measurements_;   \
  jason["dynamic_keypoints"] = input.dynamic_keypoint_measurements_; \
  jason["T_world_camera"] = input.T_world_camera_;                   \
  jason["timestamp"] = input.getTimestamp();                         \
  jason["frame_id"] = input.getFrameId();                            \
  jason["ground_truth"] = input.gt_packet_;

namespace nlohmann {

// void adl_serializer<dyno::FrontendOutputPacketBase>::to_json(json& j, const
// dyno::FrontendOutputPacketBase& input) {
//     DYNO_FRONTEND_OUTPUT_PACKET_BASE_TO_JSON(j, input)
// }

// dyno::FrontendOutputPacketBase
// adl_serializer<dyno::FrontendOutputPacketBase>::from_json(const json& j) {
//     using namespace dyno;

//     StatusKeypointMeasurements static_keypoints =
//     j["static_keypoints"].template get<StatusKeypointMeasurements>();
//     StatusKeypointMeasurements dynamic_keypoints =
//     j["dynamic_keypoints"].template get<StatusKeypointMeasurements>();
//     gtsam::Pose3 T_world_camera = j["T_world_camera"].template
//     get<gtsam::Pose3>(); Timestamp timestamp = j["timestamp"].template
//     get<Timestamp>(); FrameId frame_id = j["frame_id"].template
//     get<FrameId>(); GroundTruthInputPacket::Optional gt_packet =
//     j["ground_truth"].template get<GroundTruthInputPacket::Optional>();

// }

// void adl_serializer<dyno::RGBDInstanceOutputPacket>::to_json(
//     json& j, const dyno::RGBDInstanceOutputPacket& input) {
//   using namespace dyno;

//   DYNO_FRONTEND_OUTPUT_PACKET_BASE_TO_JSON(j, input)
//   j["static_landmarks"] = input.static_landmarks_;
//   j["dynamic_landmarks"] = input.dynamic_landmarks_;
//   j["estimated_motions"] = input.object_motions_;
//   j["propogated_object_poses"] = input.propogated_object_poses_;
//   j["camera_poses"] = input.camera_poses_;
// }

// dyno::RGBDInstanceOutputPacket
// adl_serializer<dyno::RGBDInstanceOutputPacket>::from_json(const json& j) {
//   using namespace dyno;

//   StatusKeypointVector static_keypoints =
//       j["static_keypoints"].template get<StatusKeypointVector>();
//   StatusKeypointVector dynamic_keypoints =
//       j["dynamic_keypoints"].template get<StatusKeypointVector>();

//   gtsam::Pose3 T_world_camera =
//       j["T_world_camera"].template get<gtsam::Pose3>();

//   Timestamp timestamp = j["timestamp"].template get<Timestamp>();
//   FrameId frame_id = j["frame_id"].template get<FrameId>();

//   StatusLandmarkVector static_landmarks =
//       j["static_landmarks"].template get<StatusLandmarkVector>();
//   StatusLandmarkVector dynamic_landmarks =
//       j["dynamic_landmarks"].template get<StatusLandmarkVector>();
//   // Base is a std::map with the right custom allocation
//   // we use the std::map version so that all the automagic with
//   nlohmann::json
//   // can work
//   ObjectMotionMap estimated_motions(
//       j["estimated_motions"].template get<ObjectMotionMap>());
//   ObjectPoseMap propogated_object_poses =
//       j["propogated_object_poses"].template get<ObjectPoseMap>();
//   gtsam::Pose3Vector camera_poses =
//       j["camera_poses"].template get<gtsam::Pose3Vector>();

//   GroundTruthInputPacket::Optional gt_packet =
//       j["ground_truth"].template get<GroundTruthInputPacket::Optional>();

//   return dyno::RGBDInstanceOutputPacket(
//       static_keypoints, dynamic_keypoints, static_landmarks,
//       dynamic_landmarks, T_world_camera, timestamp, frame_id,
//       estimated_motions, propogated_object_poses, camera_poses, nullptr, /*
//       no camera*/ gt_packet, std::nullopt                         /* no debug
//       imagery */
//   );
// }

// TODO:
void adl_serializer<dyno::VisionImuPacket>::to_json(
    json& j, const dyno::VisionImuPacket& input) {
  using namespace dyno;
  // j["timestamp"] = input.timestamp;
  // j["frame_id"] = input.frame_id;
  // //NOTE: IMU not seralized!
  // json static_tracks;
  // static_tracks[""]
}

// TODO:
dyno::VisionImuPacket adl_serializer<dyno::VisionImuPacket>::from_json(
    const json& j) {
  using namespace dyno;
  return dyno::VisionImuPacket{};
}

}  // namespace nlohmann
