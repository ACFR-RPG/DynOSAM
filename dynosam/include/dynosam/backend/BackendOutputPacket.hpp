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

#include "dynosam/backend/BackendDefinitions.hpp"  //for TemporalObjectMetaData
#include "dynosam/common/Types.hpp"

namespace dyno {

struct BackendOutputPacket {
  DYNO_POINTER_TYPEDEFS(BackendOutputPacket)

  StatusLandmarkVector static_landmarks;   // all frames?
  StatusLandmarkVector dynamic_landmarks;  // all objects only this frame?
  // TODO: depricate and let output viz handle now it has shared module...
  FrameIdTimestampMap involved_timestamp;
  gtsam::Pose3 T_world_camera;
  FrameId frame_id;
  Timestamp timestamp;
  ObjectMotionMap optimized_object_motions;
  ObjectPoseMap optimized_object_poses;
  gtsam::Pose3Vector optimized_camera_poses;
  std::vector<TemporalObjectMetaData> temporal_object_data;
  // streamline imagery with incremental visualisation tools!
  cv::Mat debug_image;

  inline FrameId getFrameId() const { return frame_id; }
  inline Timestamp getTimestamp() const { return timestamp; }
  const gtsam::Pose3& pose() const { return T_world_camera; }
};

}  // namespace dyno
