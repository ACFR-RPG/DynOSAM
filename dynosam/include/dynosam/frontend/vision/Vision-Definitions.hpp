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

#include <mutex>

#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_cv/Feature.hpp"
#include "dynosam_cv/ImageContainer.hpp"

namespace dyno {

template <typename U, typename V>
struct TrackletCorrespondance {
  TrackletId tracklet_id_;
  U ref_;
  V cur_;

  TrackletCorrespondance() {}
  TrackletCorrespondance(TrackletId tracklet_id, U ref, V cur)
      : tracklet_id_(tracklet_id), ref_(ref), cur_(cur) {}
};

template <typename RefType, typename CurType>
using GenericCorrespondences =
    std::vector<TrackletCorrespondance<RefType, CurType>>;

//! Correspondes format for a 3D->2D PnP solver. In the form of 3D Landmark in
//! the world frame, and 2D observation in the current camera frame
using AbsolutePoseCorrespondence = TrackletCorrespondance<Landmark, Keypoint>;
using AbsolutePoseCorrespondences = std::vector<AbsolutePoseCorrespondence>;

//! Correspondes format for a 2D->2D PnP solver. In the form of 2D observation
//! in the ref camera frame, and 2D observation in the current camera frame
using RelativePoseCorrespondence = TrackletCorrespondance<Keypoint, Keypoint>;
using RelativePoseCorrespondences = std::vector<RelativePoseCorrespondence>;

//! Correspondes format for a 3D->3D PnP solver. In the form of a 3D Landmark in
//! the world frame
using PointCloudCorrespondence = TrackletCorrespondance<Landmark, Landmark>;
using PointCloudCorrespondences = std::vector<PointCloudCorrespondence>;

struct PerObjectStatus {
  ObjectId object_id;
  size_t num_previous_track{0};  // number of (inlier) features tracked in the
                                 // previous frame that MAY be used
  size_t num_track{
      0};  // actual number of features tracked (ie. used) from the previous
           // frame - does not include newly sampled points!
  size_t num_sampled{
      0};  // num new points sampled and added to the set of features
  size_t num_outside_shrunken_image{0};  // sampled or tracked
  size_t num_zero_flow{0};               // sampled or tracked
  size_t num_tracked_with_different_label{
      0};  // number of points tracked from previous frame where the current
           // label is different
  size_t num_tracked_with_background_label{
      0};  // number of points tracked from previous frame wehre current label
           // is the background
  // would be nice to have some histogram data about each tracked point etc...
  bool object_new{false};
  bool object_resampled{false};

  PerObjectStatus(ObjectId id) : object_id(id) {}
};

struct FeatureTrackerInfo {
  FrameId frame_id;
  Timestamp timestamp;

  // static track info
  size_t static_track_optical_flow{0};
  size_t static_track_detections{0};

  //! If new features were extracted at this frame
  //! Set in the (static) tracker and partially used to indicate if a new static
  //! kf is needed
  bool new_static_detections{false};

  inline PerObjectStatus& getObjectStatus(ObjectId object_id) {
    if (!dynamic_track.exists(object_id)) {
      dynamic_track.insert2(object_id, PerObjectStatus(object_id));
    }
    // CHECK(dynamic_track.exists(object_id));
    return dynamic_track.at(object_id);
  }

  gtsam::FastMap<ObjectId, PerObjectStatus> dynamic_track;
};

template <>
inline std::string to_string(const FeatureTrackerInfo& info) {
  std::stringstream ss;
  ss << "FeatureTrackerInfo: \n"
     << " - frame id: " << info.frame_id << "\n"
     << " - timestamp: " << std::setprecision(15) << info.timestamp << "\n"
     << "\t- # optical flow: " << info.static_track_optical_flow << "\n"
     << "\t- # detections: " << info.static_track_detections << "\n"
     << "\t- # new detections: " << info.new_static_detections << "\n";

  for (const auto& [object_id, object_status] : info.dynamic_track) {
    ss << "\t- Object: " << object_id << ": \n";
    ss << "\t\t - num_track " << object_status.num_track << "\n";
    ss << "\t\t - num_sampled " << object_status.num_sampled << "\n";
    ss << "\t\t - is new " << std::boolalpha << object_status.object_new
       << "\n";
    ss << "\t\t - resampled " << std::boolalpha
       << object_status.object_resampled << "\n";

    if (VLOG_IS_ON(20)) {
      ss << "\t\t - num_previous_track " << object_status.num_previous_track
         << "\n";
      ss << "\t\t - num_outside_shrunken_image "
         << object_status.num_outside_shrunken_image << "\n";
      ss << "\t\t - num_zero_flow " << object_status.num_zero_flow << "\n";
      ss << "\t\t - num_tracked_with_different_label "
         << object_status.num_tracked_with_different_label << "\n";
      ss << "\t\t - num_tracked_with_background_label "
         << object_status.num_tracked_with_background_label << "\n";
    }
  };
  return ss.str();
}

}  // namespace dyno
