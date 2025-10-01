/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/frontend/VisionImuOutputPacket.hpp"

namespace dyno {

Timestamp VisionImuPacket::timestamp() const { return timestamp_; }

FrameId VisionImuPacket::frameId() const { return frame_id_; }

ImuFrontend::PimPtr VisionImuPacket::pim() const { return pim_; }

Camera::ConstPtr VisionImuPacket::camera() const { return camera_; }

PointCloudLabelRGB::Ptr VisionImuPacket::denseLabelledCloud() const {
  return dense_labelled_cloud_;
}

const VisionImuPacket::CameraTracks& VisionImuPacket::cameraTracks() const {
  return camera_tracks_;
}
const VisionImuPacket::ObjectTrackMap& VisionImuPacket::objectTracks() const {
  return object_tracks_;
}

const PoseEstimateMap& VisionImuPacket::objectPoses() const {
  return cached_object_poses_;
}

const ObjectIds& VisionImuPacket::getObjectIds() const {
  return cached_object_ids_;
}

const gtsam::Pose3& VisionImuPacket::cameraPose() const {
  return camera_tracks_.X_W_k;
}
const gtsam::Pose3& VisionImuPacket::relativeCameraTransform() const {
  return camera_tracks_.T_k_1_k;
}

const MotionEstimateMap& VisionImuPacket::objectMotions() const {
  return cached_object_motions_;
}

const GroundTruthInputPacket::Optional& VisionImuPacket::groundTruthPacket()
    const {
  return ground_truth_;
}

const DebugImagery::Optional& VisionImuPacket::debugImagery() const {
  return debug_imagery_;
}

const CameraMeasurementStatusVector& VisionImuPacket::objectMeasurements()
    const {
  return cached_object_measurements_;
}
const CameraMeasurementStatusVector& VisionImuPacket::staticMeasurements()
    const {
  return camera_tracks_.measurements;
}

StatusLandmarkVector VisionImuPacket::staticLandmarkMeasurements() const {
  StatusLandmarkVector static_landmarks;
  fillLandmarkMeasurements(static_landmarks, staticMeasurements());
  return static_landmarks;
}
StatusLandmarkVector VisionImuPacket::dynamicLandmarkMeasurements() const {
  StatusLandmarkVector dynamic_landmarks;
  fillLandmarkMeasurements(dynamic_landmarks, objectMeasurements());
  return dynamic_landmarks;
}

VisionImuPacket& VisionImuPacket::timestamp(Timestamp ts) {
  timestamp_ = ts;
  return *this;
}

VisionImuPacket& VisionImuPacket::frameId(FrameId id) {
  frame_id_ = id;
  return *this;
}

VisionImuPacket& VisionImuPacket::pim(const ImuFrontend::PimPtr& pim) {
  pim_ = pim;
  return *this;
}

VisionImuPacket& VisionImuPacket::camera(const Camera::Ptr& cam) {
  camera_ = cam;
  return *this;
}

VisionImuPacket& VisionImuPacket::denseLabelledCloud(
    const PointCloudLabelRGB::Ptr& cloud) {
  dense_labelled_cloud_ = cloud;
  return *this;
}

VisionImuPacket& VisionImuPacket::cameraTracks(
    const VisionImuPacket::CameraTracks& camera_tracks) {
  camera_tracks_ = camera_tracks;
  return *this;
}

VisionImuPacket& VisionImuPacket::objectTracks(
    const VisionImuPacket::ObjectTrackMap& object_tracks) {
  object_tracks_ = object_tracks;
  updateObjectTrackCaches();
  return *this;
}

VisionImuPacket& VisionImuPacket::objectTracks(
    const VisionImuPacket::ObjectTracks& object_track, ObjectId object_id) {
  if (object_tracks_.exists(object_id)) {
    DYNO_THROW_MSG(DynosamException)
        << "Cannot add object track j=" << object_id
        << " to VisionImuPacket k=" << frameId()
        << " as object already exists!";
  }
  object_tracks_.insert2(object_id, object_track);
  updateObjectTrackCaches();
  return *this;
}

VisionImuPacket& VisionImuPacket::groundTruthPacket(
    const GroundTruthInputPacket::Optional& gt) {
  ground_truth_ = gt;
  return *this;
}

VisionImuPacket& VisionImuPacket::debugImagery(
    const DebugImagery::Optional& dbg) {
  debug_imagery_ = dbg;
  return *this;
}

void VisionImuPacket::updateObjectTrackCaches() {
  cached_object_ids_.clear();
  cached_object_motions_.clear();
  cached_object_measurements_.clear();
  cached_object_poses_.clear();

  cached_object_ids_.reserve(object_tracks_.size());
  for (const auto& [object_id, object_track] : object_tracks_) {
    cached_object_ids_.push_back(object_id);
    cached_object_motions_.insert2(object_id, object_track.H_W_k_1_k);
    cached_object_poses_.insert2(object_id, object_track.L_W_k);
    // TODO: if valid??
    cached_object_measurements_ += object_track.measurements;
  }
}

void VisionImuPacket::fillLandmarkMeasurements(
    StatusLandmarkVector& landmarks,
    const CameraMeasurementStatusVector& camera_measurements) {
  landmarks.reserve(camera_measurements.size());
  // implicit cast during iteration
  for (const CameraMeasurementStatus& cms : camera_measurements) {
    const CameraMeasurement& measurement = cms.value();
    if (measurement.hasLandmark()) {
      landmarks.push_back(LandmarkStatus(measurement.landmark(), cms.frameId(),
                                         cms.trackletId(), cms.objectId(),
                                         cms.referenceFrame()));
    }
  }
}

}  // namespace dyno

namespace nlohmann {
void adl_serializer<dyno::VisionImuPacket>::to_json(
    json& j, const dyno::VisionImuPacket& input) {
  using namespace dyno;
  throw DynosamException(
      "nlohmann::to_json not implemented for dyno::VisionImuPacket");
}

// TODO:
dyno::VisionImuPacket adl_serializer<dyno::VisionImuPacket>::from_json(
    const json& j) {
  using namespace dyno;
  throw DynosamException(
      "nlohmann::from_json not implemented for dyno::VisionImuPacket");
}
}  // namespace nlohmann
