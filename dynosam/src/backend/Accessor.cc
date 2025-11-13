#include "dynosam/backend/Accessor.hpp"

#include <glog/logging.h>

namespace dyno {

bool Accessor::hasObjectMotionEstimate(FrameId frame_id, ObjectId object_id,
                                       Motion3& motion) const {
  return this->hasObjectMotionEstimate(frame_id, object_id, &motion);
}

bool Accessor::hasObjectPoseEstimate(FrameId frame_id, ObjectId object_id,
                                     gtsam::Pose3& pose) const {
  return this->hasObjectPoseEstimate(frame_id, object_id, &pose);
}

std::tuple<gtsam::Point3, bool> Accessor::computeObjectCentroid(
    FrameId frame_id, ObjectId object_id) const {
  const StatusLandmarkVector& dynamic_lmks =
      this->getDynamicLandmarkEstimates(frame_id, object_id);

  // convert to point cloud - should be a map with only one map in it
  CloudPerObject object_clouds =
      groupObjectCloud(dynamic_lmks, this->getSensorPose(frame_id).get());
  if (object_clouds.size() == 0) {
    VLOG(20) << "Cannot collect object clouds from dynamic landmarks of "
             << object_id << " and frame " << frame_id << "!! "
             << " # Dynamic lmks in the map for this object at this frame was "
             << dynamic_lmks.size();  //<< " but reocrded lmks was " <<
                                      // dynamic_landmarks.size();
    return {gtsam::Point3{}, false};
  }
  CHECK_EQ(object_clouds.size(), 1);
  CHECK(object_clouds.exists(object_id));

  const auto dynamic_point_cloud = object_clouds.at(object_id);
  pcl::PointXYZ centroid;
  pcl::computeCentroid(dynamic_point_cloud, centroid);
  // TODO: outlier reject?
  gtsam::Point3 translation = pclPointToGtsam(centroid);
  return {translation, true};
}

StatusLandmarkVector Accessor::getLandmarkEstimates(FrameId frame_id) const {
  StatusLandmarkVector estimates;
  estimates += getStaticLandmarkEstimates(frame_id);
  estimates += getDynamicLandmarkEstimates(frame_id);
  return estimates;
}

StateQuery<Motion3ReferenceFrame> Accessor::getObjectMotionReferenceFrame(
    FrameId frame_id, ObjectId object_id) const {
  StateQuery<Motion3> motion_query = this->getObjectMotion(frame_id, object_id);
  if (motion_query) {
    return StateQuery<Motion3ReferenceFrame>(
        motion_query.key_,
        Motion3ReferenceFrame(motion_query.get(),
                              Motion3ReferenceFrame::Style::F2F,
                              ReferenceFrame::GLOBAL, frame_id - 1u, frame_id));
  } else {
    return StateQuery<Motion3ReferenceFrame>(motion_query.key_,
                                             motion_query.status_);
  }
}

}  // namespace dyno
