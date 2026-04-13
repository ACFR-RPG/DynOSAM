#include "dynosam_common/Trajectories.hpp"

namespace dyno {

MultiObjectTrajectories::MultiObjectTrajectories(
    const ObjectPoseMap& poses, const ObjectMotionMap& motion,
    const FrameIdTimestampMap& times) {
  for (const auto& [object_id, per_frame_pose_map] : poses) {
    if (!motion.exists(object_id)) {
      throw DynosamException(
          "Cannot create MultiObjectTrajectories from ObjectPoseMap and "
          "ObjectMotionMap "
          "as motion map missing j=" +
          std::to_string(object_id));
    }

    const auto& per_frame_motion_map = motion.at(object_id);
    for (const auto& [frame_id, L] : per_frame_pose_map) {
      if (!per_frame_motion_map.exists(frame_id)) {
        throw DynosamException(
            "Cannot create MultiObjectTrajectories from ObjectPoseMap and "
            "ObjectMotionMap "
            "as motion map for j=" +
            std::to_string(object_id) +
            " missing pose at k=" + std::to_string(frame_id));
      }

      if (!times.exists(frame_id)) {
        throw DynosamException(
            "Cannot create MultiObjectTrajectories from ObjectPoseMap and "
            "ObjectMotionMap "
            "as missing timestamp at k=" +
            std::to_string(frame_id));
      }

      const auto H = per_frame_motion_map.at(frame_id);
      const auto time = times.at(frame_id);
      this->insert(object_id, frame_id, time, PoseWithMotion{L, H});
    }
  }
}

void MultiObjectTrajectories::insert(ObjectId object_id, FrameId frame_id,
                                     Timestamp timestamp,
                                     const PoseWithMotion& data) {
  if (!hasObject(object_id)) {
    this->insert2(object_id, PoseWithMotionTrajectory{});
  }

  this->at(object_id).insert(frame_id, timestamp, data);
}

bool MultiObjectTrajectories::hasObject(ObjectId object_id) const {
  return this->exists(object_id);
}
size_t MultiObjectTrajectories::numObjects() const { return this->size(); }
ObjectIds MultiObjectTrajectories::objectIds() const {
  ObjectIds object_ids;
  std::transform(this->begin(), this->end(), std::back_inserter(object_ids),
                 [](auto entry) { return entry.first; });
  return object_ids;
}

bool MultiObjectTrajectories::hasFrame(ObjectId object_id,
                                       FrameId frame_id) const {
  if (!hasObject(object_id)) {
    return false;
  }

  const auto& trajectory = this->at(object_id);
  return trajectory.exists(frame_id);
}

FrameId MultiObjectTrajectories::lastFrame() const {
  FrameId frame_id;
  Timestamp timestamp;
  CHECK(getTemporalLastEntry(frame_id, timestamp));
  (void)timestamp;

  return frame_id;
}

Timestamp MultiObjectTrajectories::lastTimestamp() const {
  FrameId frame_id;
  Timestamp timestamp;
  CHECK(getTemporalLastEntry(frame_id, timestamp));
  (void)frame_id;

  return timestamp;
}

MultiObjectTrajectories::EntryMap MultiObjectTrajectories::entriesAtFrame(
    FrameId frame_id) const {
  EntryMap entry_map;
  for (const auto& [object_id, trajectory] : *this) {
    if (this->hasFrame(object_id, frame_id)) {
      const auto& trajectory = this->at(object_id);
      entry_map.insert2(object_id, trajectory.get(frame_id));
    }
  }
  return entry_map;
}

MultiObjectTrajectories MultiObjectTrajectories::trajectoriesAtFrame(
    FrameId frame_id) const {
  MultiObjectTrajectories trajectories;
  for (const auto& [object_id, trajectory] : *this) {
    if (this->hasFrame(object_id, frame_id)) {
      const auto& trajectory = this->at(object_id);
      trajectories.insert2(object_id, trajectory);
    }
  }
  return trajectories;
}

ObjectPoseMap MultiObjectTrajectories::toObjectPoseMap() const {
  ObjectPoseMap pose_map;
  for (const auto& [object_id, trajectory] : *this) {
    for (const auto& entry : trajectory) {
      pose_map.insert22(object_id, entry.frame_id, entry.data.pose);
    }
  }
  return pose_map;
}

ObjectMotionMap MultiObjectTrajectories::toObjectMotionMap() const {
  ObjectMotionMap motion_map;
  for (const auto& [object_id, trajectory] : *this) {
    for (const auto& entry : trajectory) {
      motion_map.insert22(object_id, entry.frame_id, entry.data.motion);
    }
  }
  return motion_map;
}

bool MultiObjectTrajectories::getTemporalLastEntry(FrameId& frame_id,
                                                   Timestamp& timestamp) const {
  if (this->empty()) {
    return false;
  }

  auto max_it = std::max_element(
      this->begin(), this->end(), [](const auto& a, const auto& b) {
        return a.second.maxFrame() < b.second.maxFrame();
      });

  if (max_it != this->end()) {
    frame_id = max_it->second.maxFrame();
    timestamp = max_it->second.endTime();
    return true;
  }
  return false;
}

}  // namespace dyno
