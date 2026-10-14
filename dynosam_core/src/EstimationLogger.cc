#include "dynosam_core/EstimationLogger.hpp"

#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/CsvParser.hpp"
#include "dynosam_common/utils/JsonUtils.hpp"
#include "dynosam_core/GroundTruthPacket.hpp"
#include "dynosam_core/PointCloudProcess.hpp"

namespace dyno {

EstimationModuleLogger::EstimationModuleLogger(const std::string& module_name)
    : module_name_(module_name),
      object_pose_file_name_(module_name_ + "_object_pose_log.csv"),
      object_motion_file_name_(module_name_ + "_object_motion_log.csv"),
      camera_pose_file_name_(module_name_ + "_camera_pose_log.csv"),
      map_points_file_name_(module_name_ + "_map_points_log.csv") {
  camera_pose_csv_ = std::make_unique<CsvWriter>(CsvHeader(
      "timestamp", "frame_id", "tx", "ty", "tz", "qx", "qy", "qz", "qw",
      "gt_tx", "gt_ty", "gt_tz", "gt_qx", "gt_qy", "gt_qz", "gt_qw"));

  object_pose_csv_ = std::make_unique<CsvWriter>(CsvHeader(
      "timestamp", "frame_id", "object_id", "tx", "ty", "tz", "qx", "qy", "qz",
      "qw", "gt_tx", "gt_ty", "gt_tz", "gt_qx", "gt_qy", "gt_qz", "gt_qw"));

  object_motion_csv_ = std::make_unique<CsvWriter>(CsvHeader(
      "timestamp", "frame_id", "object_id", "tx", "ty", "tz", "qx", "qy", "qz",
      "qw", "gt_tx", "gt_ty", "gt_tz", "gt_qx", "gt_qy", "gt_qz", "gt_qw"));

  map_points_csv_ = std::make_unique<CsvWriter>(CsvHeader(
      "frame_id", "object_id", "tracklet_id", "x_world", "y_world", "z_world"));
}

EstimationModuleLogger::~EstimationModuleLogger() {
  LOG(INFO) << "Writing out " << module_name_ << " logger...";
  OfstreamWrapper::WriteOutCsvWriter(*object_pose_csv_, object_pose_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*object_motion_csv_,
                                     object_motion_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*camera_pose_csv_, camera_pose_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*map_points_csv_, map_points_file_name_);
}

size_t EstimationModuleLogger::logObjectTrajectory(
    FrameId frame_id, const MultiObjectTrajectories& object_trajectories,
    const std::optional<GroundTruthPacketMap>& gt_packets) {
  size_t num_objects_logged = 0;
  for (const auto& [object_id, object_trajectory] : object_trajectories) {
    if (object_trajectory.exists(frame_id)) {
      const auto& entry = object_trajectory.get(frame_id);
      if (logObjectTrajectoryEntry(entry, object_id, gt_packets)) {
        num_objects_logged++;
        // continue to next object
        continue;
      }
    }
  }
  return num_objects_logged;
}

size_t EstimationModuleLogger::logObjectTrajectory(
    const MultiObjectTrajectories& object_trajectories,
    const std::optional<GroundTruthPacketMap>& gt_packets) {
  size_t number_logged = 0;
  for (const auto& [object_id, object_trajectory] : object_trajectories) {
    for (const auto& entry : object_trajectory) {
      if (logObjectTrajectoryEntry(entry, object_id, gt_packets)) {
        number_logged++;
      }
    }
  }
  return number_logged;
}

// assume poses are in world?
size_t EstimationModuleLogger::logCameraPose(
    FrameId frame_id, const PoseTrajectory& camera_poses,
    const std::optional<GroundTruthPacketMap>& gt_packets) {
  if (camera_poses.exists(frame_id)) {
    const auto& entry = camera_poses.get(frame_id);
    if (logCameraPoseEntry(entry, gt_packets)) {
      return 1;
    }
  }

  return 0;
}

size_t EstimationModuleLogger::logCameraPose(
    const PoseTrajectory& camera_poses,
    const std::optional<GroundTruthPacketMap>& gt_packets) {
  size_t number_logged = 0;
  for (const auto& entry : camera_poses) {
    if (logCameraPoseEntry(entry, gt_packets)) {
      number_logged++;
    }
  }
  return number_logged;
}

bool EstimationModuleLogger::logCameraPoseEntry(
    const PoseTrajectoryEntry& entry,
    const std::optional<GroundTruthPacketMap>& gt_packets) {
  const gtsam::Pose3& pose_est = entry.data;
  const FrameId frame_id = entry.frame_id;
  const Timestamp timestamp = entry.timestamp;

  gtsam::Pose3 pose_gt = gtsam::Pose3::Identity();
  // if gt packet provided by no data exists at this frame
  if (gt_packets && !gt_packets->exists(frame_id)) {
    VLOG(100) << "No gt packet at frame id " << frame_id
              << ". Unable to log object motions";
    return false;
  } else if (gt_packets && gt_packets->exists(frame_id)) {
    const GroundTruthInputPacket& gt_packet_k = gt_packets->at(frame_id);
    pose_gt = gt_packet_k.X_world_;
  }

  logCameraSE3(*camera_pose_csv_, pose_est, pose_gt, timestamp, frame_id);
  return true;
}

void EstimationModuleLogger::logPoints(FrameId frame_id,
                                       const gtsam::Pose3& T_world_local_k,
                                       const StatusLandmarkVector& landmarks) {
  for (const auto& status_lmks : landmarks) {
    const TrackletId tracklet_id = status_lmks.trackletId();
    ObjectId object_id = status_lmks.objectId();
    Landmark lmk_world = status_lmks.value();

    if (status_lmks.referenceFrame() == ReferenceFrame::LOCAL) {
      lmk_world = T_world_local_k * status_lmks.value();
    } else if (status_lmks.referenceFrame() == ReferenceFrame::OBJECT) {
      throw DynosamException(
          "Cannot log object point in the object reference frame");
    }

    *map_points_csv_ << frame_id << object_id << tracklet_id << lmk_world(0)
                     << lmk_world(1) << lmk_world(2);
  }
}

void EstimationModuleLogger::logMapPoints(
    const StatusLandmarkVector& landmarks) {
  for (const auto& status_lmks : landmarks) {
    const TrackletId tracklet_id = status_lmks.trackletId();
    ObjectId object_id = status_lmks.objectId();
    Landmark lmk_world = status_lmks.value();
    FrameId frame_id = status_lmks.frameId();

    if (status_lmks.referenceFrame() != ReferenceFrame::GLOBAL) {
      throw DynosamException(
          "Failure in logMapPoints(): Map point is not in the GLOBAL frame");
    }

    *map_points_csv_ << frame_id << object_id << tracklet_id << lmk_world(0)
                     << lmk_world(1) << lmk_world(2);
  }
}

bool EstimationModuleLogger::logObjectTrajectoryEntry(
    const PoseWithMotionEntry& entry, const ObjectId object_id,
    const std::optional<GroundTruthPacketMap>& gt_packets) {
  // TODO: check F2F
  gtsam::Pose3 motion_gt = gtsam::Pose3::Identity();
  gtsam::Pose3 pose_gt = gtsam::Pose3::Identity();

  if (entry.data.motion.style() != MotionRepresentationStyle::F2F) {
    DYNO_THROW_MSG(DynosamException)
        << "Error logging PoseWithMotionEntry entry: "
        << " motion entry " << info_string(entry.frame_id, object_id)
        << " does not"
        << " represent frame-to-frame motion!";
  }

  const Motion3ReferenceFrame& motion_est_ref = entry.data.motion;
  const gtsam::Pose3& motion_est = motion_est_ref;
  const gtsam::Pose3& pose_est = entry.data.pose;
  const FrameId frame_id = entry.frame_id;
  const Timestamp timestamp = entry.timestamp;

  auto object_ground_truth =
      getObjectGroundTruthHelper(frame_id, object_id, gt_packets);

  // check that we have the necessary object ground truth
  if (object_ground_truth &&
      !object_ground_truth->prev_H_current_world_.has_value()) {
    // in the case where we dont have ground truth motion, this is ONLY
    // acceptable in the case where the motion provided has from_frame == to
    // frame (ie. in the hybrid case where we have a motion on a KF pose, at the
    // motion = I)
    if (motion_est_ref.from() != motion_est_ref.to()) {
      DYNO_THROW_MSG(DynosamException)
          << "GTObjectPacket found but issing H_W_km1_k ground truth motion "
          << info_string(frame_id, object_id);
    }
  }
  if (object_ground_truth) {
    // we have already checked that prev_H_current_world_ exists in the previous
    // if condition!
    motion_gt = *(object_ground_truth->prev_H_current_world_);
    pose_gt = object_ground_truth->L_world_;
  }

  logObjectSE3(*object_motion_csv_, motion_est, motion_gt, timestamp, frame_id,
               object_id);

  logObjectSE3(*object_pose_csv_, pose_est, pose_gt, timestamp, frame_id,
               object_id);

  return true;
}

std::optional<ObjectPoseGT> EstimationModuleLogger::getObjectGroundTruthHelper(
    FrameId frame_id, ObjectId object_id,
    const std::optional<GroundTruthPacketMap>& gt_packets) const {
  std::optional<ObjectPoseGT> possible_object_gt = {};

  if (gt_packets && gt_packets->exists(frame_id)) {
    const GroundTruthInputPacket& gt_packet_k = gt_packets->at(frame_id);

    ObjectPoseGT object_gt_k;
    if (gt_packet_k.getObject(object_id, object_gt_k)) {
      possible_object_gt.emplace(object_gt_k);
    }
  }
  return possible_object_gt;
}

void EstimationModuleLogger::logObjectSE3(CsvWriter& writer,
                                          const gtsam::Pose3& se3_est,
                                          const gtsam::Pose3& se3_gt,
                                          Timestamp timestamp, FrameId frame_id,
                                          ObjectId object_id) {
  const auto timestamp_nano = timestampToNano(timestamp);

  const auto& quat_est = se3_est.rotation().toQuaternion();
  const auto& gt_quat = se3_gt.rotation().toQuaternion();

  writer << timestamp_nano << frame_id << object_id << se3_est.x()
         << se3_est.y() << se3_est.z() << quat_est.x() << quat_est.y()
         << quat_est.z() << quat_est.w() << se3_gt.x() << se3_gt.y()
         << se3_gt.z() << gt_quat.x() << gt_quat.y() << gt_quat.z()
         << gt_quat.w();
}

void EstimationModuleLogger::logCameraSE3(CsvWriter& writer,
                                          const gtsam::Pose3& se3_est,
                                          const gtsam::Pose3& se3_gt,
                                          Timestamp timestamp,
                                          FrameId frame_id) {
  const auto timestamp_nano = timestampToNano(timestamp);

  const auto& quat_est = se3_est.rotation().toQuaternion();
  const auto& gt_quat = se3_gt.rotation().toQuaternion();

  writer << timestamp_nano << frame_id << se3_est.x() << se3_est.y()
         << se3_est.z() << quat_est.x() << quat_est.y() << quat_est.z()
         << quat_est.w() << se3_gt.x() << se3_gt.y() << se3_gt.z()
         << gt_quat.x() << gt_quat.y() << gt_quat.z() << gt_quat.w();
}

}  // namespace dyno
