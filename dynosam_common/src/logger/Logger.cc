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

#include "dynosam_common/logger/Logger.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <filesystem>

#include "dynosam_common/utils/Statistics.hpp"

DEFINE_string(output_path, "./", "Path where to store dynosam's log output.");

namespace dyno {

namespace fs = std::filesystem;

std::string getOutputFilePath(const std::string& file_name) {
  fs::path out_path(FLAGS_output_path);
  return out_path / file_name;
}

void writeStatisticsSamplesToFile(const std::string& file_name) {
  utils::Statistics::WriteAllSamplesToCsvFile(getOutputFilePath(file_name));
}

void writeStatisticsSummaryToFile(const std::string& file_name) {
  utils::Statistics::WriteSummaryToCsvFile(getOutputFilePath(file_name));
}

void writeStatisticsModuleSummariesToFile() {
  utils::Statistics::WritePerModuleSummariesToCsvFile(getOutputFilePath(""));
}

bool createDirectory(const std::string& path) {
  std::filesystem::path dir_path(path);

  // Check if directory already exists
  if (std::filesystem::exists(dir_path)) {
    if (std::filesystem::is_directory(dir_path)) {
      VLOG(10) << "Directory already exists: " << path;
      return true;
    } else {
      VLOG(10) << "Path exists but is not a directory: " << path;
      return false;
    }
  }

  // Create directory if it doesn't exist
  if (std::filesystem::create_directories(dir_path)) {
    VLOG(10) << "Directory created: " << path;
    return true;
  } else {
    LOG(WARNING) << "Failed to create directory: " << path;
    return false;
  }
}

// This constructor will directly open the log file when called.
OfstreamWrapper::OfstreamWrapper(const std::string& filename,
                                 const bool& open_file_in_append_mode)
    : OfstreamWrapper(filename, FLAGS_output_path, open_file_in_append_mode) {}

OfstreamWrapper::OfstreamWrapper(const std::string& filename,
                                 const std::string& output_path,
                                 const bool& open_file_in_append_mode)
    : filename_(filename), output_path_(output_path) {
  openLogFile(open_file_in_append_mode);
}

// This destructor will directly close the log file when the wrapper is
// destructed. So no need to explicitly call .close();
OfstreamWrapper::~OfstreamWrapper() {
  VLOG(20) << "Closing output file: " << filename_.c_str();
  ofstream_.flush();
  ofstream_.close();
}

void OfstreamWrapper::closeAndOpenLogFile() {
  ofstream_.flush();
  ofstream_.close();
  CHECK(!filename_.empty());
  OpenFile(output_path_ + '/' + filename_, &ofstream_, false);
}

bool OfstreamWrapper::WriteOutCsvWriter(const CsvWriter& csv,
                                        const std::string& filename) {
  // set append mode to false as we never want to write over the top of a csv
  // file as this will upset the header
  OfstreamWrapper ofsw(filename, false);
  return csv.write(ofsw.ofstream_);
}

void OfstreamWrapper::openLogFile(bool open_file_in_append_mode) {
  CHECK(!filename_.empty());
  CHECK(!output_path_.empty());
  LOG(INFO) << "Opening output file: " << filename_.c_str();
  OpenFile((std::string)getFilePath(), &ofstream_, open_file_in_append_mode);
}

fs::path OfstreamWrapper::getFilePath() const {
  fs::path fs_out_path(output_path_);
  if (!fs::exists(fs_out_path))
    throw std::runtime_error("OfstreamWrapper - Output path does not exist: " +
                             output_path_);

  return fs_out_path / fs::path(filename_);
}

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

  const gtsam::Pose3& motion_est = entry.data.motion;
  const gtsam::Pose3& pose_est = entry.data.pose;
  const FrameId frame_id = entry.frame_id;
  const Timestamp timestamp = entry.timestamp;

  // if (gt_packets) {
  //   if (gt_packets->exists(frame_id)) {
  //     const GroundTruthInputPacket& gt_packet_k = gt_packets->at(frame_id);
  //     // check object exists in this frame
  //     ObjectPoseGT object_gt_k;
  //     if (!gt_packet_k.getObject(object_id, object_gt_k)) {
  //       // if no packet for this object found, continue and do not log
  //       // return false;
  //     } else {
  //       CHECK(object_gt_k.prev_H_current_world_) << info_string(frame_id,
  //       object_id); motion_gt = *object_gt_k.prev_H_current_world_; pose_gt =
  //       object_gt_k.L_world_;
  //     }
  //   } else {
  //     // gt packet has no entry for this frame id and the object ground truth
  //     is
  //     // valid
  //     // TODO: for now?
  //     // return false;
  //   }
  // }
  auto object_ground_truth =
      getObjectGroundTruthHelper(frame_id, object_id, gt_packets);

  // TODO: return false?
  if (object_ground_truth) {
    CHECK(object_ground_truth->prev_H_current_world_)
        << info_string(frame_id, object_id);
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
