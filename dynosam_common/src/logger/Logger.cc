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
      object_bbx_file_name_(module_name_ + "_object_bbx_log.csv"),
      // camera_pose_errors_file_name_(module_name_ +
      // "_camera_pose_error_log.csv"),
      camera_pose_file_name_(module_name_ + "_camera_pose_log.csv"),
      map_points_file_name_(module_name_ + "_map_points_log.csv"),
      frame_id_to_timestamp_file_name_(
          "frame_id_timestamp.csv")  // NOTE: not prefixed by module
{
  camera_pose_csv_ = std::make_unique<CsvWriter>(
      CsvHeader("frame_id", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "gt_tx",
                "gt_ty", "gt_tz", "gt_qx", "gt_qy", "gt_qz", "gt_qw"));

  object_pose_csv_ = std::make_unique<CsvWriter>(CsvHeader(
      "frame_id", "object_id", "tx", "ty", "tz", "qx", "qy", "qz", "qw",
      "gt_tx", "gt_ty", "gt_tz", "gt_qx", "gt_qy", "gt_qz", "gt_qw"));

  object_motion_csv_ = std::make_unique<CsvWriter>(CsvHeader(
      "frame_id", "object_id", "tx", "ty", "tz", "qx", "qy", "qz", "qw",
      "gt_tx", "gt_ty", "gt_tz", "gt_qx", "gt_qy", "gt_qz", "gt_qw"));

  map_points_csv_ = std::make_unique<CsvWriter>(CsvHeader(
      "frame_id", "object_id", "tracklet_id", "x_world", "y_world", "z_world"));

  object_bbx_csv_ = std::make_unique<CsvWriter>(
      CsvHeader("frame_id", "object_id", "min_bbx_x", "min_bbx_y", "min_bbx_z",
                "max_bbx_x", "max_bbx_y", "max_bbx_z", "px", "py", "pz", "qw",
                "qx", "qy", "qz"));

  frame_id_timestamp_csv_ =
      std::make_unique<CsvWriter>(CsvHeader("frame_id", "timestamp [ns]"));
}

EstimationModuleLogger::~EstimationModuleLogger() {
  LOG(INFO) << "Writing out " << module_name_ << " logger...";
  OfstreamWrapper::WriteOutCsvWriter(*object_pose_csv_, object_pose_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*object_bbx_csv_, object_bbx_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*object_motion_csv_,
                                     object_motion_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*camera_pose_csv_, camera_pose_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*map_points_csv_, map_points_file_name_);
  // OfstreamWrapper::WriteOutCsvWriter(*frame_id_timestamp_csv_,
  //                                    frame_id_to_timestamp_file_name_);
}

std::optional<size_t> EstimationModuleLogger::logObjectMotion(
    FrameId frame_id, const MotionEstimateMap& motion_estimates,
    const std::optional<GroundTruthPacketMap>& gt_packets) {
  size_t number_logged = 0;
  for (const auto& [object_id, motions] : motion_estimates) {
    // use identity motion if no ground truth so that we keep the csv file
    // format
    if (logObjectMotion(*object_motion_csv_, motions, frame_id, object_id,
                        gt_packets))
      number_logged++;
  }
  return number_logged;
}

std::optional<size_t> EstimationModuleLogger::logObjectMotion(
    const ObjectMotionMap& object_motions,
    const std::optional<GroundTruthPacketMap>& gt_packets) {
  size_t number_logged = 0;
  for (const auto& [object_id, motions_per_object] : object_motions) {
    for (const auto& [frame_id, motion] : motions_per_object) {
      if (logObjectMotion(*object_motion_csv_, motion, frame_id, object_id,
                          gt_packets))
        number_logged++;
    }
  }
  return number_logged;
}

// assume poses are in world?
std::optional<size_t> EstimationModuleLogger::logObjectPose(
    FrameId frame_id, const PoseEstimateMap& object_poses,
    const std::optional<GroundTruthPacketMap>& gt_packets) {
  size_t number_logged = 0;
  // assume object poses get logged in frame order!!!
  for (const auto& [object_id, L_W_k] : object_poses) {
    if (logObjectPose(*object_pose_csv_, L_W_k, frame_id, object_id,
                      gt_packets))
      number_logged++;
  }
  return number_logged;
}

// // assume poses are in world?
// std::optional<size_t> EstimationModuleLogger::logObjectPose(
//     FrameId frame_id, const PoseEstimateMap& object_poses,
//     const std::optional<GroundTruthPacketMap>& gt_packets) {
//   size_t number_logged = 0;
//   // assume object poses get logged in frame order!!!
//   for (const auto& [object_id, poses_map] : propogated_poses) {
//     // do not draw if in current frame
//     if (!poses_map.exists(frame_id)) {
//       VLOG(100) << "Cannot log object pose (id=" << object_id << ") for frame
//       "
//                 << frame_id << " as it does not exist in the map";
//       continue;
//     }

//     const gtsam::Pose3& L_W_k = poses_map.at(frame_id);
//     if (logObjectPose(*object_pose_csv_, L_W_k, frame_id, object_id,
//                       gt_packets))
//       number_logged++;
//     // use identity if no ground truth so that we keep the csv file format
//     // gtsam::Pose3 gt_L_world_k = gtsam::Pose3::Identity();
//     // const gtsam::Pose3& L_world_k = poses_map.at(frame_id);

//     // // gt packet provided so only log if gt pose data found
//     // if (gt_packets) {
//     //   if (gt_packets->exists(frame_id)) {
//     //     const GroundTruthInputPacket& gt_packet_k =
//     gt_packets->at(frame_id);
//     //     // check object exists in this frame
//     //     ObjectPoseGT object_gt_k;
//     //     if (!gt_packet_k.getObject(object_id, object_gt_k)) {
//     //       // if no packet for this object found, continue and do not log
//     //       continue;
//     //     } else {
//     //       gt_L_world_k = object_gt_k.L_world_;
//     //     }
//     //   } else {
//     //     // gt packet has no entry for this frame id so skip
//     //     continue;
//     //   }
//     // }
//     // // we have either skipped logging (if gt provided by the object gt
//     pose
//     // was
//     // // not found) or no gt was provided
//     // const auto& gt_R_k = gt_L_world_k.rotation().toQuaternion();
//     // // estimate
//     // const auto R_world_k = L_world_k.rotation().toQuaternion();
//     // // write out object pose with gt
//     // *object_pose_csv_ << frame_id << object_id << L_world_k.x() <<
//     // L_world_k.y()
//     //                   << L_world_k.z() << R_world_k.x() << R_world_k.y()
//     //                   << R_world_k.z() << R_world_k.w() <<
//     gt_L_world_k.x()
//     //                   << gt_L_world_k.y() << gt_L_world_k.z() <<
//     gt_R_k.x()
//     //                   << gt_R_k.y() << gt_R_k.z() << gt_R_k.w();
//   }
//   return number_logged;
// }

std::optional<size_t> EstimationModuleLogger::logObjectPose(
    const ObjectPoseMap& object_poses,
    const std::optional<GroundTruthPacketMap>& gt_packets) {
  size_t number_logged = 0;
  for (const auto& [object_id, poses_per_object] : object_poses) {
    for (const auto& [frame_id, pose] : poses_per_object) {
      if (logObjectPose(*object_pose_csv_, pose, frame_id, object_id,
                        gt_packets))
        number_logged++;
    }
  }
  return number_logged;
}

std::optional<size_t> EstimationModuleLogger::logCameraPose(
    FrameId frame_id, const gtsam::Pose3& T_world_camera,
    const std::optional<GroundTruthPacketMap>& gt_packets) {
  gtsam::Pose3 gt_T_world_camera_k = gtsam::Pose3::Identity();
  // if gt packet provided by no data exists at this frame
  if (gt_packets && !gt_packets->exists(frame_id)) {
    VLOG(100) << "No gt packet at frame id " << frame_id
              << ". Unable to log object motions";
    return {};
  } else if (gt_packets && gt_packets->exists(frame_id)) {
    const GroundTruthInputPacket& gt_packet_k = gt_packets->at(frame_id);
    gt_T_world_camera_k = gt_packet_k.X_world_;
  }

  const auto& rot = T_world_camera.rotation().toQuaternion();
  const auto& gt_rot = gt_T_world_camera_k.rotation().toQuaternion();
  *camera_pose_csv_ << frame_id << T_world_camera.x() << T_world_camera.y()
                    << T_world_camera.z() << rot.x() << rot.y() << rot.z()
                    << rot.w() << gt_T_world_camera_k.x()
                    << gt_T_world_camera_k.y() << gt_T_world_camera_k.z()
                    << gt_rot.x() << gt_rot.y() << gt_rot.z() << gt_rot.w();
  return 1;
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

void EstimationModuleLogger::logObjectBbxes(FrameId frame_id,
                                            const BbxPerObject& object_bbxes) {
  for (const auto& [object_id, this_object_bbx] : object_bbxes) {
    const gtsam::Quaternion& q = this_object_bbx.orientation_.toQuaternion();
    *object_bbx_csv_ << frame_id << object_id
                     << this_object_bbx.min_bbx_point_.x()
                     << this_object_bbx.min_bbx_point_.y()
                     << this_object_bbx.min_bbx_point_.z()
                     << this_object_bbx.max_bbx_point_.x()
                     << this_object_bbx.max_bbx_point_.y()
                     << this_object_bbx.max_bbx_point_.z()
                     << this_object_bbx.bbx_position_.x()
                     << this_object_bbx.bbx_position_.y()
                     << this_object_bbx.bbx_position_.z() << q.w() << q.x()
                     << q.y() << q.z();
  }
}

void EstimationModuleLogger::logFrameIdToTimestamp(FrameId frame_id,
                                                   Timestamp timestamp) {
  long int nano_seconds = timestamp * 1e+9;
  *frame_id_timestamp_csv_ << frame_id << nano_seconds;
}

bool EstimationModuleLogger::logObjectMotion(
    CsvWriter& writer, const Motion3ReferenceFrame& motion, FrameId frame_id,
    ObjectId object_id, const std::optional<GroundTruthPacketMap>& gt_packets) {
  gtsam::Pose3 gt_motion = gtsam::Pose3::Identity();
  CHECK_EQ(motion.style(), MotionRepresentationStyle::F2F)
      << "Output evaluation expects motion in F2F representation";
  const gtsam::Pose3& estimate = motion;

  if (gt_packets) {
    if (gt_packets->exists(frame_id)) {
      const GroundTruthInputPacket& gt_packet_k = gt_packets->at(frame_id);

      // check object exists in this frame
      ObjectPoseGT object_gt_k;
      if (!gt_packet_k.getObject(object_id, object_gt_k)) {
        // if no packet for this object found, continue and do not log
        // return false;
      } else {
        CHECK(object_gt_k.prev_H_current_world_);
        gt_motion = *object_gt_k.prev_H_current_world_;
      }
    } else {
      // gt packet has no entry for this frame id and the object ground truth is
      // valid
      // TODO: for now?
      // return false;
    }
  }

  const auto& quat = estimate.rotation().toQuaternion();
  const auto& gt_quat = gt_motion.rotation().toQuaternion();

  writer << frame_id << object_id << estimate.x() << estimate.y()
         << estimate.z() << quat.x() << quat.y() << quat.z() << quat.w()
         << gt_motion.x() << gt_motion.y() << gt_motion.z() << gt_quat.x()
         << gt_quat.y() << gt_quat.z() << gt_quat.w();
  return true;
}

bool EstimationModuleLogger::logObjectPose(
    CsvWriter& writer, const gtsam::Pose3& pose, FrameId frame_id,
    ObjectId object_id, const std::optional<GroundTruthPacketMap>& gt_packets) {
  // use identity if no ground truth so that we keep the csv file format
  gtsam::Pose3 gt_L_world_k = gtsam::Pose3::Identity();
  const gtsam::Pose3& L_world_k = pose;

  // gt packet provided so only log if gt pose data found
  if (gt_packets) {
    if (gt_packets->exists(frame_id)) {
      const GroundTruthInputPacket& gt_packet_k = gt_packets->at(frame_id);
      // check object exists in this frame
      ObjectPoseGT object_gt_k;
      if (!gt_packet_k.getObject(object_id, object_gt_k)) {
        // if no packet for this object found, continue and do not log
        // return false;
      } else {
        gt_L_world_k = object_gt_k.L_world_;
      }
    } else {
      // gt packet has no entry for this frame id so skip
      // TODO: for now?
      // return false;
    }
  }
  // we have either skipped logging (if gt provided by the object gt pose was
  // not found) or no gt was provided
  const auto& gt_R_k = gt_L_world_k.rotation().toQuaternion();
  // estimate
  const auto R_world_k = L_world_k.rotation().toQuaternion();
  // write out object pose with gt
  writer << frame_id << object_id << L_world_k.x() << L_world_k.y()
         << L_world_k.z() << R_world_k.x() << R_world_k.y() << R_world_k.z()
         << R_world_k.w() << gt_L_world_k.x() << gt_L_world_k.y()
         << gt_L_world_k.z() << gt_R_k.x() << gt_R_k.y() << gt_R_k.z()
         << gt_R_k.w();
  return true;
}

}  // namespace dyno
