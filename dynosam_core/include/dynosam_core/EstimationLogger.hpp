#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_common/utils/CsvParser.hpp"
#include "dynosam_core/GroundTruthPacket.hpp"

namespace dyno {

class EstimationModuleLogger {
 public:
  DYNO_POINTER_TYPEDEFS(EstimationModuleLogger)

  /**
   * @brief Construct a new Estimation Module Logger object.
   *
   * Module name is prefixed to the output file names which are hardcoded per
   * log function. Unless otherwise specified, all the (base) functions operate
   * per frame.
   *
   * Will write to the file path sepcified by FLAGS_output_path
   *
   * @param module_name
   */
  EstimationModuleLogger(const std::string& module_name);
  // write to file on destructor
  virtual ~EstimationModuleLogger();

  // logs to motion errors
  size_t logObjectTrajectory(
      FrameId frame_id, const MultiObjectTrajectories& object_trajectories,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

  size_t logObjectTrajectory(
      const MultiObjectTrajectories& object_trajectories,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

  size_t logCameraPose(
      FrameId frame_id, const PoseTrajectory& camera_poses,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

  size_t logCameraPose(
      const PoseTrajectory& camera_poses,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

  virtual void logPoints(FrameId frame_id, const gtsam::Pose3& X_W_k,
                         const StatusLandmarkVector& landmarks);

  virtual void logMapPoints(const StatusLandmarkVector& landmarks);

  inline const std::string& moduleName() const { return module_name_; }

 protected:
  inline long int timestampToNano(Timestamp timestamp) const {
    return timestamp * 1e+9;
  }

  void logObjectSE3(CsvWriter& writer, const gtsam::Pose3& se3_est,
                    const gtsam::Pose3& se3_gt, Timestamp timestamp,
                    FrameId frame_id, ObjectId object_id);

  void logCameraSE3(CsvWriter& writer, const gtsam::Pose3& se3_est,
                    const gtsam::Pose3& se3_gt, Timestamp timestamp,
                    FrameId frame_id);

  virtual bool logObjectTrajectoryEntry(
      const PoseWithMotionEntry& entry, const ObjectId object_id,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

  virtual bool logCameraPoseEntry(
      const PoseTrajectoryEntry& entry,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

  std::optional<ObjectPoseGT> getObjectGroundTruthHelper(
      FrameId frame_id, ObjectId object_id,
      const std::optional<GroundTruthPacketMap>& gt_packets) const;

 protected:
  const std::string module_name_;
  const std::string object_pose_file_name_;
  const std::string object_motion_file_name_;
  const std::string camera_pose_file_name_;
  const std::string map_points_file_name_;

  CsvWriter::UniquePtr object_pose_csv_;
  CsvWriter::UniquePtr object_motion_csv_;
  CsvWriter::UniquePtr camera_pose_csv_;
  CsvWriter::UniquePtr map_points_csv_;
};

}  // namespace dyno
