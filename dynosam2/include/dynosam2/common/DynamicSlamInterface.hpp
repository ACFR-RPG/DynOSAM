#pragma once

#include "dynosam2/common/StateEstimate.hpp"
#include "dynosam2/common/Types.hpp"
#include "dynosam_common/ModuleBase.hpp"
#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Diagnostics.hpp"
#include "dynosam_common/utils/RollingInterval.hpp"
#include "dynosam_core/DynoState.hpp"
#include "dynosam_processing/ImageViewer.hpp"
#include "dynosam_processing/ThreadSafeQueue.hpp"
#include "dynosam_sensors/VIFrontendInput.hpp"

namespace dyno2 {

/// @brief Realtime publishing data struct.
struct PublicationData {
  DYNO_POINTER_TYPEDEFS(PublicationData)

  dyno::FrameId frame_id;
  dyno::Timestamp timestamp;

  PublicationData(dyno::FrameId fi, dyno::Timestamp t)
      : frame_id(fi), timestamp(t) {}
  PublicationData() {}

  // tracking state information
  dyno::ViTrackingState vi_tracking_state;
  //! Set of co-visible frames involved in the most recent bundle adjustment
  std::unordered_set<dyno::FrameId> vi_covisables;
  //! This also indicates well-tracked/observed objects!
  std::vector<dyno::DoTrackingState> do_tracking_states;

  // TODO: draw links between affected objects: covisible!
  UpdatedStates updated_states;
};

class DynamicSlamDiagnostics;

class DynamicSlamInterface
    : public dyno::ModuleBase<dyno::VIFrontendInput, StateEstimate> {
 public:
  typedef dyno::ModuleBase<dyno::VIFrontendInput, StateEstimate> Base;

  DynamicSlamInterface();
  virtual ~DynamicSlamInterface() = default;

  void setPublicationQueue(ThreadsafeQueue<PublicationData::ConstPtr>* queue);
  void setImageDisplayQueue(dyno::ImageDisplayQueue* queue);

  bool pushImageToDisplayQueue(const std::string& wname, const cv::Mat& image);
  bool pushPublicationDataToQueue(PublicationData::ConstPtr publication_data);

  std::shared_ptr<DynamicSlamDiagnostics> getDiagnostics() const {
    return diagnostics_;
  }

 protected:
  virtual StateEstimate::Ptr spinOnce(
      dyno::VIFrontendInput::ConstPtr input) = 0;

 private:
  inline Base::SpinReturn boostrapSpin(
      dyno::VIFrontendInput::ConstPtr input) override {
    return {Base::State::Nominal, spinOnce(input)};
  }
  inline Base::SpinReturn nominalSpin(
      dyno::VIFrontendInput::ConstPtr input) override {
    return {Base::State::Nominal, spinOnce(input)};
  }
  void validateInput(
      const dyno::VIFrontendInput::ConstPtr& input) const override;

 private:
  ThreadsafeQueue<PublicationData::ConstPtr>* publication_queue_{nullptr};
  dyno::ImageDisplayQueue* image_display_queue_{nullptr};

 protected:
  std::shared_ptr<DynamicSlamDiagnostics> diagnostics_;
};

struct TimeDetail {
  /// @brief Most recent actual time according to the ROS clock
  dyno::Timestamp real_time{0};
  /// @brief Most recent time of the incoming (synced) image container
  dyno::Timestamp image_time{0};
  /// @brief Associated frame id
  dyno::FrameId fid{0};
};

class DynamicSlamDiagnostics : public dyno::DiagnosticTaskRunner {
 public:
  /// @}
  /// @name Static diagnostic data that should be set once at the start (ie not
  /// thread safe!)
  /// @{
  bool enforcing_realtime{false};
  bool using_full_graph{false};
  bool estimating_do{true};
  bool using_imu{false};
  double realtime_time_limit{0.00};
  // number of frames recieved
  std::atomic<std::size_t> num_frames_processed{0};

  DynamicSlamDiagnostics() = default;

  /// @}
  /// @name Thread-safe tick functions
  /// @{
  void tickFrameProcessed(dyno::Timestamp frame_timestamp,
                          dyno::Timestamp compute_dt);
  inline void tickDetectAndTrack(double dt_ms) {
    detect_and_track_ri_.tickDelta(dt_ms);
  }
  inline void tickOdometries(double dt_ms) { odometries_ri_.tickDelta(dt_ms); }

  /**
   * @brief Update the most recent TimeDetail which represents
   * the most recent actual ros time and the most time/fid of the most recent
   * synchronized image data in the incoming image queue.
   *
   * This data is then compared with the most recent processed data to check
   * for lag between the processing and the incoming data
   *
   * @param time_detail
   */
  void updateImageSyncTimeDetail(const TimeDetail& time_detail);

 protected:
  bool run(dyno::DiagnosticsStatus& status) override;

 private:
  using RollingIntervalT = dyno::RollingInterval<200>;
  /// @brief Output rate of the entire system, inherently also includes wait
  /// time between ticks
  RollingIntervalT pipeline_output_ri_;

  using RollingIntervalDeltaT = dyno::RollingIntervalDelta<200>;
  RollingIntervalDeltaT detect_and_track_ri_;
  RollingIntervalDeltaT odometries_ri_;
  /// @brief The actual dt to process/compute a single frame; unlike
  /// pipeline_output_ri_ which which also accounts for waiting time between
  /// ticks
  RollingIntervalDeltaT compute_ri_;

  /// @brief Timelag interval compared to the incoming datastream
  dyno::RollingIntervalDelta<50> time_lag_ri_;

  /// @brief Timing information about the most recent synchronized
  /// ImageContainer that is in the processing queue
  TimeDetail image_sync_detail_;
  /// @brief Timestamp of the most recent processed frame
  dyno::Timestamp most_recent_frame_timestamp_;

  std::mutex mutex_;
};

}  // namespace dyno2
