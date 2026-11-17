#include "dynosam2/common/DynamicSlamInterface.hpp"

#include "dynosam_common/utils/Diagnostics.hpp"

namespace dyno2 {

// should be in common but whatever
struct InvalidImageContainerException : public dyno::DynosamException {
  InvalidImageContainerException(const dyno::ImageContainer& container,
                                 const std::string& what)
      : dyno::DynosamException(
            "Image container with config: " + container.toString() +
            "\n was invalid - " + what) {}
};

DynamicSlamInterface::DynamicSlamInterface()
    : Base("dynosam2"),
      diagnostics_(std::make_shared<DynamicSlamDiagnostics>()) {}

void DynamicSlamInterface::setPublicationQueue(
    ThreadsafeQueue<PublicationData::ConstPtr>* queue) {
  CHECK_NOTNULL(queue);
  publication_queue_ = queue;
}

void DynamicSlamInterface::setImageDisplayQueue(
    dyno::ImageDisplayQueue* queue) {
  CHECK_NOTNULL(queue);
  image_display_queue_ = queue;
}

bool DynamicSlamInterface::pushImageToDisplayQueue(const std::string& wname,
                                                   const cv::Mat& image) {
  if (!image_display_queue_) {
    return false;
  }

  image_display_queue_->push(dyno::ImageToDisplay(wname, image));
  return true;
}

bool DynamicSlamInterface::pushPublicationDataToQueue(
    PublicationData::ConstPtr publication_data) {
  if (!publication_queue_) {
    return false;
  }

  publication_queue_->push(publication_data);
  return true;
}

void DynamicSlamInterface::validateInput(
    const dyno::VIFrontendInput::ConstPtr& input) const {
  const auto image_container = input->image_container_;

  const bool has_rgb = image_container->hasRgb();
  const bool has_depth_image = image_container->hasDepth();
  const bool has_stereo = image_container->hasRightRgb();

  if (!has_rgb) {
    throw InvalidImageContainerException(*image_container, "Missing RGB");
  }

  if (!has_depth_image && !has_stereo) {
    throw InvalidImageContainerException(*image_container,
                                         "Missing Depth or Stereo");
  }
}

void DynamicSlamDiagnostics::tickFrameProcessed(dyno::Timestamp frame_timestamp,
                                                dyno::Timestamp compute_dt) {
  TimeDetail image_sync_detail;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    most_recent_frame_timestamp_ = frame_timestamp;
    image_sync_detail = image_sync_detail_;
  }

  ++num_frames_processed;
  pipeline_output_ri_.tick();

  const double dt = image_sync_detail.image_time - frame_timestamp;
  time_lag_ri_.tickDelta(dt);
  compute_ri_.tickDelta(compute_dt);
}

void DynamicSlamDiagnostics::updateImageSyncTimeDetail(
    const TimeDetail& time_detail) {
  std::lock_guard<std::mutex> lock(mutex_);
  image_sync_detail_ = time_detail;
}

bool DynamicSlamDiagnostics::run(dyno::DiagnosticsStatus& status) {
  // fill out static diagnostics
  status.add("Enforcing Realtime", enforcing_realtime);
  if (enforcing_realtime) {
    status.add("Realtime limit [ms]", realtime_time_limit);
  }
  status.add("Using full-graph", using_full_graph);
  status.add("Estimating Dynamic Objects", estimating_do);
  status.add("Using IMU", using_imu);

  // write out to local variable for thread-safe access
  const std::size_t num = this->num_frames_processed;
  status.add("Frames processed", num);
  if (num == 0) {
    status.summary(dyno::DiagnosticsStatus::Level::Warn, "No frames processed");
    return true;
  }

  status.summary(dyno::DiagnosticsStatus::Level::Okay, "Processing is okay");

  {
    const auto stats = compute_ri_.getStats();
    const int interval_samples = stats.intervals.total_samples();

    const double lifetime_mean = stats.intervals.Mean();

    const double rolling_stddev = stats.rollingStdDev();
    const auto [frequency, rolling_mean] = stats.frequencyAndRollingMean();

    // status.add("Compute Frequency (Hz)", frequency);
    status.add("Compute lifetime mean period (ms)", lifetime_mean);

    if (interval_samples > 0) {
      status.add("Compute rolling mean period (ms)", rolling_mean);
      status.add("Compute rolling stddev period (ms)", rolling_stddev);
    }

    if (enforcing_realtime) {
      const double approx_realtime_time_limit = 1.2 * realtime_time_limit;

      // Prefer rolling statistics when available.
      const double compute_mean =
          interval_samples > 0 ? rolling_mean : lifetime_mean;

      if (compute_mean > approx_realtime_time_limit) {
        status.mergeSummary(dyno::DiagnosticsStatus::Level::Warn,
                            "Realtime processing constraint is not being met!");
      }
    }
  }

  // mean dt of the entire pipeline, which also depends on input sensor rate
  double pipeline_output_mean_ms{0.0};
  {
    const auto stats = pipeline_output_ri_.getStats();
    const int interval_samples = stats.intervals.total_samples();

    const double lifetime_mean = stats.intervals.Mean();
    pipeline_output_mean_ms = lifetime_mean;

    const double rolling_stddev = stats.rollingStdDev();
    const auto [frequency, rolling_mean] = stats.frequencyAndRollingMean();

    status.add("Pipeline Frequency (Hz)", frequency);
    status.add("Pipeline lifetime mean period (ms)", lifetime_mean);

    if (interval_samples > 0) {
      status.add("Processing rolling mean period (ms)", rolling_mean);
      status.add("Processing rolling stddev period (ms)", rolling_stddev);

      pipeline_output_mean_ms = rolling_mean;
    }
  }

  {
    const auto stats = time_lag_ri_.getStats();
    const int interval_samples = stats.intervals.total_samples();

    const double lifetime_mean = stats.intervals.Mean();
    const double rolling_stddev = stats.rollingStdDev();
    const double rolling_mean = stats.rollingMean();

    status.add("Lag lifetime mean (ms)", lifetime_mean);

    if (interval_samples > 0) {
      status.add("Lag rolling mean (ms)", rolling_mean);
      status.add("Lag rolling stddev (ms)", rolling_stddev);
    }

    // Adaptive synchronisation lag threshold.
    // Based on a minimum acceptable processing frequency of 12 Hz.
    constexpr double kMinProcessingHz = 12.0;
    constexpr double kMinProcessingPeriodMs = 1000.0 / kMinProcessingHz;

    const double lag_timeout =
        3.0 * std::max(kMinProcessingPeriodMs, pipeline_output_mean_ms);

    status.add("Lag timeout (ms)", lag_timeout);

    // Prefer rolling statistics when available.
    const double lag_mean = interval_samples > 0 ? rolling_mean : lifetime_mean;

    if (interval_samples > 0 || stats.intervals.total_samples() > 0) {
      if (lag_mean > lag_timeout) {
        status.mergeSummary(
            dyno::DiagnosticsStatus::Level::Warn,
            "Synchronisation lag consistently exceeds acceptable threshold");
      }
    }
  }

  return true;
}

}  // namespace dyno2
