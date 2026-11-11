#include "dynosam_ros_common/diagnostics/SensorDiagnostics.hpp"

#include "dynosam_common/utils/Accumulator.hpp"
#include "dynosam_common/utils/Timing.hpp"

namespace dyno {

SensorDiagnostics::SensorDiagnostics(const std::string& sensor_name)
    : sensor_name_(sensor_name) {}

void SensorDiagnostics::tick() {
  const auto now = Clock::now();

  std::lock_guard<std::mutex> lock(mutex_);
  ++received_count_;

  if (has_received_message_) {
    // compute timig internval in milliseconds
    const double interval =
        utils::Timer::diffUnits<std::milli>(now, last_message_time_);
    intervals_.Add(interval);
  }

  last_message_time_ = now;
  has_received_message_ = true;
}

bool SensorDiagnostics::run(DiagnosticsStatus& status) {
  // Snapshot state under the mutex.
  uint64_t received_count;
  bool has_received_message;
  Clock::time_point last_message_time;
  IntervalAccumulator intervals;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    received_count = received_count_;
    has_received_message = has_received_message_;
    last_message_time = last_message_time_;
    intervals = intervals_;
  }

  status.add("Sensor", sensor_name_);
  status.add("Messages received", received_count);

  if (!has_received_message) {
    status.summary(DiagnosticsStatus::Level::Warn,
                   "Waiting for sensor messages");
    return true;
  }

  const auto now = Clock::now();

  // age in milliseconds
  const double message_age =
      utils::Timer::diffUnits<std::milli>(now, last_message_time);

  const int interval_samples = intervals.total_samples();

  // Lifetime statistics.
  const double lifetime_mean = intervals.Mean();

  // Rolling statistics.
  double rolling_mean = 0.0;
  double rolling_stddev = 0.0;

  if (interval_samples > 0) {
    rolling_mean = intervals.RollingMean();
    rolling_stddev = intervals.StandardDeviation();
  }

  const double frequency = rolling_mean > 0.0 ? 1000.0 / rolling_mean : 0.0;

  // Adaptive timeout: at least 1 second, or 3x the observed period.
  const double timeout = std::max(1000.0, 3.0 * rolling_mean);

  // Reception statistics.
  status.add("Message age (ms)", message_age);
  status.add("Reception frequency (Hz)", frequency);

  // Lifetime inter-message statistics.
  status.add("Lifetime mean period (ms)", lifetime_mean);

  if (interval_samples > 0) {
    status.add("Lifetime min period (ms)", intervals.min());
    status.add("Lifetime max period (ms)", intervals.max());

    // Rolling inter-message statistics.
    status.add("Rolling mean period (ms)", rolling_mean);
    status.add("Rolling stddev period (ms)", rolling_stddev);
  }

  status.add("Interval samples", interval_samples);
  status.add("Reception timeout (ms)", timeout);

  // Diagnostic health.
  if (message_age > timeout) {
    std::ostringstream message;
    message << "Sensor message timeout: last message received " << std::fixed
            << std::setprecision(2) << message_age << " milliseconds ago";

    status.summary(DiagnosticsStatus::Level::Error, message.str());
  } else {
    status.summary(DiagnosticsStatus::Level::Okay,
                   "Sensor receiving messages normally");
  }

  return true;
}

SensorSyncDiagnostics::SensorSyncDiagnostics(const std::string& name)
    : name_(name) {}

void SensorSyncDiagnostics::tick(const std::vector<Timestamp>& timestamps) {
  CHECK_GE(timestamps.size(), 2);

  const auto [min_it, max_it] =
      std::minmax_element(timestamps.begin(), timestamps.end());

  const double timestamp_difference_ms = (*max_it - *min_it) * 1000.0;

  const auto now = Clock::now();

  std::lock_guard<std::mutex> lock(mutex_);

  ++synchronized_count_;

  // milliseconds
  timestamp_differences_.Add(timestamp_difference_ms);

  if (has_synchronized_) {
    // milliseconds
    const double interval =
        utils::Timer::diffUnits<std::milli>(now, last_sync_time_);
    sync_intervals_.Add(interval);
  }

  last_sync_time_ = now;
  has_synchronized_ = true;
}

void SensorSyncDiagnostics::registerUnsynchronizable() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++unsynchronizable_count_;
}

void SensorSyncDiagnostics::registerDroppedMessage() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++dropped_count_;
}

bool SensorSyncDiagnostics::run(DiagnosticsStatus& status) {
  uint64_t synchronized_count;
  uint64_t unsynchronizable_count;
  uint64_t dropped_count;

  bool has_synchronized;
  Clock::time_point last_sync_time;

  TimestampDifferenceAccumulator timestamp_differences;
  SyncIntervalAccumulator sync_intervals;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    synchronized_count = synchronized_count_;
    unsynchronizable_count = unsynchronizable_count_;
    dropped_count = dropped_count_;

    has_synchronized = has_synchronized_;
    last_sync_time = last_sync_time_;

    timestamp_differences = timestamp_differences_;
    sync_intervals = sync_intervals_;
  }

  status.add("Synchronizer", name_);
  status.add("Synchronized sets", synchronized_count);
  status.add("Unsynchronizable messages", unsynchronizable_count);
  status.add("Dropped messages", dropped_count);

  if (!has_synchronized) {
    status.summary(DiagnosticsStatus::Level::Warn,
                   "Waiting for synchronized sensor measurements");
    return true;
  }

  const auto now = Clock::now();

  // milliseconds
  const double sync_age =
      utils::Timer::diffUnits<std::milli>(now, last_sync_time);

  const int sync_interval_samples = sync_intervals.total_samples();

  const double mean_sync_period = sync_intervals.Mean();
  const double rolling_sync_period = sync_intervals.RollingMean();

  const double sync_frequency =
      rolling_sync_period > 0.0 ? 1000.0 / rolling_sync_period : 0.0;

  const int timestamp_samples = timestamp_differences.total_samples();

  const double mean_timestamp_difference = timestamp_differences.Mean();

  const double rolling_timestamp_difference =
      timestamp_differences.RollingMean();

  const double timestamp_difference_stddev =
      timestamp_differences.StandardDeviation();

  const double timeout = std::max(1000.0, 3.0 * rolling_sync_period);

  status.add("Time since last synchronization (ms)", sync_age);
  status.add("Synchronization frequency (Hz)", sync_frequency);

  if (sync_interval_samples > 0) {
    status.add("Mean synchronization period (ms)", mean_sync_period);

    status.add("Rolling synchronization period (ms)", rolling_sync_period);
  }

  status.add("Mean timestamp difference (ms)", mean_timestamp_difference);

  status.add("Rolling timestamp difference (ms)", rolling_timestamp_difference);

  status.add("Timestamp difference stddev (ms)", timestamp_difference_stddev);

  if (timestamp_samples > 0) {
    status.add("Max timestamp difference (ms)", timestamp_differences.max());
  }

  status.add("Synchronization samples", sync_interval_samples);
  status.add("Timestamp samples", timestamp_samples);
  status.add("Synchronization timeout (ms)", timeout);

  if (sync_age > timeout) {
    std::ostringstream message;
    message << "Synchronization timeout: last synchronized set produced "
            << std::fixed << std::setprecision(2) << sync_age << " seconds ago";

    status.summary(DiagnosticsStatus::Level::Error, message.str());
  } else {
    status.summary(DiagnosticsStatus::Level::Okay,
                   "Sensor synchronization operating normally");
  }

  return true;
}

SensorDiagnosticsRegistra::SensorDiagnosticsRegistra(
    DynoDiagnosticsTaskManager* manager)
    : DiagnosticsRegistra(CHECK_NOTNULL(manager)) {}

SensorDiagnosticsRegistra::SensorDiagnosticsRegistra()
    : DiagnosticsRegistra(nullptr) {}

SensorDiagnosticsBase::Ptr SensorDiagnosticsRegistra::create(
    const std::string& sensor_name) {
  if (isValid()) {
    auto diagnostics = std::make_shared<SensorDiagnostics>(sensor_name);
    registerTask(sensor_name, diagnostics);
    return diagnostics;
  }
  // return base class that does nothing
  return std::make_shared<SensorDiagnosticsBase>();
}

}  // namespace dyno
