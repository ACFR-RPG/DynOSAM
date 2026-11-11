#pragma once

#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Accumulator.hpp"
#include "dynosam_common/utils/Timing.hpp"
#include "dynosam_ros_common/Diagnostics.hpp"

namespace dyno {

using Clock = utils::Timer::Clock;

/**
 * @brief Base SensorDiagnostics class that does nothing but provided
 * the desired API.
 *
 */
class SensorDiagnosticsBase : public DiagnosticTaskRunner {
 public:
  DYNO_POINTER_TYPEDEFS(SensorDiagnosticsBase)

  virtual ~SensorDiagnosticsBase() = default;
  virtual inline void tick(){};
  virtual inline bool run(DiagnosticsStatus&) override { return false; }
};

static constexpr int kIntervalWindowSize = 200;

/**
 * @brief Class to track and monitor incoming sensor data.
 * Mirrors diagnostic_updater::FrequencyStatus in some ways
 *
 */
class SensorDiagnostics : public SensorDiagnosticsBase {
 public:
  DYNO_POINTER_TYPEDEFS(SensorDiagnostics)

  SensorDiagnostics(const std::string& sensor_name);

  void tick();

 protected:
  bool run(DiagnosticsStatus& status) override;

 private:
  std::string sensor_name_;

  uint64_t received_count_{0};

  Clock::time_point last_message_time_{};
  bool has_received_message_{false};
  // Inter-message interval statistics (seconds).
  using IntervalAccumulator =
      utils::Accumulator<double, double, kIntervalWindowSize>;
  IntervalAccumulator intervals_;

  mutable std::mutex mutex_;
};

/**
 * @brief  Tracks the health and timing statistics of a multi-sensor
 * synchronizer. Tracks synchronization frequency, timestamp spread,
 * unsynchronizable messages and dropped messages. Call tick() for each
 * successfully synchronized measurement set, passing the timestamps of its
 * measurements.
 *
 */
class SensorSyncDiagnostics : public DiagnosticTaskRunner {
 public:
  DYNO_POINTER_TYPEDEFS(SensorSyncDiagnostics)
  explicit SensorSyncDiagnostics(const std::string& name);

  /// @brief Register a synchronized measurement set.
  /// @param timestamps Timestamps of the synchronized measurements.
  void tick(const std::vector<Timestamp>& timestamps);

  /// @brief Register a message that could not be synchronized.
  void registerUnsynchronizable();

  /// @brief Register a message discarded by the synchronizer.
  void registerDroppedMessage();

 protected:
  bool run(DiagnosticsStatus& status) override;

 private:
  std::string name_;

  static constexpr int kStatisticsWindowSize = 200;
  using TimestampDifferenceAccumulator =
      utils::Accumulator<double, double, kStatisticsWindowSize>;

  using SyncIntervalAccumulator =
      utils::Accumulator<double, double, kStatisticsWindowSize>;

  mutable std::mutex mutex_;

  uint64_t synchronized_count_{0};
  uint64_t unsynchronizable_count_{0};
  uint64_t dropped_count_{0};

  Clock::time_point last_sync_time_{};
  bool has_synchronized_{false};

  // Maximum timestamp spread within each synchronized set (milliseconds).
  TimestampDifferenceAccumulator timestamp_differences_;

  // Intervals between synchronized sets (milliseconds).
  SyncIntervalAccumulator sync_intervals_;
};

/**
 * @brief A maanger class that generates SensorDiagnostics, registering them
 * with the underling DynoDiagnosticsTaskManager.
 *
 * So we can optionally configure diagnostics, this class can also act as a
 * dummy registra (when no DynoDiagnosticsTaskManager is provided), instead an
 * instance of SensorDiagnosticsBase which does nothing.
 *
 */
class SensorDiagnosticsRegistra : public DiagnosticsRegistra {
 public:
  SensorDiagnosticsRegistra(DynoDiagnosticsTaskManager* manager);
  SensorDiagnosticsRegistra();

  SensorDiagnosticsBase::Ptr create(const std::string& sensor_name);
};

}  // namespace dyno
