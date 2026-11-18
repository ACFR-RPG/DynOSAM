#pragma once

#include <cstddef>
#include <mutex>

#include "dynosam_common/utils/Accumulator.hpp"
#include "dynosam_common/utils/Timing.hpp"

namespace dyno {

static constexpr int kRollingIntervalWindowSize{200};

template <typename Accumulator =
              utils::Accumulatord<kRollingIntervalWindowSize>>
struct IntervalStats {
  Accumulator intervals;

  inline int intervalSamples() const { return intervals.total_samples(); }
  inline double rollingMean() const {
    return (intervalSamples() > 0) ? intervals.RollingMean() : 0.0;
  }
  inline double rollingStdDev() const {
    return (intervalSamples() > 0) ? intervals.StandardDeviation() : 0.0;
  }
  inline double frequency() const {
    // does recompute rolling mean but oh well
    const double rolling_mean = rollingMean();
    return rolling_mean > 0.0 ? 1000.0 / rolling_mean : 0.0;
  }

  /// @brief Computes frequency and mean from rolling statistics
  /// @return std::pair<double, double>. {Frequency, Rolling Mean}
  inline std::pair<double, double> frequencyAndRollingMean() const {
    const double rolling_mean = rollingMean();
    const double frequency = rolling_mean > 0.0 ? 1000.0 / rolling_mean : 0.0;
    return {frequency, rolling_mean};
  }
};

template <std::size_t WindowSize = 200>
struct RollingIntervalDelta {
  using Accumulator = utils::Accumulatord<WindowSize>;

  using Stats = IntervalStats<Accumulator>;
  Stats stats;
  mutable std::mutex mutex;

  void tickDelta(double dt_ms) {
    std::lock_guard<std::mutex> lock(mutex);
    stats.intervals.Add(dt_ms);
  }

  /// @brief Get threadsafe stats copy
  /// @return
  Stats getStats() const {
    std::lock_guard<std::mutex> lock(mutex);
    return stats;
  }
};

/**
 * @brief Class to help calculating the rolling intervial of some time delta,
 * calculating upon tick!
 *
 * Computes time detla using milliseconds.
 *
 * @tparam Interval
 */
template <std::size_t WindowSize = 200, typename Clock = utils::Timer::Clock>
struct RollingInterval {
  using Accumulator = utils::Accumulatord<WindowSize>;

  struct Stats : public IntervalStats<Accumulator> {
    uint64_t received_count{0};
    bool has_received_message{false};
    typename Clock::time_point last_message_time{};
  };

  Stats stats;
  mutable std::mutex mutex;

  void tick() {
    const auto now = Clock::now();

    std::lock_guard<std::mutex> lock(mutex);
    ++stats.received_count;

    if (stats.has_received_message) {
      // compute timig internval in milliseconds
      const double interval =
          utils::Timer::diffUnits<std::milli>(now, stats.last_message_time);
      stats.intervals.Add(interval);
    }

    stats.last_message_time = now;
    stats.has_received_message = true;
  }

  /// @brief Get threadsafe stats copy
  /// @return
  Stats getStats() const {
    std::lock_guard<std::mutex> lock(mutex);
    return stats;
  }
};

}  // namespace dyno
