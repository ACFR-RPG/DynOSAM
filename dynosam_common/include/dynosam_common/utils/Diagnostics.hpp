#pragma once

#include <gtsam/base/FastMap.h>

#include <functional>
#include <string>

namespace dyno {

/**
 * @brief A loose copy of ROS2's diagnostic_msgs/msg/DiagnosticStatus message
 * type
 *
 */
struct DiagnosticsStatus {
  enum Level { Okay = 0, Warn = 1, Error = 2, Stale = 3 };

  Level level{Level::Stale};
  std::string name;
  std::string message;
  std::string hardware_id;
  gtsam::FastMap<std::string, std::string> values;

  void add(const std::string& key, double value) {
    values.insert2(key, std::to_string(value));
  }
};

/**
 * @brief Dynosam implementation of a Diagnostic task that fills a
 * DiagnosticsStatus.
 *
 */
class DiagnosticTaskRunner {
 public:
  DiagnosticTaskRunner() = default;
  virtual ~DiagnosticTaskRunner() = default;

  virtual bool run(DiagnosticsStatus&) { return false; }
};

}  // namespace dyno
