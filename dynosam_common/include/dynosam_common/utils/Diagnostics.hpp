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

  template <typename T>
  void add(const std::string& key, T value) {
    values.insert2(key, std::to_string(value));
  }

  void summary(Level lvl, const std::string& s) {
    level = lvl;
    message = s;
  }
};

template <>
inline void DiagnosticsStatus::add<const std::string&>(
    const std::string& key, const std::string& value) {
  values.insert2(key, value);
}

template <>
inline void DiagnosticsStatus::add<std::string>(const std::string& key,
                                                std::string value) {
  values.insert2(key, value);
}

template <>
inline void DiagnosticsStatus::add<const char*>(const std::string& key,
                                                const char* value) {
  values.insert2(key, std::string(value));
}

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
