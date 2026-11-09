#pragma once

#include <mutex>

#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Diagnostics.hpp"
#include "dynosam_common/utils/Timing.hpp"

namespace dyno2 {

enum class SystemState {
  INITIALISING,
  WAITING_FOR_SENSORS,
  RUNNING,
  SHUTTING_DOWN
};

std::ostream& operator<<(std::ostream& os, SystemState state);

class LifecycleMonitor : public dyno::DiagnosticTaskRunner {
 public:
  LifecycleMonitor(SystemState starting_state = SystemState::INITIALISING);
  void setState(SystemState state);
  void setSensorMode(const std::string& sensor_mode);
  void recordHeartBeat();

 protected:
  bool run(dyno::DiagnosticsStatus& status) override;

 private:
  SystemState system_state_;
  // Some metadata about what the source of input is
  // If offline this will just be dataset
  std::string sensor_mode_;

  dyno::utils::Timer::TimePoint startup_time_;
  dyno::utils::Timer::TimePoint last_processing_time_;
};

}  // namespace dyno2

template <>
std::string dyno::to_string(const dyno2::SystemState& state);
