#include "dynosam2_ros/impl/SystemMonitors.hpp"

#include <chrono>
#include <iomanip>
#include <unordered_map>

template <>
inline std::string dyno::to_string(const dyno2::SystemState& state) {
  using namespace dyno2;
  static const std::unordered_map<dyno2::SystemState, std::string> map = {
      {SystemState::INITIALISING, "INITIALISING"},
      {SystemState::WAITING_FOR_SENSORS, "WAITING_FOR_SENSORS"},
      {SystemState::RUNNING, "RUNNING"},
      {SystemState::SHUTTING_DOWN, "SHUTTING_DOWN"},
  };
  return map.at(state);
}

namespace dyno2 {

std::ostream& operator<<(std::ostream& os, SystemState state) {
  os << dyno::to_string(state);
  return os;
}

LifecycleMonitor::LifecycleMonitor(SystemState starting_state)
    : system_state_(starting_state),
      startup_time_(dyno::utils::Timer::tic()),
      last_processing_time_(dyno::utils::Timer::tic()) {}

void LifecycleMonitor::setState(SystemState state) { system_state_ = state; }
void LifecycleMonitor::recordHeartBeat() {
  last_processing_time_ = dyno::utils::Timer::tic();
}

bool LifecycleMonitor::run(dyno::DiagnosticsStatus& status) {
  status.name = "Lifecycle Monitor";
  status.values.insert2("Status", dyno::to_string(system_state_));

  const auto elapsed = last_processing_time_ - startup_time_;

  const auto hours = std::chrono::duration_cast<std::chrono::hours>(elapsed);
  const auto minutes =
      std::chrono::duration_cast<std::chrono::minutes>(elapsed) % 60;
  const auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(elapsed) % 60;
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(2) << hours.count() << ":"
     << std::setw(2) << minutes.count() << ":" << std::setw(2)
     << seconds.count();

  status.values.insert2("Uptime", ss.str());
  status.level = dyno::DiagnosticsStatus::Level::Okay;
  return true;
}

}  // namespace dyno2
