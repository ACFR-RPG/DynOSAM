#include "dynosam_ros/Diagnostics.hpp"

#include "dynosam_common/Types.hpp"  //convert

namespace dyno {

template <>
bool convert(const DiagnosticsStatus& dyno_diagnostics,
             DiagnosticStatusRos& ros_diagnostics) {
  switch (dyno_diagnostics.level) {
    case DiagnosticsStatus::Level::Okay:
      ros_diagnostics.level = DiagnosticStatusRos::OK;
      break;
    case DiagnosticsStatus::Level::Warn:
      ros_diagnostics.level = DiagnosticStatusRos::WARN;
      break;
    case DiagnosticsStatus::Level::Error:
      ros_diagnostics.level = DiagnosticStatusRos::ERROR;
      break;
    case DiagnosticsStatus::Level::Stale:
      ros_diagnostics.level = DiagnosticStatusRos::STALE;
      break;
  }

  ros_diagnostics.name = dyno_diagnostics.name;
  ros_diagnostics.message = dyno_diagnostics.message;
  ros_diagnostics.hardware_id = dyno_diagnostics.hardware_id;

  for (const auto& [key, value] : dyno_diagnostics.values) {
    ros_diagnostics.values.emplace_back();
    ros_diagnostics.values.back().key = key;
    ros_diagnostics.values.back().value = value;
  }
  return true;
}

template <>
bool convert(const DiagnosticsStatus& dyno_diagnostics,
             DiagnosticStatusWrapperRos& ros_diagnostics) {
  DiagnosticStatusRos& status =
      static_cast<DiagnosticStatusRos&>(ros_diagnostics);
  return convert(dyno_diagnostics, status);
}

DynoDiagnosticsTask::DynoDiagnosticsTask(const std::string& task_name,
                                         DiagnosticTaskRunner* dyno_task)
    : diagnostic_updater::DiagnosticTask(task_name), dyno_task_(dyno_task) {}

void DynoDiagnosticsTask::run(DiagnosticStatusWrapperRos& stat) {
  if (dyno_task_) {
    // collect diagnostics internally and publish if available
    DiagnosticsStatus dyno_diagnostics;
    if (dyno_task_->run(dyno_diagnostics)) {
      // convert to diagnostics status wrapper
      convert(dyno_diagnostics, stat);
    }
  } else {
    // task is somehow null
    // report this
    stat.summary(DiagnosticStatusRos::ERROR, "DynoTask is invalid and null");
  }
}

void DynoDiagnosticsTaskManager::registerTask(
    DynoDiagnosticsTask::Ptr diagnostics_task) {
  // properly store the task in memory and provide a reference to the updater
  dyno_diagnostics_.push_back(diagnostics_task);
  updater_.add(*diagnostics_task);
}

void DynoDiagnosticsTaskManager::forceUpdate() { updater_.force_update(); }
void DynoDiagnosticsTaskManager::broadcast(unsigned char lvl,
                                           const std::string msg) {
  updater_.broadcast(lvl, msg);
}
void DynoDiagnosticsTaskManager::setHardwareId(const std::string& hwid) {
  updater_.setHardwareID(hwid);
}

}  // namespace dyno
