#pragma once

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_updater/diagnostic_status_wrapper.hpp"
#include "diagnostic_updater/diagnostic_updater.hpp"
#include "diagnostic_updater/publisher.hpp"
#include "dynosam_common/utils/Diagnostics.hpp"
#include "dynosam_common/utils/Macros.hpp"

namespace dyno {

using DiagnosticTaskRos = diagnostic_updater::DiagnosticTask;
using DiagnosticStatusWrapperRos = diagnostic_updater::DiagnosticStatusWrapper;
using DiagnosticStatusRos = diagnostic_msgs::msg::DiagnosticStatus;
using DiagnosticUpdaterRos = diagnostic_updater::Updater;

/**
 * @brief An interface between the ROS level diagnostic task and the Dynosam
 * implementation of a diagnostic task (DiagnosticTaskRunner).
 *
 * The DiagnosticTaskRunner itself is stored at the DynoPipelineManager level.
 *
 */
class DynoDiagnosticsTask : public DiagnosticTaskRos {
 public:
  DYNO_POINTER_TYPEDEFS(DynoDiagnosticsTask)

  DynoDiagnosticsTask(const std::string& task_name,
                      DiagnosticTaskRunner* dyno_task);
  /* ROS level runner for the task */
  void run(DiagnosticStatusWrapperRos& stat) override;

 private:
  DiagnosticTaskRunner* dyno_task_{nullptr};
};

/**
 * @brief Simple diagnostics task manager to ensure all tasks are stored
 * properly in memory etc becuase the structure of the diagnostic_updater is...
 * odd... at best :)
 *
 */
class DynoDiagnosticsTaskManager {
 public:
  DYNO_POINTER_TYPEDEFS(DynoDiagnosticsTaskManager)

  /** Like a child we will use the NodeT syntax for old ROS2 construction */
  template <class NodeT>
  explicit DynoDiagnosticsTaskManager(NodeT node) : updater_(node) {
    // set hardware id to None to avoid warnings
    setHardwareId("none");
  }

  void registerTask(DynoDiagnosticsTask::Ptr diagnostics_task);

  /* Force the internal udpater to update if something has changed */
  void forceUpdate();
  /* Output a message on all the known DiagnosticStatus.  */
  void broadcast(unsigned char lvl, const std::string msg);
  void setHardwareId(const std::string& hwid);

 private:
  DiagnosticUpdaterRos updater_;
  // Need a vector becuase DiagnosticTaskVector stores everything with
  // references and we manage all tasks statefully so need to store the objects
  // somewhere Specifically store the dyno diagnostic task wrapper as they only
  // exist as an interface between the DiagnosticTaskRunner (which is stored in
  // the Pipeline) and the ROS level runner
  std::vector<DynoDiagnosticsTask::Ptr> dyno_diagnostics_;
};

}  // namespace dyno
