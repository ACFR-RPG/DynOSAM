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

class SensorDiagnosticTask : public DiagnosticTaskRos {
  /**
   * @brief Simple diagnostics task manager to ensure all tasks are stored
   * properly in memory etc becuase the structure of the diagnostic_updater
   * is... odd... at best :)
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

    void registerTask(
        const std::string& name,
        const std::shared_ptr<dyno::DiagnosticTaskRunner>& diagnostic_task) {
      // properly store the task in memory
      dyno_diagnostic_tasks_.push_back(diagnostic_task);
      registerTask(
          std::make_shared<DynoDiagnosticsTask>(name, diagnostic_task.get()));
    }

    void registerTask(
        const std::shared_ptr<dyno::DiagnosticTaskRos>& diagnostic_task) {
      ros_diagnostics_shared_.push_back(diagnostic_task);
      updater_.add(*diagnostic_task);
    }

    /**
     * @brief Register an external ROS diagnostic task without taking memory
     * ownership of the object.
     *
     * @param diagnostic_task DiagnosticTaskRos&
     */
    void registerTask(DiagnosticTaskRos& diagnostic_task) {
      updater_.add(diagnostic_task);
    }

    /* Force the internal udpater to update if something has changed */
    void forceUpdate();
    /* Output a message on all the known DiagnosticStatus.  */
    void broadcast(unsigned char lvl, const std::string msg);
    void setHardwareId(const std::string& hwid);

   private:
    DiagnosticUpdaterRos updater_;
    // Need a vector becuase DiagnosticTaskVector stores everything with
    // references and we manage all tasks statefully so need to store the
    // objects somewhere to avoid them going out of scope (and deallocating
    // memory!)
    std::vector<std::shared_ptr<DiagnosticTaskRos>> ros_diagnostics_shared_;
    // dyno tasks that were created via heap memory
    std::vector<std::shared_ptr<DiagnosticTaskRunner>> dyno_diagnostic_tasks_;
  };

}  // namespace dyno
