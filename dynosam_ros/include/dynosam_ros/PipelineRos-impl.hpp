#pragma once

#include "dynosam/dataprovider/DataProvider.hpp"
#include "dynosam/pipeline/PipelineManager.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam_ros/Diagnostics.hpp"
#include "dynosam_ros/PipelineRos.hpp"

namespace dyno {

struct DynosamNodeImpl {
  rclcpp::Node& dynosam_node;
  DynoDiagnosticsTaskManager::UniquePtr diagnostics_updater;
  DynoPipelineManager::UniquePtr pipeline;

  rclcpp::CallbackGroup::SharedPtr spin_timer_group;
  rclcpp::TimerBase::SharedPtr spin_timer;

  explicit DynosamNodeImpl(rclcpp::Node& node);
  ~DynosamNodeImpl();

  std::string getDatasetPath();
  std::string getParamsPath();

  void init();

  dyno::DataProvider::Ptr createDataProvider(DynoParams& dyno_params,
                                             bool is_online);

  dyno::DataProvider::Ptr createOnlineDataProvider(DynoParams& dyno_params);

  dyno::DataProvider::Ptr createDatasetDataProvider(
      const DynoParams& dyno_params);

  std::string safeGetFilePath(const std::string& param_name,
                              const std::string& default_path,
                              const std::string& description);
};

}  // namespace dyno
