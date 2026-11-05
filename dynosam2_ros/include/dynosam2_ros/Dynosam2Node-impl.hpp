#pragma once
#include <filesystem>

#include "dynosam2/DynamicSlam.hpp"
#include "dynosam2/common/Parameters.hpp"
#include "dynosam2/common/StateEstimate.hpp"
#include "dynosam2_ros/DynamicSlamVisualizerRos.hpp"
#include "dynosam2_ros/Dynosam2Node.hpp"
#include "dynosam_common/utils/FileSystem.hpp"
#include "dynosam_common/utils/Spinner.hpp"
#include "dynosam_processing/DataInterfacePipeline.hpp"
#include "dynosam_processing/DataReader.hpp"
#include "dynosam_processing/PipelineHooks.hpp"
#include "dynosam_processing/PipelineModuleProcessor.hpp"
#include "dynosam_processing/datasets/DataProviderFactory.hpp"
#include "dynosam_ros_common/Diagnostics.hpp"
#include "dynosam_sensors/VIFrontendInput.hpp"

namespace dyno2 {

using DynamicSlamPipeline =
    dyno::PipelineModuleProcessor<dyno::VIFrontendInput, StateEstimate>;

struct Dynosam2Node::Impl {
 public:
  explicit Impl(rclcpp::Node& node);
  ~Impl();

 private:
  bool loadParameters(PipelineParameters& parameters);
  dyno::DataReader::Ptr createDataReader(PipelineParameters& parameters,
                                         bool* is_online);

  /**
   * @brief Construct the necessary file paths to load all configs from a
   * provided params path that is loaded from ROS2 params: "params_path"
   *
   * The provided path can either be a full path to a .yaml file in which case
   * all paramters are expected to be loaded from there or a path to a folder
   * in which case we can load camera and imu parameters separately if required.
   *
   */
  struct ParamterFileSystem {
    //! Provided path from ROS2 params. This can either be a full pfile path to
    //! a .yaml file or a folder with a dynosam2.yaml file inside it
    std::filesystem::path provided_params_path;
    std::filesystem::path dynosam2_param_file_path;
    //! May be the same as dynosam2_param_file_path
    std::filesystem::path camera_param_file_path;
    //! May be the same as dynosam2_param_file_path
    std::filesystem::path imu_param_file_path;
    //! If the provided path was a folder or the user directly provided a single
    //! .yaml file
    bool is_provided_folder;

    ParamterFileSystem(rclcpp::Node& node);
  };

 private:
  rclcpp::Node& node_;
  ParamterFileSystem params_path_;

  rclcpp::CallbackGroup::SharedPtr spin_timer_group_;
  rclcpp::TimerBase::SharedPtr spin_timer_;
  dyno::DynoDiagnosticsTaskManager::UniquePtr diagnostics_updater_;

  dyno::DataReader::Ptr data_reader_{nullptr};
  dyno::DataInterfacePipeline::UniquePtr data_interface_{nullptr};

  DynamicSlamPipeline::InputQueue input_queue_;
  // DynamicSlamPipeline::OutputQueue output_queue_;
  DynamicSlamPipeline::UniquePtr pipeline_;

  DynamicSlam::Ptr slam_;
  // Viz and pipeline!
  DynamicSlamVisualizerRos::UniquePtr viz_;

  dyno::Spinner::UniquePtr data_interface_spinner_;
  dyno::Spinner::UniquePtr pipeline_spinner_;
  dyno::Spinner::UniquePtr viz_spinner_;

  dyno::ImageDisplayQueue display_queue_;
  PublicationDataQueue publication_queue_;

  //   dyno::PipelineHooks hooks_;
};

}  // namespace dyno2
