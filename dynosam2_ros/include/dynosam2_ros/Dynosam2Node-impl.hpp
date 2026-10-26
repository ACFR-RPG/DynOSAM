#pragma once
#include "dynosam2/DynamicSlam.hpp"
#include "dynosam2/common/Parameters.hpp"
#include "dynosam2/common/StateEstimate.hpp"
#include "dynosam2_ros/DynamicSlamVisualizerRos.hpp"
#include "dynosam2_ros/Dynosam2Node.hpp"
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

 private:
  rclcpp::Node& node_;
  std::string params_path_;

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
