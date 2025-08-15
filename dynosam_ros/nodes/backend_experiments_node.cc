/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include <glog/logging.h>

#include <dynosam/backend/BackendFactory.hpp>
#include <dynosam/common/Map.hpp>
#include <dynosam/frontend/RGBDInstanceFrontendModule.hpp>
#include <dynosam/logger/Logger.hpp>

#include "dynosam_ros/PipelineRos.hpp"
#include "dynosam_ros/Utils.hpp"
#include "rclcpp/executor.hpp"
#include "rclcpp/rclcpp.hpp"

// already defined in DataProviderFactory.cc
DECLARE_int32(ending_frame);

namespace dyno {

class BackendExperimentsNode : public DynoNode {
 public:
  BackendExperimentsNode() : DynoNode("dynosam_experiments") {
    RCLCPP_INFO_STREAM(this->get_logger(), "Starting BackendExperimentsNode");

    auto data_loader = this->createDataProvider();
    auto params = this->getDynoParams();

    CameraParams camera_params;
    if (params.preferDataProviderCameraParams() &&
        data_loader->getCameraParams().has_value()) {
      LOG(INFO) << "Using camera params from DataProvider, not the config in "
                   "the CameraParams.yaml!";
      camera_params = *data_loader->getCameraParams();
    } else {
      LOG(INFO) << "Using camera params specified in CameraParams.yaml!";
      camera_params = params.camera_params_;
    }

    Camera::Ptr camera = std::make_shared<Camera>(camera_params);

    using BackendModuleTraits = RegularBackendModule::ModuleTraits;

    LOG(INFO) << "Offline RGBD frontend";
    const std::string file_path =
        getOutputFilePath(kRgbdFrontendOutputJsonFile);

    using OfflineFrontend =
        FrontendOfflinePipeline<RegularBackendModule::ModuleTraits>;

    OfflineFrontend::UniquePtr offline_frontend =
        std::make_unique<OfflineFrontend>("offline-rgbdfrontend", file_path,
                                          FLAGS_ending_frame);

    // raw ptr type becuase we cannot copy the unique ptr!! This is only becuase
    // we need it in the lambda function which is a temporary solution
    OfflineFrontend* offline_frontend_ptr = offline_frontend.get();

    BackendModule::Ptr backend = nullptr;
    params.backend_params_.full_batch_frame = offline_frontend->endingFrame();

    const auto& backend_type = params.backend_type;
    backend = BackendFactory::createModule(backend_type, params.backend_params_,
                                           camera);
    CHECK(backend);

    backend_pipeline_ = std::make_unique<BackendPipeline>(
        "backend-pipeline", &backend_input_queue_, backend);
    backend_pipeline_->parallelRun(params.parallelRun());
    // also register connection between front and back
    offline_frontend_ptr->registerOutputQueue(&backend_input_queue_);
    // NO OUTPUT!!

    // convert pipeline to base type
    frontend_pipeline_ = std::move(offline_frontend);
  }

  // TODO: copy pasted from DynoManager class - streamline/make shutdown hook
  ~BackendExperimentsNode() {
    // TODO: make shutdown hook!
    writeStatisticsSamplesToFile("statistics_samples.csv");
    writeStatisticsModuleSummariesToFile();
  }

  bool spinOnce() override {
    RCLCPP_INFO_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                getStats());
    if (frontend_pipeline_->isWorking()) {
      frontend_pipeline_->spinOnce();
      backend_pipeline_->spinOnce();
      return true;
    }
    return false;
  }

 private:
  PipelineBase::UniquePtr frontend_pipeline_{nullptr};
  BackendPipeline::UniquePtr backend_pipeline_{nullptr};
  FrontendPipeline::OutputQueue backend_input_queue_;
};

}  // namespace dyno

int main(int argc, char* argv[]) {
  auto non_ros_args = dyno::initRosAndLogging(argc, argv);

  rclcpp::NodeOptions options;
  options.arguments(non_ros_args);
  options.use_intra_process_comms(true);

  rclcpp::executors::SingleThreadedExecutor exec;
  auto ros_pipeline = std::make_shared<dyno::BackendExperimentsNode>();

  exec.add_node(ros_pipeline);
  while (rclcpp::ok()) {
    if (!ros_pipeline->spinOnce()) {
      break;
    }
    exec.spin_some();
  }

  ros_pipeline.reset();
}
