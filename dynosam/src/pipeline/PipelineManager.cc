/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/pipeline/PipelineManager.hpp"

#include <glog/logging.h>

#include "dynosam/backend/BackendFactory.hpp"
#include "dynosam/frontend/PoseChangeVIFrontend.hpp"
#include "dynosam/frontend/RegularVIFrontend.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

DEFINE_bool(use_backend, false, "If any backend should be initalised");
DEFINE_bool(use_opencv_display, true,
            "If true, the OpenCVImageDisplayQueue will be processed");

namespace dyno {

DynoPipelineManager::DynoPipelineManager(
    const DynoParams& params, DataProvider::Ptr data_loader,
    FrontendDisplay::Ptr frontend_display, BackendDisplay::Ptr backend_display,
    BackendModuleFactory::Ptr factory, const ExternalHooks::Ptr external_hooks)
    : params_(params),
      data_loader_(std::move(data_loader)),
      displayer_(&display_queue_, params.parallelRun())

{
  LOG(INFO) << "Starting DynoPipelineManager";

  CHECK(data_loader_);
  CHECK(frontend_display);

  data_interface_ =
      std::make_unique<DataInterfacePipeline>(params_.parallelRun());
  data_loader_->registerImageContainerCallback(
      std::bind(&dyno::DataInterfacePipeline::fillImageContainerQueue,
                data_interface_.get(), std::placeholders::_1));

  // if an external hook exists to update the time, add callback in
  // datainterface that will be triggered when new data is added to the output
  // queue (from the DataInterfacePipeline) this is basically used to alert ROS
  // to a new timestamp which we then publish to /clock
  if (external_hooks && external_hooks->update_time) {
    LOG(INFO) << "Added pre-queue callback to register new Timestampd data "
                 "with an external module";
    data_interface_->registerPreQueueContainerCallback(
        [external_hooks](const ImageContainer::Ptr image_container) -> void {
          external_hooks->update_time(image_container->timestamp());
        });
  }

  // ground truth
  data_loader_->registerGroundTruthPacketCallback(
      std::bind(&dyno::DataInterfacePipeline::addGroundTruthPacket,
                data_interface_.get(), std::placeholders::_1));

  // register single and multi IMU callbacks to the data loader
  data_loader_->registerImuSingleCallback(std::bind(
      static_cast<void (DataInterfacePipeline::*)(const ImuMeasurement&)>(
          &dyno::DataInterfacePipeline::fillImuQueue),
      data_interface_.get(), std::placeholders::_1));

  data_loader_->registerImuMultiCallback(std::bind(
      static_cast<void (DataInterfacePipeline::*)(const ImuMeasurements&)>(
          &dyno::DataInterfacePipeline::fillImuQueue),
      data_interface_.get(), std::placeholders::_1));

  // preprocessing
  data_interface_->registerImageContainerPreprocessor(
      std::bind(&dyno::DataProvider::imageContainerPreprocessor,
                data_loader_.get(), std::placeholders::_1));

  // push data from the data interface to the frontend module
  data_interface_->registerOutputQueue(&frontend_input_queue_);

  CameraParams camera_params;
  if (params_.preferDataProviderCameraParams() &&
      data_loader_->getCameraParams().has_value()) {
    LOG(INFO) << "Using camera params from DataProvider, not the config in the "
                 "CameraParams.yaml!";
    camera_params = *data_loader_->getCameraParams();
  } else {
    LOG(INFO) << "Using camera params specified in CameraParams.yaml!";
    camera_params = params_.camera_params_;
  }
  /// NOTE: no need to update the camera params like the imu params as we parse
  /// the camera params into the loadPipeline functions separately!

  ImuParams imu_params;
  if (params_.preferDataProviderImuParams() &&
      data_loader_->getImuParams().has_value()) {
    LOG(INFO) << "Using imu params from DataProvider, not the config in the "
                 "ImuParams.yaml!";
    imu_params = *data_loader_->getImuParams();
  } else {
    LOG(INFO) << "Using imu params specified in ImuParams.yaml!";
    imu_params = params_.imu_params_;
  }

  // update the imu params that will actually get sent to the frontend
  params_.frontend_params_.imu_params = imu_params;

  loadPipelines(camera_params, frontend_display, backend_display, factory);

  std::string cuda_enabled_message;
  utils::opencvCudaAvailable(&cuda_enabled_message);
  LOG(INFO) << cuda_enabled_message;

  launchSpinners();
}

DynoPipelineManager::~DynoPipelineManager() {
  shutdownPipelines();
  shutdownSpinners();

  // TODO: make shutdown hook!
  writeStatisticsSamplesToFile("statistics_samples.csv");
  writeStatisticsModuleSummariesToFile();
}

void DynoPipelineManager::shutdownSpinners() {
  if (frontend_pipeline_spinner_) frontend_pipeline_spinner_->shutdown();

  if (backend_pipeline_spinner_) backend_pipeline_spinner_->shutdown();

  if (data_provider_spinner_) data_provider_spinner_->shutdown();

  if (frontend_viz_pipeline_spinner_)
    frontend_viz_pipeline_spinner_->shutdown();

  if (backend_viz_pipeline_spinner_) backend_viz_pipeline_spinner_->shutdown();
}

void DynoPipelineManager::shutdownPipelines() {
  display_queue_.shutdown();
  frontend_pipeline_->shutdown();

  if (backend_pipeline_) backend_pipeline_->shutdown();

  data_interface_->shutdown();

  if (frontend_viz_pipeline_) frontend_viz_pipeline_->shutdown();
  if (backend_viz_pipeline_) backend_viz_pipeline_->shutdown();
}

bool DynoPipelineManager::spin() {
  utils::ChronoTimingStats timer("pipeline_spin");

  if (data_loader_->spin() || frontend_pipeline_->isWorking() ||
      (backend_pipeline_ && backend_pipeline_->isWorking())) {
    if (!params_.parallelRun()) {
      frontend_pipeline_->spinOnce();
      if (backend_pipeline_) backend_pipeline_->spinOnce();
    }
    // TODO: this definately takes some time...
    spinViz();  // for now
    // a later problem!
    return true;
  }
  return false;
}

bool DynoPipelineManager::spinViz() {
  if (FLAGS_use_opencv_display) {
    displayer_.process();
  }
  return true;
}

void DynoPipelineManager::launchSpinners() {
  LOG(INFO) << "Running PipelineManager with parallel_run="
            << params_.parallelRun();

  if (params_.parallelRun()) {
    frontend_pipeline_spinner_ = std::make_unique<dyno::Spinner>(
        std::bind(&dyno::PipelineBase::spin, frontend_pipeline_.get()),
        "frontend-pipeline-spinner");

    if (backend_pipeline_)
      backend_pipeline_spinner_ = std::make_unique<dyno::Spinner>(
          std::bind(&dyno::PipelineBase::spin, backend_pipeline_.get()),
          "backend-pipeline-spinner");
  }

  data_provider_spinner_ = std::make_unique<dyno::Spinner>(
      std::bind(&dyno::DataInterfacePipeline::spin, data_interface_.get()),
      "data-interface-spinner");

  if (frontend_viz_pipeline_)
    frontend_viz_pipeline_spinner_ = std::make_unique<dyno::Spinner>(
        std::bind(&dyno::FrontendVizPipeline::spin,
                  frontend_viz_pipeline_.get()),
        "frontend-display-spinner");

  if (backend_viz_pipeline_)
    backend_viz_pipeline_spinner_ = std::make_unique<dyno::Spinner>(
        std::bind(&dyno::BackendVizPipeline::spin, backend_viz_pipeline_.get()),
        "backend-display-spinner");
}

void DynoPipelineManager::loadPipelines(const CameraParams& camera_params,
                                        FrontendDisplay::Ptr frontend_display,
                                        BackendDisplay::Ptr backend_display,
                                        BackendModuleFactory::Ptr factory) {
  const auto backend_type = params_.backend_type;

  CameraParams mutable_camera_params = camera_params;
  if (!mutable_camera_params.hasDepthParams()) {
    LOG(INFO) << "Updating camera params: converting to Fake Stereo camera";
    // mutable_camera_params.setDepthParams(0.07);
    mutable_camera_params.setDepthParams(0.1);
  }
  Camera::Ptr camera = std::make_shared<Camera>(mutable_camera_params);
  CHECK_NOTNULL(camera);

  // Output regustra for the backend (ie. DynoState output) since this is the
  // same for all backend modules
  std::shared_ptr<BackendOutputRegistra> output_registra = nullptr;
  // load frontend
  if (backend_type == BackendType::KF_HYBRID) {
    // PoseChangeVIFrontend
    loadPoseChangeModules(camera, factory, frontend_, backend_,
                          external_backend_display_, output_registra);
  } else {
    loadRegularOrParallelHybridModules(camera, factory, frontend_, backend_,
                                       external_backend_display_,
                                       output_registra);
  }

  if (!frontend_) {
    throw DynosamException("Pipeline failure: no front-end could be loaded");
  }

  FrontendPipeline::UniquePtr frontend_pipeline_derived =
      std::make_unique<FrontendPipeline>("frontend-pipeline",
                                         &frontend_input_queue_, frontend_);
  const auto parallel_run = params_.parallelRun();
  frontend_pipeline_derived->parallelRun(parallel_run);
  frontend_pipeline_derived->registerOutputQueue(&frontend_viz_input_queue_);
  // conver pipeline to base type
  frontend_pipeline_ = std::move(frontend_pipeline_derived);

  if (frontend_display) {
    LOG(INFO) << "Loading frontend viz pipeline";
    frontend_viz_pipeline_ = std::make_unique<FrontendVizPipeline>(
        "frontend-viz-pipeline", &frontend_viz_input_queue_, frontend_display);
    frontend_viz_pipeline_->parallelRun(true);
  }

  if (backend_) {
    if (backend_display && output_registra) {
      LOG(INFO) << "Loaded backend with output queue registra. Constructing "
                   "backend viz";
      backend_viz_pipeline_ = std::make_unique<BackendVizPipeline>(
          "backend-viz-pipeline", &backend_viz_input_queue_, backend_display);
      backend_viz_pipeline_->parallelRun(true);
      // register viz queue as output of backend
      output_registra->registerQueue(&backend_viz_input_queue_);
    }

    if (external_backend_display_ && output_registra) {
      LOG(INFO) << "Connecting BackendModuleDisplay";
      auto external_backend_display = external_backend_display_;
      output_registra->registerCallback(
          [external_backend_display](const auto& output) -> void {
            external_backend_display->spinOnce(output);
          });
    }
  }
}

void DynoPipelineManager::loadPoseChangeModules(
    Camera::Ptr camera, BackendModuleFactory::Ptr factory,
    Frontend::Ptr& frontend_out, Backend::Ptr& backend_out,
    BackendModuleDisplay::Ptr& external_backend_display_out,
    std::shared_ptr<BackendOutputRegistra>& output_registra_out) {
  const auto backend_type = params_.backend_type;
  CHECK_EQ(backend_type, BackendType::KF_HYBRID);

  const auto parallel_run = params_.parallelRun();

  Sensors sensors;
  sensors.camera = camera;
  // TODO: display queue not used anymore!!!
  // TODO: ground truth!
  ModuleParams module_params;
  module_params.backend_params = params_.backend_params_;
  module_params.sensors = sensors;
  module_params.shared_ground_truth = data_interface_->getSharedGroundTruth();
  BackendWrapper backend_wrapper = factory->createModule(module_params);

  auto pose_change_backend =
      std::dynamic_pointer_cast<PoseChangeVIBackendModule>(
          backend_wrapper.backend);

  if (!pose_change_backend) {
    throw DynosamException(
        "Dyno pipeline loaded with BackendType::KF_HYBRID but loaded "
        "backend was not PoseChangeVIBackendModule!");
  }

  HybridFormulationKeyFrame::Ptr kf_formulation =
      pose_change_backend->getFormulation();
  CHECK_NOTNULL(kf_formulation);

  auto pc_vi_frontend = std::make_shared<PoseChangeVIFrontend>(
      params_, camera, kf_formulation, &display_queue_,
      data_interface_->getSharedGroundTruth());
  LOG(INFO) << "Made PoseChangeVIFrontend";

  frontend_out = pc_vi_frontend;

  using PoseChangePipeline =
      PipelineModuleProcessor<PoseChangeInput, DynoState>;
  using PoseChangeQueue = PoseChangePipeline::InputQueue;
  // construct backend pipeline and connect from frontend!!
  std::shared_ptr<PoseChangeQueue> backend_input_queue =
      std::make_shared<PoseChangeQueue>();
  std::unique_ptr<PoseChangePipeline> backend_pipeline =
      std::make_unique<PoseChangePipeline>("pose-change-vi-pipeline",
                                           backend_input_queue.get(),
                                           pose_change_backend);
  backend_pipeline->parallelRun(parallel_run);

  if (FLAGS_use_backend) {
    // register update function from backend to frontend
    pose_change_backend->registerUpdateCallback(
        std::bind(&PoseChangeVIFrontend::onBackendUpdateComplete,
                  pc_vi_frontend.get(), std::placeholders::_1));

    // register output function from frontend
    pc_vi_frontend->addPoseChangeOutputSink(
        [backend_input_queue](const PoseChangeInput::ConstPtr& pc_packet) {
          CHECK(backend_input_queue);
          backend_input_queue->push(pc_packet);
        });
  }

  output_registra_out = backend_pipeline->getOutputRegistra();

  // create storage object for queue since we currently use raw-pointers!
  // EEK!
  backend_input_queue_ = GenericThreadSafeQueueHolder(backend_input_queue);
  // must set backend pipeline in function here
  backend_pipeline_ = std::move(backend_pipeline);
  external_backend_display_out = backend_wrapper.backend_viz;
  backend_out = pose_change_backend;
}

// TODO: what we should get out are the two OutputReigstras for the frontend AND
// the backend
//  since we always know these types (i.e DynosamState/Realtime output)
//  then register these with VIZ if necessary?
void DynoPipelineManager::loadRegularOrParallelHybridModules(
    Camera::Ptr camera, BackendModuleFactory::Ptr factory,
    Frontend::Ptr& frontend_out, Backend::Ptr& backend_out,
    BackendModuleDisplay::Ptr& external_backend_display_out,
    std::shared_ptr<BackendOutputRegistra>& output_registra_out) {
  auto regular_vi_frontend = std::make_shared<RegularVIFrontend>(
      params_, camera, &display_queue_,
      data_interface_->getSharedGroundTruth());
  LOG(INFO) << "Made RegularVIFrontend";

  frontend_out = regular_vi_frontend;

  const auto parallel_run = params_.parallelRun();
  if (FLAGS_use_backend) {
    LOG(INFO) << "Construcing Backend";

    params_.backend_params_.full_batch_frame = data_loader_->datasetSize();
    Sensors sensors;
    sensors.camera = camera;

    // TODO: display queue not used anymore!!!
    // TODO: ground truth!
    ModuleParams module_params;
    module_params.backend_params = params_.backend_params_;
    module_params.sensors = sensors;
    module_params.shared_ground_truth = data_interface_->getSharedGroundTruth();

    // // this should be Backend::Ptr not backend module
    BackendWrapper backend_wrapper = factory->createModule(module_params);

    using VisionIMUBackendModule = BackendModule<VisionImuPacket>;
    auto vision_imu_backend_module =
        std::dynamic_pointer_cast<VisionIMUBackendModule>(
            backend_wrapper.backend);

    if (vision_imu_backend_module) {
      // TODO: better naming options for pipelines since they now all output
      // DynosamState!!!
      using VisionImuPipeline =
          PipelineModuleProcessor<VisionImuPacket, DynoState>;
      using VisionImuQueue = VisionImuPipeline::InputQueue;

      // construct backend pipeline and connect from frontend!!
      std::shared_ptr<VisionImuQueue> backend_input_queue =
          std::make_shared<VisionImuQueue>();
      std::unique_ptr<VisionImuPipeline> backend_pipeline =
          std::make_unique<VisionImuPipeline>("regular-vi-pipeline",
                                              backend_input_queue.get(),
                                              vision_imu_backend_module);
      backend_pipeline->parallelRun(parallel_run);

      // register output function from frontend
      regular_vi_frontend->addVIOutputSink(
          [backend_input_queue](const VisionImuPacket::ConstPtr& vi_packet) {
            CHECK(backend_input_queue);
            backend_input_queue->push(vi_packet);
          });

      output_registra_out = backend_pipeline->getOutputRegistra();

      // create storage object for queue since we currently use raw-pointers!
      // EEK!
      backend_input_queue_ = GenericThreadSafeQueueHolder(backend_input_queue);
      // must set backend pipeline in function here
      backend_pipeline_ = std::move(backend_pipeline);
      external_backend_display_out = backend_wrapper.backend_viz;
      backend_out = vision_imu_backend_module;
    } else {
      // TODO: make exception!!
      LOG(FATAL) << "IS BAD";
    }
  }
}

}  // namespace dyno
