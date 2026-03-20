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

#pragma once

#include "dynosam/backend/BackendModuleFactory.hpp"
// TODO : try and delete
#include "dynosam/dataprovider/DataInterfacePipeline.hpp"
#include "dynosam/dataprovider/DataProvider.hpp"
#include "dynosam/frontend/VIFrontend.hpp"
#include "dynosam/pipeline/PipelineHooks.hpp"
#include "dynosam/pipeline/PipelineModuleProcessor.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam/visualizer/VisualizerPipelines.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Spinner.hpp"

namespace dyno {

using FrontendPipeline =
    PipelineModuleProcessor<VIFrontendInput, RealtimeOutput>;

using BackendOutputRegistra = QueueRegistra<DynoState>;

namespace internal {}

class DynoPipelineManager {
 public:
  DYNO_POINTER_TYPEDEFS(DynoPipelineManager)

  // why are some unique and some shared?? silly silly
  DynoPipelineManager(const DynoParams& params, DataProvider::Ptr data_loader,
                      FrontendDisplay::Ptr frontend_display,
                      BackendDisplay::Ptr backend_display,
                      BackendModuleFactory::Ptr factory,
                      const ExternalHooks::Ptr external_hooks = nullptr);
  ~DynoPipelineManager();

  /**
   * @brief Spins the whole pipeline by spinning the data provider.
   * If in sequential mode, it will return for each spin
   * If in parallel mode, it will not return until the pipeline is shutdown.
   *
   * The return value indicates the data provider modules state: false if
   * finished or shutdown, true its working nominally
   *
   * @return true
   * @return false
   */
  virtual bool spin();

  virtual bool spinViz();

 private:
  void launchSpinners();
  void shutdownSpinners();
  void shutdownPipelines();

  void loadPipelines(const CameraParams& camera_params,
                     FrontendDisplay::Ptr frontend_display,
                     BackendDisplay::Ptr backend_display,
                     BackendModuleFactory::Ptr factory);

  void loadPoseChangeModules(
      Camera::Ptr camera, BackendModuleFactory::Ptr factory,
      Frontend::Ptr& frontend_out, Backend::Ptr& backend_out,
      BackendModuleDisplay::Ptr& external_backend_display_out,
      std::shared_ptr<BackendOutputRegistra>& output_registra_out);

  void loadRegularOrParallelHybridModules(
      Camera::Ptr camera, BackendModuleFactory::Ptr factory,
      Frontend::Ptr& frontend_out, Backend::Ptr& backend_out,
      BackendModuleDisplay::Ptr& external_backend_display_out,
      std::shared_ptr<BackendOutputRegistra>& output_registra_out);

 private:
  DynoParams params_;
  Frontend::Ptr frontend_;
  Backend::Ptr backend_;

  //! Use the pipeline base version of the frontend pipeline
  //! This allows us to have different pipelines to provide data to the backend
  //! (e.g. load seralized data) which have different templates
  PipelineBase::UniquePtr frontend_pipeline_{nullptr};
  FrontendPipeline::InputQueue frontend_input_queue_;
  FrontendPipeline::OutputQueue frontend_viz_input_queue_;

  // BackendPipeline::UniquePtr backend_pipeline_{nullptr};
  PipelineBase::UniquePtr backend_pipeline_{nullptr};
  BackendOutputRegistra::Queue backend_viz_input_queue_;
  //! Generic holder for the backend input queue to ensure its held in memory
  //! As we dont know the type of the backend input at compile time
  GenericThreadSafeQueueHolder backend_input_queue_;

  // Data-provider pointers
  DataInterfacePipeline::UniquePtr data_interface_;
  DataProvider::Ptr data_loader_;

  // Display and Viz
  BackendModuleDisplay::Ptr external_backend_display_;
  FrontendVizPipeline::UniquePtr frontend_viz_pipeline_;
  BackendVizPipeline::UniquePtr backend_viz_pipeline_;
  ImageDisplayQueue display_queue_;
  OpenCVImageDisplayQueue displayer_;

  // Threaded spinners
  Spinner::UniquePtr data_provider_spinner_;
  Spinner::UniquePtr frontend_pipeline_spinner_;
  Spinner::UniquePtr backend_pipeline_spinner_;
  Spinner::UniquePtr frontend_viz_pipeline_spinner_;
  Spinner::UniquePtr backend_viz_pipeline_spinner_;
};

}  // namespace dyno
