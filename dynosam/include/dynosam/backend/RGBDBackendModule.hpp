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
#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/rgbd/WorldMotionEstimator.hpp"
#include "dynosam/backend/rgbd/WorldPoseEstimator.hpp"
#include "dynosam/common/Flags.hpp"
#include "dynosam/common/Map.hpp"

namespace dyno {

using RGBDBackendModuleTraits =
    BackendModuleTraits<RGBDInstanceOutputPacket, LandmarkKeypoint>;

class RGBDBackendModule : public BackendModuleType<RGBDBackendModuleTraits> {
 public:
  DYNO_POINTER_TYPEDEFS(RGBDBackendModule)

  using Base = BackendModuleType<RGBDBackendModuleTraits>;
  using RGBDMap = Base::MapType;

  enum UpdaterType {
    MotionInWorld = 0,
    LLWorld = 1,
    ObjectCentric = 2,
    OC_SD = 3,
    OC_D = 4,
    OC_S = 5
  };

  RGBDBackendModule(const BackendParams& backend_params, Camera::Ptr camera,
                    const UpdaterType& updater_type,
                    ImageDisplayQueue* display_queue = nullptr);
  ~RGBDBackendModule();

  using SpinReturn = Base::SpinReturn;

  // TODO: move to optimizer and put into pipeline manager where we know the
  // type and bind write output to shutdown procedure
  void saveGraph(const std::string& file = "rgbd_graph.dot");
  void saveTree(const std::string& file = "rgbd_bayes_tree.dot");

  std::tuple<gtsam::Values, gtsam::NonlinearFactorGraph> constructGraph(
      FrameId from_frame, FrameId to_frame, bool set_initial_camera_pose_prior,
      std::optional<gtsam::Values> initial_theta = {});

  // TODO: for now
 public:
  SpinReturn boostrapSpinImpl(
      RGBDInstanceOutputPacket::ConstPtr input) override;
  SpinReturn nominalSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) override;

  void updateMap(gtsam::Pose3& T_world_cam,
                 RGBDInstanceOutputPacket::ConstPtr input);

  // FOR NOW!!
  std::function<void(const Formulation<RGBDMap>::UniquePtr&, FrameId,
                     const GraphUpdateResult&)>
      callback;

 public:
  /**
   * @brief Helper struct to determine the conditions for the sliding window
   * optimisation. Only works when called in consequative frame order as we use
   * the last trigger frame (the last frame when the sliding window conditions
   * were true) to determine the next one
   */
  struct SlidingWindow {
    DYNO_POINTER_TYPEDEFS(SlidingWindow)

    /**
     * @brief Result of the condition check.
     * If the sliding window conditions have been met, and the window range
     * (start/ending frame) that should be covered. Calculated from the
     * sliding_window value.
     *
     */
    struct Result {
      bool condition;
      FrameId starting_frame;
      FrameId ending_frame;

      explicit operator bool() const { return condition; }
    };

    const int sliding_window;
    const int overlap_size;
    int previous_trigger_frame{
        -1};  //! The last frame where the sliding window conditions were true.
              //! Used to determine the next frame
    int first_frame{
        -1};  //! The first frame that is checked. Used to offset the condition
              //! checking if the first frame is not zero!

    // previous_trigger_frame starts at overlap
    SlidingWindow(const int window, const int overlap)
        : sliding_window(window),
          overlap_size(overlap),
          previous_trigger_frame(overlap) {}

    Result check(FrameId frame_k) {
      if (first_frame == -1) {
        first_frame = static_cast<int>(frame_k);
        CHECK_GE(first_frame, 0);
      }

      auto frame = static_cast<int>(frame_k) - first_frame;
      const bool condition =
          (previous_trigger_frame - (frame - sliding_window)) == overlap_size;
      if (condition) {
        previous_trigger_frame = frame;
      }

      Result result;
      result.condition = condition;
      result.ending_frame = frame_k;
      int starting_frame = frame_k - sliding_window;

      // some logic checks
      if (condition) {
        CHECK_GE(starting_frame, first_frame);
      }
      result.starting_frame = static_cast<FrameId>(starting_frame);
      return result;
    }
  };

  bool buildSlidingWindowOptimisation(FrameId frame_k,
                                      gtsam::Values& optimised_values,
                                      double& error_before,
                                      double& error_after);

  Formulation<RGBDMap>::UniquePtr makeUpdater();

  BackendMetaData createBackendMetadata() const;
  FormulationHooks createFormulationHooks() const;
  BackendOutputPacket::Ptr constructOutputPacket(FrameId frame_k,
                                                 Timestamp timestamp) const;
  static BackendOutputPacket::Ptr constructOutputPacket(
      const Formulation<RGBDMap>::UniquePtr& formulation, FrameId frame_k,
      Timestamp timestamp);

 public:
  Camera::Ptr camera_;
  const UpdaterType updater_type_;
  // Updater::UniquePtr new_updater_;
  Formulation<RGBDMap>::UniquePtr new_updater_;
  SlidingWindow::UniquePtr sliding_window_condition_;
  FrameId first_frame_id_;  // the first frame id that is received

  // new calibration every time
  inline auto getGtsamCalibration() const {
    const CameraParams& camera_params = camera_->getParams();
    return boost::make_shared<Camera::CalibrationType>(
        camera_params.constructGtsamCalibration<Camera::CalibrationType>());
  }

  // logger here!!
  BackendLogger::UniquePtr logger_{nullptr};
  DebugInfo debug_info_;
  std::unique_ptr<gtsam::ISAM2> smoother_;
  std::unique_ptr<gtsam::IncrementalFixedLagSmoother> fixed_lag_smoother_;
  std::unique_ptr<gtsam::IncrementalFixedLagSmoother>
      dynamic_fixed_lag_smoother_;
};

}  // namespace dyno
