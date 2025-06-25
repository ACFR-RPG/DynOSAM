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

#include "dynosam/common/Camera.hpp"
#include "dynosam/frontend/FrontendModule.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/vision/ObjectTracker.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"

namespace dyno {

class RGBDInstanceFrontendModule : public FrontendModule {
 public:
  RGBDInstanceFrontendModule(const FrontendParams& frontend_params,
                             Camera::Ptr camera,
                             ImageDisplayQueue* display_queue);
  ~RGBDInstanceFrontendModule();

  using SpinReturn = FrontendModule::SpinReturn;

 private:
  Camera::Ptr camera_;
  EgoMotionSolver motion_solver_;
  ObjectMotionSovlerF2F::Params object_motion_solver_params_;
  EgoMotionSolver::Params ego_motion_solver_params_;
  ObjectMotionSolver::UniquePtr object_motion_solver_;
  FeatureTracker::UniquePtr tracker_;
  RGBDFrontendLogger::UniquePtr logger_;
  //   Object

  //   ObjectMotionSolverSAM::UniquePtr experimental_solver_;

 private:
  ImageValidationResult validateImageContainer(
      const ImageContainer::Ptr& image_container) const override;
  SpinReturn boostrapSpin(FrontendInputPacketBase::ConstPtr input) override;
  SpinReturn nominalSpin(FrontendInputPacketBase::ConstPtr input) override;

  /**
   * @brief Solves PnP between frame_k-1 and frame_k using the tracked
   * correspondances to estimate the frame of the current camera
   *
   * the pose of the Frame::Ptr (frame_k) is updated, and the features marked as
   * outliers by PnP are set as outliers.
   *
   * Depending on FrontendParams::use_ego_motion_pnp, a differnet solver will be
   * used to estimate the pose
   *
   * @param frame_k
   * @param frame_k_1
   * @return true
   * @return false
   */
  bool solveCameraMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1);

  //   bool solveObjectMotion(Frame::Ptr frame_k, Frame::Ptr frame_k_1,
  //                          ObjectId object_id,
  //                          MotionEstimateMap& motion_estimates);

  //   void solveObjectMotions(Frame::Ptr frame_k, Frame::Ptr frame_k_1,
  //                           MotionEstimateMap& motion_estimates);

  RGBDInstanceOutputPacket::Ptr constructOutput(
      const Frame& frame, const ObjectMotionMap& object_motions,
      const ObjectPoseMap& object_poses, const gtsam::Pose3& T_world_camera,
      const GroundTruthInputPacket::Optional& gt_packet = std::nullopt,
      const DebugImagery::Optional& debug_imagery = std::nullopt,
      const PointCloudLabelRGB::Ptr dense_labelled_cloud = nullptr);

  cv::Mat createTrackingImage(const Frame::Ptr& frame_k,
                              const Frame::Ptr& frame_k_1,
                              const ObjectPoseMap& object_poses) const;

  // used when we want to seralize the output to json via the
  // FLAGS_save_frontend_json flag
  std::map<FrameId, RGBDInstanceOutputPacket::Ptr> output_packet_record_;
};

}  // namespace dyno
