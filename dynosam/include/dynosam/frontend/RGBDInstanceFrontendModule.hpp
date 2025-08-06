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

#include <gtsam/navigation/NavState.h>

#include "dynosam/common/Camera.hpp"
#include "dynosam/frontend/FrontendModule.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/imu/ImuFrontend.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/vision/ObjectTracker.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"

namespace dyno {

class RGBDInstanceFrontendModule : public FrontendModule {
 public:
  RGBDInstanceFrontendModule(const DynoParams& params, Camera::Ptr camera,
                             ImageDisplayQueue* display_queue);
  ~RGBDInstanceFrontendModule();

  using SpinReturn = FrontendModule::SpinReturn;

 private:
  Camera::Ptr camera_;
  EgoMotionSolver motion_solver_;
  //   ObjectMotionSovlerF2F object_motion_solver_;
  ObjectMotionSolver::UniquePtr object_motion_solver_;
  FeatureTracker::UniquePtr tracker_;
  RGBDFrontendLogger::UniquePtr logger_;

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
  bool solveCameraMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1,
                         std::optional<gtsam::Rot3> R_curr_ref = {});

  VisionImuPacket::Ptr constructOutputAndLog(
      const FrontendInputPacketBase::ConstPtr& input, const Frame& frame,
      const ObjectMotionMap& object_motions, const ObjectPoseMap& object_poses,
      const gtsam::Pose3& T_k_1_k, ImuFrontend::PimPtr pim);

  //   VisionImuPacket::Ptr constructOutput(
  //       const Frame& frame, const ObjectMotionMap& object_motions,
  //       const ObjectPoseMap& object_poses, const gtsam::Pose3&
  //       T_world_camera, const GroundTruthInputPacket::Optional& gt_packet =
  //       std::nullopt, const DebugImagery::Optional& debug_imagery =
  //       std::nullopt, const PointCloudLabelRGB::Ptr dense_labelled_cloud =
  //       nullptr);

  cv::Mat createTrackingImage(const Frame::Ptr& frame_k,
                              const Frame::Ptr& frame_k_1,
                              const ObjectPoseMap& object_poses) const;

  // used when we want to seralize the output to json via the
  // FLAGS_save_frontend_json flag
  //   std::map<FrameId, RGBDInstanceOutputPacket::Ptr> output_packet_record_;

  ImuFrontend imu_frontend_;
  gtsam::NavState nav_state_;
  // this is always udpated with the best X_k pose but the velocity may be wrong
  // if no IMU...
  gtsam::NavState previous_nav_state_;

  //! Tracks when the nav state was updated using IMU, else VO
  //! in the front-end, when updating with VO, the velocity will (currently) be
  //! wrong!!
  FrameId last_imu_nav_state_update_{0};

  //! The relative camera pose (T_k_1_k) from the previous frame
  //! this is used as a constant velocity model when VO tracking fails and the
  //! IMU is not available!
  gtsam::Pose3 vo_velocity_;
};

}  // namespace dyno
