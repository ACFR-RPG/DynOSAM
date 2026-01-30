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

#include "dynosam/frontend/FrontendModule.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/imu/ImuFrontend.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_cv/Camera.hpp"
#include "dynosam_opt/Map.hpp"

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
  // TODO: shared pointer for now during debig phase!
  ObjectMotionSolver::Ptr object_motion_solver_;
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

  // TODO: force camera kf for now!!! LOGIC is all over the place
  void fillOutputPacketWithTracks(VisionImuPacket::Ptr vision_imu_packet,
                                  const Frame& frame,
                                  const gtsam::Pose3& T_k_1_k,
                                  const ObjectMotionMap& object_motions,
                                  const ObjectPoseMap& object_poses,
                                  bool force_camera_kf = false) const;

  void sendToFrontendLogger(const Frame::Ptr& frame,
                            const VisionImuPacket::Ptr& vision_imu_packet);

  cv::Mat createTrackingImage(const Frame::Ptr& frame_k,
                              const Frame::Ptr& frame_k_1,
                              const ObjectPoseMap& object_poses) const;

  // used when we want to seralize the output to json via the
  // FLAGS_save_frontend_json flag
  //   std::map<FrameId, RGBDInstanceOutputPacket::Ptr> output_packet_record_;

  //! Imu frontend - mantains pre-integration from last kf to current k
  ImuFrontend imu_frontend_;
  //! Nav state at k
  gtsam::NavState nav_state_curr_;
  // this is always udpated with the best X_k pose but the velocity may be wrong
  // if no IMU...
  //! Nav state at k-1
  gtsam::NavState nav_state_prev_;

  //! Nav state of the last key-frame
  gtsam::NavState nav_state_lkf_;

  //! Tracks when the nav state was updated using IMU, else VO
  //! in the front-end, when updating with VO, the velocity will (currently) be
  //! wrong!!
  FrameId last_imu_k_{0};

  //! The relative camera pose (T_k_1_k) from the previous frame
  //! this is used as a constant velocity model when VO tracking fails and the
  //! IMU is not available!
  gtsam::Pose3 T_k_1_k_;

  gtsam::Pose3 T_lkf_k;

  //! Last (camera) keyframe
  Frame::Ptr frame_lkf_;

  // should not share nodes!!
  MapVision::Ptr full_local_map_;
  MapVision::Ptr kf_local_map_;
};

}  // namespace dyno
