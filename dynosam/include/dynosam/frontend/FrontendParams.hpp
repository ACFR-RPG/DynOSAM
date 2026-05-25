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

#include <cmath>
#include <string>

#include "dynosam/frontend/solvers/ConsecutiveFrameObjectMotionSolver.hpp"
#include "dynosam/frontend/solvers/PnPRansac.hpp"
#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"  //for ImageTracksParams
#include "dynosam/frontend/vision/TrackerParams.hpp"
#include "dynosam_sensors/ImuParams.hpp"

namespace dyno {

struct FrontendParams {
  // scene flow thresholds
  double scene_flow_magnitude = 0.12;
  double scene_flow_percentage = 0.5;

  // depth thresholds
  double max_background_depth = 40.0;
  double max_object_depth = 25.0;

  // TODO: add depth cov
  // TODO: add projection cov (should this be for back and frontend?)

  //! When using RGBD pipeline, ego-motion will be sovled using pnp (3d2d
  //! correspondences). Else, stereo
  bool use_ego_motion_pnp = true;

  //! When using RGBD pipeline, object motion will be sovled using pnp (3d2d
  //! correspondences). Else, stereo
  bool use_object_motion_pnp = true;

  // Refine the camera pose with oint optical flow optimisation
  bool refine_camera_pose_with_joint_of = true;

  // TODO: load camera ransac and joint of
  // TODO: load object ransac and joint of and 3d motion
  // TODO: load special motion solver params separately? MAYBE?
  ConsecutiveFrameObjectMotionSolverParams cf_object_motion_solver_params;
  PnPRansacSolverParams ego_motion_pnp_ransac_params;

  TrackerParams tracker_params = TrackerParams();
  ImageTracksParams image_tracks_vis_params = ImageTracksParams();
  ImuCalibration imu_calib = ImuCalibration();
};

void declare_config(FrontendParams& config);

}  // namespace dyno
