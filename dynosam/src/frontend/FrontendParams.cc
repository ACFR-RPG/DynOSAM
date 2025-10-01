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

#include "dynosam/frontend/FrontendParams.hpp"

#include <config_utilities/config_utilities.h>

#include <string>

#include "dynosam_common/Flags.hpp"

DEFINE_bool(refine_motion_estimate, true,
            "If true, 3D motion refinement will be used");
// TODO: clear up flags - this is defined in Flags.h
DEFINE_bool(refine_with_optical_flow, true,
            "If true, then joint refinement with optical flow will be used");

namespace dyno {

void declare_config(FrontendParams& config) {
  using namespace config;

  name("FrontendParams");
  field(config.scene_flow_magnitude, "scene_flow_magnitude");
  field(config.scene_flow_percentage, "scene_flow_percentage");

  field(config.max_background_depth, "max_background_depth");
  field(config.max_object_depth, "max_object_depth");

  field(config.use_ego_motion_pnp, "use_ego_motion_pnp");
  field(config.use_object_motion_pnp, "use_object_motion_pnp");
  field(config.refine_camera_pose_with_joint_of,
        "refine_camera_pose_with_joint_of");

  field(config.object_motion_solver_params, "object_motion_solver");
  field(config.ego_motion_solver_params, "camera_motion_solver");
  field(config.tracker_params, "tracker_params");

  field(config.image_tracks_vis_params, "image_tracks_vis_params");

  // update with flags
  config.object_motion_solver_params.refine_motion_with_joint_of =
      FLAGS_refine_with_optical_flow;
  config.object_motion_solver_params.refine_motion_with_3d =
      FLAGS_refine_motion_estimate;

  LOG(INFO) << "config.object_motion_solver_params.refine_motion_with_joint_of "
            << config.object_motion_solver_params.refine_motion_with_joint_of;
}

}  // namespace dyno
