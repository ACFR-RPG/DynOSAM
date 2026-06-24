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

namespace dyno {

void declare_config(CameraPoseSolver& config) {
  using namespace config;

  name("CameraPoseSolver");
  field(config.pnp_ransac_params, "pnp_ransac");
  field(config.optical_flow_solver_params, "optical_flow_solver");
  field(config.refine_with_flow, "refine_with_flow");
}

void declare_config(FrontendParams& config) {
  using namespace config;

  name("FrontendParams");
  field(config.scene_flow_magnitude, "scene_flow_magnitude");
  field(config.scene_flow_percentage, "scene_flow_percentage");

  field(config.max_background_depth, "max_background_depth");
  field(config.max_object_depth, "max_object_depth");

  field(config.regular_object_motion_solver_params,
        "regular_object_motion_solver");
  field(config.hybrid_object_motion_solver_params,
        "hybrid_object_motion_solver");
  field(config.camera_pose_solver_params, "camera_pose_solver");
  field(config.tracker_params, "tracker_params");

  field(config.image_tracks_vis_params, "image_tracks_vis_params");
}

}  // namespace dyno
