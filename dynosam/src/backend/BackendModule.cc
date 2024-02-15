/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "dynosam/backend/BackendModule.hpp"

#include <glog/logging.h>

namespace dyno {

BackendModule::BackendModule(const BackendParams& params, Camera::Ptr camera, ImageDisplayQueue* display_queue, bool use_logger)
    :   Base("backend"),
        base_params_(params),
        camera_(CHECK_NOTNULL(camera)),
        display_queue_(display_queue)

{
    const auto& camera_params = camera_->getParams();
    gtsam_calibration_ = boost::make_shared<Camera::CalibrationType>(
        camera_params.constructGtsamCalibration<Camera::CalibrationType>());

    CHECK(gtsam_calibration_);
    setFactorParams(params);

    //create callback to update gt_packet_map_ values so the derived classes dont need to manage this
    //TODO: this logic is exactly the same as in FrontendModule - functionalise!!
    registerInputCallback([=](BackendInputPacket::ConstPtr input) {
        if(input->gt_packet_) gt_packet_map_.insert2(input->getFrameId(), *input->gt_packet_);
    });

    if(use_logger) {
        VLOG(5) << "Using backend logger";
        logger_ = std::make_unique<BackendLogger>();
    }
}

void BackendModule::setFactorParams(const BackendParams& backend_params) {
    //set static noise
    static_pixel_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, backend_params.static_smart_projection_noise_sigma_);
    CHECK(static_pixel_noise_);

    //set dynamic noise
    dynamic_pixel_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, backend_params.dynamic_smart_projection_noise_sigma_);
    CHECK(dynamic_pixel_noise_);

    gtsam::Vector6 odom_sigmas;
    odom_sigmas.head<3>().setConstant(backend_params.odometry_rotation_sigma_);
    odom_sigmas.tail<3>().setConstant(
        backend_params.odometry_translation_sigma_);
    odometry_noise_ = gtsam::noiseModel::Diagonal::Sigmas(odom_sigmas);
    CHECK(odometry_noise_);

    initial_pose_prior_ =  gtsam::noiseModel::Isotropic::Sigma(6u, 0.0001);
    CHECK(initial_pose_prior_);

    landmark_motion_noise_ = gtsam::noiseModel::Isotropic::Sigma(3u, backend_params.motion_ternary_factor_noise_sigma_);
    CHECK(landmark_motion_noise_);

    gtsam::Vector6 object_constant_vel_sigmas;
    object_constant_vel_sigmas.head<3>().setConstant(backend_params.constant_object_motion_rotation_sigma_);
    object_constant_vel_sigmas.tail<3>().setConstant(
        backend_params.constant_object_motion_translation_sigma_);
    object_smoothing_noise_ = gtsam::noiseModel::Diagonal::Sigmas(object_constant_vel_sigmas);
    CHECK(object_smoothing_noise_);

}


void BackendModule::validateInput(const BackendInputPacket::ConstPtr&) const {

}

} //dyno
