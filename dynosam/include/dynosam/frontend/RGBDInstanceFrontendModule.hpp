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

#pragma once

#include "dynosam/common/Camera.hpp"
#include "dynosam/frontend/FrontendModule.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"

namespace dyno {

class RGBDInstanceFrontendModule : public FrontendModule {

public:
    RGBDInstanceFrontendModule(const FrontendParams& frontend_params, Camera::Ptr camera, ImageDisplayQueue* display_queue);

    using SpinReturn = FrontendModule::SpinReturn;

private:
    Camera::Ptr camera_;
    FeatureTracker::UniquePtr tracker_;
    FrontendLogger::UniquePtr logger_;

private:

    ImageValidationResult validateImageContainer(const ImageContainer::Ptr& image_container) const override;
    SpinReturn boostrapSpin(FrontendInputPacketBase::ConstPtr input) override;
    SpinReturn nominalSpin(FrontendInputPacketBase::ConstPtr input) override;

    /**
     * @brief Solves PnP between frame_k-1 and frame_k using the tracked correspondances
     * to estimate the frame of the current camera
     *
     * the pose of the Frame::Ptr (frame_k) is updated, and the features marked as outliers
     * by PnP are set as outliers.
     *
     * @param frame_k
     * @param frame_k_1
     * @return true
     * @return false
     */
    bool solveCameraMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1);


    bool solveObject3d2dMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1, ObjectId object_id, MotionEstimateMap& motion_estimates);

    RGBDInstanceOutputPacket::Ptr constructOutput(
        const Frame& frame,
        const MotionEstimateMap& estimated_motions,
        const gtsam::Pose3& T_world_camera,
        const cv::Mat& debug_image = cv::Mat(),
        const GroundTruthInputPacket::Optional& gt_packet = std::nullopt);

    //updates the object_poses_ map which is then sent to the viz via the output
    //this updates the estimated trajectory of the object from a starting pont using the
    //object motion to propofate the object pose
    void propogateObjectPoses(const MotionEstimateMap& motion_estimates, FrameId frame_id);
    //updates object poses
    void propogateObjectPose(const gtsam::Pose3& prev_H_world_curr, ObjectId object_id, FrameId frame_id);




};


} //dyno
