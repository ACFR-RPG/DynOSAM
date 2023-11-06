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

namespace dyno {

class RGBDInstanceFrontendModule : public FrontendModule {

public:
    RGBDInstanceFrontendModule(const FrontendParams& frontend_params, Camera::Ptr camera, ImageDisplayQueue* display_queue);

    using SpinReturn = FrontendModule::SpinReturn;

private:
    Camera::Ptr camera_;
    MotionSolver motion_solver_;
    FeatureTracker::UniquePtr tracker_;

    Frame::Ptr previous_frame_;

    std::map<ObjectId, gtsam::Pose3> object_poses_; //just for now (viz)


    bool validateImageContainer(const ImageContainer::Ptr& image_container) const override;
    SpinReturn boostrapSpin(FrontendInputPacketBase::ConstPtr input) override;
    SpinReturn nominalSpin(FrontendInputPacketBase::ConstPtr input) override;

};


} //dyno
