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

#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"


#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac/Ransac.hpp>

#include <optional>

using AbsolutePoseProblem = opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
using AbsolutePoseSacProblem = opengv::sac::Ransac<AbsolutePoseProblem>;

namespace dyno {

class MotionResult : public std::optional<gtsam::Pose3> {
public:
    enum Status { VALID, NOT_ENOUGH_CORRESPONDENCES, NOT_ENOUGH_INLIERS, UNSOLVABLE };
    Status status;

private:
    MotionResult(Status s) : status(s) {};

public:
    MotionResult() {}

    MotionResult(const gtsam::Pose3& pose) : status(VALID) { emplace(pose); }

    operator const gtsam::Pose3&() const { return get(); }

    static MotionResult NotEnoughCorrespondences() { return MotionResult(NOT_ENOUGH_CORRESPONDENCES); }
    static MotionResult NotEnoughInliers() { return MotionResult(NOT_ENOUGH_INLIERS); }
    static MotionResult Unsolvable() { return MotionResult(UNSOLVABLE); }

    inline bool valid() const { return status == VALID; }
    inline bool notEnoughCorrespondences() const { return status == NOT_ENOUGH_CORRESPONDENCES; }
    inline bool notEnoughInliers() const { return status == NOT_ENOUGH_INLIERS; }
    inline bool unsolvable() const { return status == UNSOLVABLE; }

    const gtsam::Pose3& get() const {
        if (!has_value()) throw std::runtime_error("MotionResult has no value");
        return value();
    }

};


class MotionSolver {

public:


    MotionSolver(const FrontendParams& params, const CameraParams& camera_params);

    //current_keypoints->2d observations in current frame, previous_points->3d landmarks in world frame
    MotionResult solveCameraPose(const AbsolutePoseCorrespondences& correspondences, TrackletIds& inliers, TrackletIds& outliers);
    MotionResult solveObjectMotion(const AbsolutePoseCorrespondences& correspondences, const gtsam::Pose3& curr_T_world_camera_, TrackletIds& inliers, TrackletIds& outliers);

protected:
    MotionResult solve3D2DRansac(const AbsolutePoseCorrespondences& correspondences, TrackletIds& inliers, TrackletIds& outliers);

protected:
    const FrontendParams params_;
    const CameraParams camera_params_;

};

} //dyno
