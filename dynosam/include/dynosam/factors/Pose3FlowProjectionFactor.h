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

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point2.h>

namespace dyno {

//expects the camera to have a calibration()
template<class CAMERA>
class Pose3FlowProjectionFactor : public gtsam::NoiseModelFactor2<gtsam::Point2, gtsam::Pose3> {

public:
    using Camera = CAMERA;
    using This = Pose3FlowProjectionFactor<Camera>;
    using Base = gtsam::NoiseModelFactor2<gtsam::Point2, gtsam::Pose3>; //keypoint to camera pose

    using shared_ptr = std::shared_ptr<This>;


    Pose3FlowProjectionFactor(
        gtsam::Key optical_flow_key,
        gtsam::Key camera_pose_key,
        const gtsam::Point2& keypoint,
        double depth,
        const Camera& camera_ref,
        gtsam::SharedNoiseModel model)
    : Base(model, optical_flow_key, camera_pose_key), keypoint_(keypoint), depth_(depth), camera_ref_(camera_ref) {}


    gtsam::Vector evaluateError(const gtsam::Point2& optical_flow, const gtsam::Point3& camera_pose,
                              boost::optional<gtsam::Matrix&> J1 = boost::none,
                              boost::optional<gtsam::Matrix&> J2 = boost::none) const override
    {
        //project the ref keypoint into the world frame given the camera
        gtsam::Point3 P_W = camera_ref_.backproject(keypoint_, depth_);

        //assume current and ref camera have the same calibration
        const auto calibration = camera_ref_.calibration();
        Camera camera_curr(camera_pose, calibration);

        gtsam::Point2 predicted_keypoint = keypoint_ + optical_flow;
        //project P_w into the current frame frame which should be the predicted one
        gtsam::Point2 predicted_keypoint_from_projection = camera_curr.project2(P_W);

        //point in the current camera frame
        gtsam::Point3 P_curr = camera_curr.pose().inverse() * P_W;

        if(J1) {
            *J1 = gtsam::Matrix22::Identity();
        }

        if(J2) {
            const double x = P_curr(0);
            const double y = P_curr(1);
            const double z = P_curr(2);
            const double z_2 = z*z;

            const double fx = calibration.fx();
            const double fy = calibration.fy();

            gtsam::Matrix26 H;
            H(0,0) =  x*y/z_2 *fx;
            H(0,1) = -(1+(x*x/z_2)) *fx;
            H(0,2) = y/z *fx;
            H(0,3) = -1./z *fx;
            H(0,4) = 0;
            H(0,5) = x/z_2 *fx;

            H(1,0) = (1+y*y/z_2) *fy;
            H(1,1) = -x*y/z_2 *fy;
            H(1,2) = -x/z *fy;
            H(1,3) = 0;
            H(1,4) = -1./z *fy;
            H(1,5) = y/z_2 *fy;

            *J2 = H;
        }

        return predicted_keypoint - predicted_keypoint_from_projection;
    }


private:
    gtsam::Point2 keypoint_; //! Observed keypoint, the origin of the optical flow (in the ref frame)
    double depth_; //! Observed depth measurement of the keypoint (in the camera frame, as from an RGBD camera)
    Camera camera_ref_; //! ref Camera in which this feature was seen in
};

} //dyno
