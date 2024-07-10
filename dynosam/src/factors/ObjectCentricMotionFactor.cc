/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#include "dynosam/factors/ObjectCentricMotionFactor.hpp"

namespace dyno {

ObjectCentricMotionFactor::ObjectCentricMotionFactor(
    gtsam::Key camera_pose, gtsam::Key motion, gtsam::Key point_object,
    const gtsam::Point3& measurement, const gtsam::Pose3& L_0, gtsam::SharedNoiseModel model)
:   Base(model, camera_pose, motion, point_object), measurement_(measurement), L_0_(L_0) {}


gtsam::Vector ObjectCentricMotionFactor::evaluateError(
    const gtsam::Pose3& camera_pose,
    const gtsam::Pose3& motion,
    const gtsam::Point3& point_object,
    boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const
{

     if(J1) {
        // error w.r.t to camera pose
        Eigen::Matrix<double, 3, 6> df_dX =
            gtsam::numericalDerivative31<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(
                std::bind(&ObjectCentricMotionFactor::residual, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3, measurement_, L_0_),
            camera_pose, motion, point_object);
        *J1 = df_dX;
    }

    if(J2) {
        // error w.r.t to motion
        Eigen::Matrix<double, 3, 6> df_dH =
            gtsam::numericalDerivative32<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(
                std::bind(&ObjectCentricMotionFactor::residual, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3, measurement_, L_0_),
            camera_pose, motion, point_object);
        *J2 = df_dH;
    }

    if(J3) {
        // error w.r.t to point in local
        Eigen::Matrix<double, 3, 3> df_dm =
            gtsam::numericalDerivative33<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(
                std::bind(&ObjectCentricMotionFactor::residual, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3, measurement_, L_0_),
            camera_pose, motion, point_object);
        *J3 = df_dm;
    }

    return residual(
        camera_pose, motion, point_object, measurement_, L_0_
    );

}

gtsam::Vector ObjectCentricMotionFactor::residual(
        const gtsam::Pose3& camera_pose, const gtsam::Pose3& motion,
        const gtsam::Point3& point_object, const gtsam::Point3& measurement, const gtsam::Pose3& L_0)
{
    //apply transform to put map point into world via its motion
    gtsam::Point3 map_point_world = motion * L_0 * point_object;
    //put map_point_world into local camera coordinate
    gtsam::Point3 map_point_camera = camera_pose.inverse() * map_point_world;
    return map_point_camera - measurement;
}

}
