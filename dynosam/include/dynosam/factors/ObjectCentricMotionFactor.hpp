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

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/numericalDerivative.h>


namespace dyno {


class ObjectCentricMotionFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3> {


public:
    typedef boost::shared_ptr<ObjectCentricMotionFactor> shared_ptr;
    typedef ObjectCentricMotionFactor This;
    typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3> Base;

    gtsam::Point3 measurement_;
    gtsam::Pose3 L_0_;

    ObjectCentricMotionFactor(
        gtsam::Key camera_pose, gtsam::Key motion, gtsam::Key point_object,
        const gtsam::Point3& measurement, const gtsam::Pose3& L_0, gtsam::SharedNoiseModel model);

    gtsam::Vector evaluateError(const gtsam::Pose3& camera_pose, const gtsam::Pose3& motion,
        const gtsam::Point3& point_object,
        boost::optional<gtsam::Matrix&> J1 = boost::none,
        boost::optional<gtsam::Matrix&> J2 = boost::none,
        boost::optional<gtsam::Matrix&> J3 = boost::none) const override;

    static gtsam::Vector residual(const gtsam::Pose3& camera_pose, const gtsam::Pose3& motion,
        const gtsam::Point3& point_object, const gtsam::Point3& measurement, const gtsam::Pose3& L_0);

};

}
