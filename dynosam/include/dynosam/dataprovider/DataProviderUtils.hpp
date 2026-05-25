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

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

#include "dynosam_common/GroundTruthPacket.hpp"

namespace dyno {

/**
 * @brief Considers the conversion from the left-handed coordiante system (e.g.
 * unreal, VIODE, carla...) to the right-handed coordinate system (robotic).
 *
 * Taken from:
 * https://github.com/carla-simulator/ros-bridge/blob/master/carla_common/src/carla_common/transforms.py#L41
 *
 * @param right_handed_linear_velocity gtsam::Vector3&
 * @param right_handed_angular_velocity gtsam::Vector3&
 * @param left_handed_linear_velocity const gtsam::Vector3&
 * @param left_handed_angular_velocity const gtsam::Vector3&
 * @param left_handed_rotation std::optional<gtsam::Rot3>
 */
void toRightHandedTwist(gtsam::Vector3& right_handed_linear_velocity,
                        gtsam::Vector3& right_handed_angular_velocity,
                        const gtsam::Vector3& left_handed_linear_velocity,
                        const gtsam::Vector3& left_handed_angular_velocity,
                        std::optional<gtsam::Rot3> left_handed_rotation = {});

/**
 * @brief Considers the conversion from left-handed system (e.g. unreal, VIODE,
 * carla...) to right-handed system.
 *
 * @param left_handed_rotation const gtsam::Rot3&
 * @return gtsam::Rot3
 */
gtsam::Rot3 toRightHandedRotation(const gtsam::Rot3& left_handed_rotation);

/**
 * @brief Considers the conversion from a left-handed system (e.g. unreal,
 * VIODE, carla...) vector to right-handed system with optional rotation
 * provided in the left-handed system.
 *
 * @param left_handed_vector const gtsam::Vector3&
 * @param left_handed_rotation std::optional<gtsam::Rot3>
 * @return gtsam::Vector3
 */
gtsam::Vector3 toRightHandedVector(
    const gtsam::Vector3& left_handed_vector,
    std::optional<gtsam::Rot3> left_handed_rotation);

/**
 * @brief From an instance semantic mask (one that satisfies the requirements
 * for a SemanticMask), ie. all detected obejcts in the scene, with unique
 * instance labels as pixel values starting from 1 (background is 0). Using the
 * information in the ground truth, masks of detected objects that are not
 * moving are removed. Masks are removed by setting the pixle values to 0
 * (background).
 *
 * Expects the gt packet to be fully formed (ie. have all motion information set
 * from setMotions()).
 *
 *
 *
 *
 * NOTE: function expects to find a match between the object id's in the ground
 * truth packet, and the pixel values (instance mask values) in the image!! If
 * there is not a 1-to-1 match between ground truth packets and the pixels, the
 * function will fail
 *
 *
 * @param instance_mask const cv::Mat&
 * @param motion_mask cv::Mat&
 * @param gt_packet const GroundTruthInputPacket&
 */
void removeStaticObjectFromMask(const cv::Mat& instance_mask,
                                cv::Mat& motion_mask,
                                const GroundTruthInputPacket& gt_packet);

}  // namespace dyno
