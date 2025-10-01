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

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>

#include "dynosam_common/Types.hpp"
#include "dynosam_vision_common/CameraParams.hpp"

namespace dyno {

class RGBDCamera;

class Camera {
 public:
  DYNO_POINTER_TYPEDEFS(Camera)
  DYNO_DELETE_COPY_CONSTRUCTORS(Camera)

  Camera(const CameraParams& camera_params);
  virtual ~Camera() = default;

  // Camera implementation provided by gtsam
  // original one is Cal3_S2 -> not sure what diff is?
  // using CalibrationType = gtsam::Cal3_S2;
  using CalibrationType = gtsam::Cal3_S2;
  using CameraImpl = gtsam::PinholeCamera<CalibrationType>;

  /**
   * @brief Projects a 3D point in the camera frame into the image frame as a
   * keypoint. Doesnt do any checks if the keypoint lands in the frame; use in
   * conjunction with isLandmarkContained
   *
   * @param lmks A 3D landmark
   * @param kpts The 2D keypoint to be set
   */
  void project(const Landmark& lmk, Keypoint* kpt) const;

  /**
   * @brief Projects a list 3D points in the camera frame into the image frame.
   * Doesnt do any checks if the keypoint lands in the frame; use in conjunction
   * with isLandmarkContained
   *
   * @param lmks A list of 3D points
   * @param kpts A list of 2D keypoints to be set
   */
  void project(const Landmarks& lmks, Keypoints* kpts) const;

  /**
   * @brief Checks if a given keypoint is inside the image AND in front the
   * camera (ie. depth > 0).
   *
   * Assume that the camera is orientated with z-axis pointing along the line of
   * sight of the camera, otherwise the depth > 0 check is invalid.
   *
   * @param kpts Projected Keypoint
   * @param depth Depth of the 3D point (along the z axis)
   * @return true If the keypoint is visible from the camera frustrum
   * @return false
   */
  bool isKeypointContained(const Keypoint& kpt, Depth depth) const;
  bool isKeypointContained(const Keypoint& kpt) const;

  /**
   * @brief Back projects a list of keypoints from the image frame and into the
   * camera frame given a depth.
   *
   * Assume that the number of keypoints and number of depth values are the same
   * and are aligned by index.
   *
   * @param kps List of keypoints to back project
   * @param depths List of depth values to project along
   * @param lmks The 3D landmarks to set.
   */
  void backProject(const Keypoints& kps, const Depths& depths,
                   Landmarks* lmks) const;

  /**
   * @brief Back projects a single keypoint from the image frame and into the
   * camera frame given a depth.
   *
   * @param kp const Keypoint& Keypoint to back project
   * @param depth const Depth& Depth value to project along
   * @param lmk Landmark* 3D landmark to set.
   */
  void backProject(const Keypoint& kp, const Depth& depth, Landmark* lmk) const;
  void backProject(const Keypoint& kp, const Depth& depth, Landmark* lmk,
                   const gtsam::Pose3& X_world) const;

  /**
   * @brief Projects a point using a keypoint measurement and a Z measurement,
   * rather than depth
   *
   * Depth is the depth along a projected ray while Z is the distance the point
   * is from the camera frame along the Z axis
   *
   *                  * (P)
   *          |      /
   *          |     /
   *          |    /
   *        Z |   / d (depth)
   *          |  /
   *          | /
   *          |/
   *      --------- (Camera Plane)
   *
   * @param kp const Keypoint&
   * @param Z const double Z distance of the point along the Z axis
   * @param lmk Landmark* 3D landmark to set.
   * @param Dp gtsam::OptionalJacobian<3, 3> Jacobian w.r.t to the calculated
   * lmk. Row structure is [x,y,z] of the output lmk and Col structure is [u, v,
   * Z] (i.e [kp(0), kp(1), Z]) of the input
   */
  void backProjectFromZ(const Keypoint& kp, const double Z, Landmark* lmk,
                        gtsam::OptionalJacobian<3, 3> Dp = {}) const;

  /**
   * @brief Projects a point using a keypoint measurement and a Z measurement,
   * rather than depth
   *
   * @param kp const Keypoint&
   * @param Z double Z coordinate of the 3D point in the camera frame
   * @param lmk Landmark* 3D landmark to set in the coordinate frame determined
   * by X_world
   * @param X_world const gtsam::Pose3& pose to transform the 3D point to. If I,
   * lmk will be in the camera frame
   * @param Dp gtsam::OptionalJacobian<3, 3> Jacobian w.r.t to the calculated
   * lmk. Row structure is [x,y,z] of the output lmk and Col structure is [u, v,
   * Z] (i.e [kp(0), kp(1), Z]) of the input
   */
  void backProjectFromZ(const Keypoint& kp, const double Z, Landmark* lmk,
                        const gtsam::Pose3& X_world,
                        gtsam::OptionalJacobian<3, 3> Dp = {}) const;

  static Landmark cameraToWorldConvention(const Landmark& lmk);

  /**
   * @brief Checks if a landmark can be seen in the image frustrum. The 3D point
   * must be in the camera frame.
   *
   * If keypoint is not null, it will be set with the projected u,v coordinates.
   * The keypoint will be set regardless of whether the landmark is visible.
   *
   * @param lmk 3D landmark in the camera frame to check.
   * @param keypoint The projected of the landmark in the image frame.
   * @return true If the landmark is visible from the camera frame.
   * @return false
   */
  bool isLandmarkContained(const Landmark& lmk,
                           Keypoint* keypoint = nullptr) const;

  CameraImpl* getImplCamera() const { return camera_impl_.get(); }
  CalibrationType::shared_ptr getGtsamCalibration() const;

  const CameraParams& getParams() const { return camera_params_; }

  RGBDCamera getRGBDCamera() const;
  std::shared_ptr<RGBDCamera> safeGetRGBDCamera() const;

 protected:
  const CameraParams camera_params_;
  std::unique_ptr<CameraImpl> camera_impl_;
};

}  // namespace dyno
