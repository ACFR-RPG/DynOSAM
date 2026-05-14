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

#include "dynosam/frontend/vision/VisionTools.hpp"

namespace dyno {

namespace vision_tools {

/// @brief A modified of gtsam::triangulateSafe that allows us to parse useLOST
/// as this is not available on the 4.2 version of gtsam currently used.
/// @tparam CAMERA
/// @param cameras
/// @param measured
/// @param params
/// @param useLOST
/// @return
template <class CAMERA>
gtsam::TriangulationResult triangulateSafe(
    const gtsam::CameraSet<CAMERA>& cameras,
    const typename CAMERA::MeasurementVector& measured,
    const gtsam::TriangulationParameters& params, const bool useLOST) {
  size_t m = cameras.size();

  // if we have a single pose the corresponding factor is uninformative
  if (m < 2)
    return gtsam::TriangulationResult::Degenerate();
  else
    // We triangulate the 3D position of the landmark
    try {
      gtsam::Point3 point = gtsam::triangulatePoint3<CAMERA>(
          cameras, measured, params.rankTolerance, params.enableEPI,
          params.noiseModel, useLOST);

      // Check landmark distance and re-projection errors to avoid outliers
      size_t i = 0;
      double maxReprojError = 0.0;
      for (const CAMERA& camera : cameras) {
        const gtsam::Pose3& pose = camera.pose();
        if (params.landmarkDistanceThreshold > 0 &&
            gtsam::distance3(pose.translation(), point) >
                params.landmarkDistanceThreshold)
          return gtsam::TriangulationResult::FarPoint();
#ifdef GTSAM_THROW_CHEIRALITY_EXCEPTION
        // verify that the triangulated point lies in front of all cameras
        // Only needed if this was not yet handled by exception
        const gtsam::Point3& p_local = pose.transformTo(point);
        if (p_local.z() <= 0) return gtsam::TriangulationResult::BehindCamera();
#endif
        // Check reprojection error
        if (params.dynamicOutlierRejectionThreshold > 0) {
          const typename CAMERA::Measurement& zi = measured.at(i);
          gtsam::Point2 reprojectionError = camera.reprojectionError(point, zi);
          maxReprojError = std::max(maxReprojError, reprojectionError.norm());
        }
        i += 1;
      }
      // Flag as degenerate if average reprojection error is too large
      if (params.dynamicOutlierRejectionThreshold > 0 &&
          maxReprojError > params.dynamicOutlierRejectionThreshold)
        return gtsam::TriangulationResult::Outlier();

      // all good!
      return gtsam::TriangulationResult(point);
    } catch (gtsam::TriangulationUnderconstrainedException&) {
      // This exception is thrown if
      // 1) There is a single pose for triangulation - this should not happen
      // because we checked the number of poses before 2) The rank of the matrix
      // used for triangulation is < 3: rotation-only, parallel cameras (or
      // motion towards the landmark)
      return gtsam::TriangulationResult::Degenerate();
    } catch (const gtsam::TriangulationCheiralityException&) {
      // point is behind one of the cameras: can be the case of
      // close-to-parallel cameras or may depend on outliers
      return gtsam::TriangulationResult::BehindCamera();
    }
}

}  // namespace vision_tools
}  // namespace dyno
