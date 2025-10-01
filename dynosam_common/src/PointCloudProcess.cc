/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au), Yiduo Wang (yiduo.wang@sydney.edu.au)
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

#include "dynosam_common/PointCloudProcess.hpp"

namespace dyno {

ObjectBBX findAABBFromCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_cloud_ptr) {
  pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud;

  // statistical outlier removal
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> outlier_remover;
  outlier_remover.setInputCloud(obj_cloud_ptr);
  outlier_remover.setMeanK(100);
  outlier_remover.setStddevMulThresh(1.0);
  outlier_remover.filter(filtered_cloud);

  // axis aligned bounding box
  pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB> bbx_extractor;
  bbx_extractor.setInputCloud(
      pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >(filtered_cloud));
  bbx_extractor.compute();

  pcl::PointXYZRGB min_point_AABB;
  pcl::PointXYZRGB max_point_AABB;
  bbx_extractor.getAABB(min_point_AABB, max_point_AABB);

  ObjectBBX aabb;
  aabb.min_bbx_point_ = gtsam::Point3(static_cast<double>(min_point_AABB.x),
                                      static_cast<double>(min_point_AABB.y),
                                      static_cast<double>(min_point_AABB.z));
  aabb.max_bbx_point_ = gtsam::Point3(static_cast<double>(max_point_AABB.x),
                                      static_cast<double>(max_point_AABB.y),
                                      static_cast<double>(max_point_AABB.z));

  return aabb;
}

CloudPerObject groupObjectCloud(
    const StatusLandmarkVector& landmarks, const gtsam::Pose3& T_world_camera,
    std::function<Color(ObjectId)> colour_generator,
    std::function<void(const pcl::PointXYZRGB&, ObjectId)> point_cb) {
  CHECK(colour_generator);
  CloudPerObject clouds_per_obj;

  for (const auto& status_estimate : landmarks) {
    Landmark lmk_world = status_estimate.value();
    const ObjectId object_id = status_estimate.objectId();
    if (status_estimate.referenceFrame() == ReferenceFrame::LOCAL) {
      lmk_world = T_world_camera * status_estimate.value();
    } else if (status_estimate.referenceFrame() == ReferenceFrame::OBJECT) {
      throw DynosamException(
          "Cannot display object point in the object reference frame");
    }

    // TODO: use convert in header!!
    pcl::PointXYZRGB pt;
    if (status_estimate.isStatic()) {
      // publish static lmk's as white
      pt = pcl::PointXYZRGB(lmk_world(0), lmk_world(1), lmk_world(2), 0, 0, 0);
    } else {
      const cv::Scalar colour = Color::uniqueId(object_id);
      pt = pcl::PointXYZRGB(lmk_world(0), lmk_world(1), lmk_world(2), colour(0),
                            colour(1), colour(2));
    }

    if (point_cb) point_cb(pt, object_id);

    clouds_per_obj[object_id].push_back(pt);
  }

  return clouds_per_obj;
}

}  // namespace dyno
