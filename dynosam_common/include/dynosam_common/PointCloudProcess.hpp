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

#pragma once

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_types.h>

#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/viz/Colour.hpp"

namespace dyno {

/// @brief Map of ObjectId's to XYZRGB Point Clouds
using CloudPerObject =
    gtsam::FastMap<ObjectId, pcl::PointCloud<pcl::PointXYZRGB>>;

/// @brief Alias to XYZ point with label (std::uint32_t)
using PointLabel = pcl::PointXYZL;
/// @brief Alias to XYZ point with label (std::uint32_t) and RGB values
using PointLabelRGB = pcl::PointXYZRGBL;
/// @brief Alias to a PCL Point Cloud of type PointLabelRGB
using PointCloudLabelRGB = pcl::PointCloud<PointLabelRGB>;

template <>
inline bool convert(const LandmarkStatus& status, pcl::PointXYZRGB& point) {
  const gtsam::Point3 lmk = status.value();
  const ObjectId object_id = status.objectId();
  if (status.isStatic()) {
    point =
        pcl::PointXYZRGB(static_cast<float>(lmk(0)), static_cast<float>(lmk(1)),
                         static_cast<float>(lmk(2)), 0, 0, 0);
  } else {
    // has to recalculate this every time!!
    const Color colour = Color::uniqueId(object_id);
    point = pcl::PointXYZRGB(
        static_cast<float>(lmk(0)), static_cast<float>(lmk(1)),
        static_cast<float>(lmk(2)), static_cast<std::uint8_t>(colour.r),
        static_cast<std::uint8_t>(colour.g),
        static_cast<std::uint8_t>(colour.b));
  }
  return true;
}

template <typename Value, typename Point>
void convert_pcl(const GenericTrackedStatusVector<Value>& status_vector,
                 pcl::PointCloud<Point>& cloud) {
  for (const auto& value : status_vector) {
    Point point;
    // call the status type to the point type explicitly
    // allows any type w
    convert(value, point);
    cloud.push_back(point);
  }
}

template <typename Point>
void convert(const StatusLandmarkVector& status_vector,
             pcl::PointCloud<Point>& cloud) {
  convert_pcl(status_vector, cloud);
}

/**
 * @brief Constructs a gtsam::Point3 from the x,y,z components of the PointT.
 *
 * Casts to double type for save conversions
 *
 * @tparam PointT
 * @param point
 * @return gtsam::Point3
 */
template <typename PointT>
inline gtsam::Point3 pclPointToGtsam(const PointT& point) {
  return gtsam::Point3(static_cast<double>(point.x),
                       static_cast<double>(point.y),
                       static_cast<double>(point.z));
}

/**
 * @brief A struct for bounding boxes estimated by pcl - both oriented (OBB) and
 * axis aligned (AABB). AABB does not require orientation (should be set as
 * Identity), nor position. OBB requires all and the min/max points are w.r.t
 * the centroid (i.e. the bbx_position_)
 *
 */
struct ObjectBBX {
  //! Orientation is included, but identity for AABB
  ObjectBBX() : orientation_(gtsam::Rot3::Identity()) {}

  //! in the case of AABB, the min-max points are sufficient to put the box into
  //! the reference in the case of OBB, the min-max points only define the size
  //! of the box
  gtsam::Point3 min_bbx_point_;
  gtsam::Point3 max_bbx_point_;
  //! in the case of AABB, the position is zero
  //! in the case of OBB, the position is the centroid of the bbx
  gtsam::Point3 bbx_position_;
  //! in the case of AABB, the orientation is identity
  //! in the case of OBB, the orientation is that of the bbx
  gtsam::Rot3 orientation_;

  gtsam::Pose3 pose() const {
    return gtsam::Pose3(orientation_, bbx_position_);
  }
};

using BbxPerObject = gtsam::FastMap<ObjectId, ObjectBBX>;

/**
 * @brief This function goes through the input 3D landmarks and group them into
 * point clusters based on their object labels these 3D landmarks are expected
 * to be in the sensor reference frame, and T_world_camera puts them into the
 * world frame CloudPerObject is in the world frame.
 *
 * @param landmarks const StatusLandmarkVector&
 * @param T_world_camera const gtsam::Pose3& Transform from camera to world
 * frame
 * @param colour_generator std::function<Color(ObjectId)> functional colour
 * generator based on the object id. Defaults to Color::uniqueId.
 * @param point_cb std::function<void(const pcl::PointXYZRGB&, ObjectId)>
 * callback for each created PCL point. Default is nullptr
 * @return CloudPerObject XYZRGB point cloud per obect in the world frame
 */
CloudPerObject groupObjectCloud(
    const StatusLandmarkVector& landmarks, const gtsam::Pose3& T_world_camera,
    std::function<Color(ObjectId)> colour_generator = Color::uniqueObjectId,
    std::function<void(const pcl::PointXYZRGB&, ObjectId)> point_cb = nullptr);

/**
 * @brief Overload of groupObjectCloud.
 *
 * @param landmarks const StatusLandmarkVector&
 * @param T_world_camera const gtsam::Pose3&
 * @param point_cb std::function<void(const pcl::PointXYZRGB&, ObjectId)>
 * @return CloudPerObject
 */
inline CloudPerObject groupObjectCloud(
    const StatusLandmarkVector& landmarks, const gtsam::Pose3& T_world_camera,
    std::function<void(const pcl::PointXYZRGB&, ObjectId)> point_cb) {
  return groupObjectCloud(landmarks, T_world_camera, Color::uniqueObjectId,
                          point_cb);
}

/**
 * @brief Overload of groupObjectCloud.
 *
 * @param landmarks const StatusLandmarkVector&
 * @param T_world_camera const gtsam::Pose3&
 * @param colour_generator std::function<Color(ObjectId)>
 * @return CloudPerObject
 */
inline CloudPerObject groupObjectCloud(
    const StatusLandmarkVector& landmarks, const gtsam::Pose3& T_world_camera,
    std::function<Color(ObjectId)> colour_generator) {
  return groupObjectCloud(landmarks, T_world_camera, colour_generator, nullptr);
}

/**
 * @brief Compute a MomentOfInertiaEstimation using an input point cloud.
 *
 * Applies outlier rejection using the given arguments and then computes the
 * moment of inertial from the input cloud
 *
 * @tparam PointT A PCL type
 * @param obj_cloud_ptr const typename pcl::PointCloud<PointT>::Ptr
 * @param bbx_extractor pcl::MomentOfInertiaEstimation<PointT>&
 * @param mean_k int mean k argument for outlier rejection
 * @param std_dev_thresh float std-deviation argument for outlier rejection
 */
template <typename PointT>
void computeMomentOfInertial(
    const typename pcl::PointCloud<PointT>::Ptr obj_cloud_ptr,
    pcl::MomentOfInertiaEstimation<PointT>& bbx_extractor, int mean_k = 100,
    float std_dev_thresh = 1.0) {
  pcl::PointCloud<PointT> filtered_cloud;

  // statistical outlier removal
  pcl::StatisticalOutlierRemoval<PointT> outlier_remover;
  outlier_remover.setInputCloud(obj_cloud_ptr);
  outlier_remover.setMeanK(mean_k);
  outlier_remover.setStddevMulThresh(std_dev_thresh);
  outlier_remover.filter(filtered_cloud);

  bbx_extractor.setInputCloud(
      pcl::make_shared<pcl::PointCloud<PointT>>(filtered_cloud));
  bbx_extractor.compute();
}

/**
 * @brief Compute the Axis Aligned Bounding Box (AABB) from an input cloud.
 *
 * The returned ObjectBBX only has the min and max bbx point set.
 *
 * @tparam PointT A PCL type
 * @param obj_cloud_ptr const typename pcl::PointCloud<PointT>::Ptr
 * @return ObjectBBX
 */
template <typename PointT>
ObjectBBX findAABBFromCloud(
    const typename pcl::PointCloud<PointT>::Ptr obj_cloud_ptr) {
  pcl::PointCloud<PointT> filtered_cloud;

  pcl::MomentOfInertiaEstimation<PointT> bbx_extractor;
  computeMomentOfInertial(obj_cloud_ptr, bbx_extractor);

  PointT min_point_AABB;
  PointT max_point_AABB;
  bbx_extractor.getAABB(min_point_AABB, max_point_AABB);

  ObjectBBX aabb;
  aabb.min_bbx_point_ = pclPointToGtsam(min_point_AABB);
  aabb.max_bbx_point_ = pclPointToGtsam(max_point_AABB);

  return aabb;
}

/**
 * @brief Compute Oriented bounding box (OBB) from an input point cloud.
 *
 * The returned ObjectBBX has the min/max bbx point set, as well as the position
 * and orientation
 *
 * @tparam PointT PointT A PCL type.
 * @param obj_cloud_ptr const typename pcl::PointCloud<PointT>::Ptr.
 * @return ObjectBBX
 */
template <typename PointT>
ObjectBBX findOBBFromCloud(
    const typename pcl::PointCloud<PointT>::Ptr obj_cloud_ptr) {
  // oriented aligned bounding box
  pcl::MomentOfInertiaEstimation<PointT> bbx_extractor;
  computeMomentOfInertial(obj_cloud_ptr, bbx_extractor);

  PointT min_point_AABB;
  PointT max_point_AABB;
  PointT position;
  Eigen::Matrix3f orientation;
  bbx_extractor.getOBB(min_point_AABB, max_point_AABB, position, orientation);

  ObjectBBX obb;
  obb.min_bbx_point_ = pclPointToGtsam(min_point_AABB);
  obb.max_bbx_point_ = pclPointToGtsam(max_point_AABB);
  obb.bbx_position_ = pclPointToGtsam(position);
  obb.orientation_ = gtsam::Rot3(orientation.cast<double>());

  return obb;
}

template <typename PointT>
pcl::PointCloud<pcl::PointXYZ> findLineListPointsFromAABBMinMax(
    const PointT& min_point_AABB, const PointT& max_point_AABB) {
  pcl::PointCloud<pcl::PointXYZ> line_list_points;

  // bottom
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
  // top
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, max_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, max_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z));
  // vertical
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, max_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
  line_list_points.push_back(
      pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z));

  return line_list_points;
}

/**
 * @brief Specalise function on gtsam::Point3 type.
 *
 * gtsam::Point3 does not follow the expect concept definition of a
 * pcl::PointXYZ type (i.e x is a function, not a property)
 *
 */
template <>
inline pcl::PointCloud<pcl::PointXYZ> findLineListPointsFromAABBMinMax(
    const gtsam::Point3& min_point_AABB, const gtsam::Point3& max_point_AABB) {
  pcl::PointXYZ min_point_AABB_pcl =
      pcl::PointXYZ((float)min_point_AABB.x(), (float)min_point_AABB.y(),
                    (float)min_point_AABB.z());
  pcl::PointXYZ max_point_AABB_pcl =
      pcl::PointXYZ((float)max_point_AABB.x(), (float)max_point_AABB.y(),
                    (float)max_point_AABB.z());
  return findLineListPointsFromAABBMinMax<pcl::PointXYZ>(min_point_AABB_pcl,
                                                         max_point_AABB_pcl);
}

}  // namespace dyno
