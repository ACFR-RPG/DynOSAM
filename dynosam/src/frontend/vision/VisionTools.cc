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

#include "dynosam/frontend/vision/VisionTools.hpp"

#include <algorithm>  // std::set_difference, std::sort
#include <cmath>
#include <execution>
#include <future>
#include <thread>
#include <vector>  // std::vector

#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam_common/Cuda.hpp"
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_cv/CvUtils.hpp"

namespace dyno {

namespace vision_tools {

std::pair<bool, gtsam::StereoPoint2> getStereo(Feature::Ptr feature,
                                               const RGBDCamera::Ptr camera,
                                               const bool force_recalculation) {
  if (!feature->hasDepth()) {
    return {false, gtsam::StereoPoint2{}};
  }
  if (!feature->hasRightKeypoint() || force_recalculation) {
    // update the right keypoint of feature
    const auto depth = feature->depth();

    Keypoint right_kp;
    if (!camera->safeProjectRight(depth, feature->keypoint(), right_kp)) {
      return {false, gtsam::StereoPoint2{}};
    }

    // update right keypoint of feature
    feature->rightKeypoint(right_kp);
  }

  gtsam::StereoPoint2 stereo_point;
  CHECK(feature->stereoPoint(stereo_point));
  return {true, stereo_point};
}

void computeObjectMaskBoundaryMaskHelper(
    ObjectBoundaryMaskResult& result, const cv::Mat& mask, int thickness,
    bool use_as_feature_detection_mask,
    std::function<ObjectIds()> get_object_labels) {
  cv::Mat thicc_boarder;  // god im so funny
  cv::Scalar fill_colour;

  thicc_boarder = cv::Mat::zeros(mask.size(), CV_8UC1);
  const int outer_thickness = thickness;

  // background should be 255 as we're can detect in this region and boarder
  // region should be zero
  if (use_as_feature_detection_mask) {
    result.boundary_mask = cv::Mat(mask.size(), CV_8U, cv::Scalar(255));
    fill_colour = cv::Scalar(0);
  } else {
    result.boundary_mask = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));
    fill_colour = cv::Scalar(255);
  }

  cv::Mat viz = cv::Mat(mask.size(), CV_8UC3, cv::Scalar(0));

  result.objects_detected = get_object_labels();
  // this basically just creates a full mask over the existing masks using the
  // detected contours
  for (const auto object_id : result.objects_detected) {
    std::vector<std::vector<cv::Point>> detected_contours;
    // NOTE: if we use the object detection result I guess the discovered
    // rectangle here could be different to detection rectangle!
    cv::Rect detected_rect;
    CHECK(findObjectBoundingBox(mask, object_id, detected_rect,
                                detected_contours));

    CHECK_LE(object_id, 255);  // works only with uint8 types...
    cv::drawContours(thicc_boarder, detected_contours, -1, object_id,
                     cv::FILLED);
    result.object_bounding_boxes.push_back(detected_rect);
  }

  // Dilate the mask to expand outwards by 'thickness' pixels
  cv::Mat dilated_mask;
  cv::dilate(thicc_boarder, dilated_mask,
             cv::getStructuringElement(
                 cv::MORPH_RECT,
                 cv::Size(2 * outer_thickness + 1, 2 * outer_thickness + 1)));
  // Compute the outer border mask
  cv::Mat thicc_outer_boarder_mask = dilated_mask - thicc_boarder;

  // Additionally erode a little but on the inner-side of the mask
  // This helps get rid of pixels directly on the boarder which often have poor
  // depth
  static constexpr int inner_thickness = 10;
  // Generate the inner border
  cv::Mat eroded_mask;
  cv::erode(thicc_boarder, eroded_mask,
            cv::getStructuringElement(
                cv::MORPH_RECT,
                cv::Size(2 * inner_thickness + 1, 2 * inner_thickness + 1)));
  cv::Mat thicc_inner_boarder_mask = thicc_boarder - eroded_mask;

  // iterate over the objects in the eroded mask again to calculate their
  // bounding box's this will be 'very approximate' area which features can be
  // detected for object j it is approximate because it may not well be
  // approximated by a bounding box
  for (const auto object_id : result.objects_detected) {
    cv::Rect detected_rect;
    findObjectBoundingBox(eroded_mask, object_id, detected_rect);
    result.inner_boarder_object_bounding_boxes.push_back(detected_rect);
  }

  // set boarder pixels to fill colour (e.g. zero if to be used as feature
  // detection mask)
  result.boundary_mask.setTo(fill_colour, thicc_outer_boarder_mask);
  result.boundary_mask.setTo(fill_colour, thicc_inner_boarder_mask);

  result.labelled_boundary_mask = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));
  cv::bitwise_or(thicc_outer_boarder_mask, thicc_inner_boarder_mask,
                 result.labelled_boundary_mask);

  result.is_feature_detection_mask = use_as_feature_detection_mask;
}

void computeObjectMaskBoundaryMask(ObjectBoundaryMaskResult& result,
                                   const cv::Mat& mask, int thickness,
                                   bool use_as_feature_detection_mask) {
  computeObjectMaskBoundaryMaskHelper(
      result, mask, thickness, use_as_feature_detection_mask,
      [&mask]() -> ObjectIds { return getObjectLabels(mask); });
}

void computeObjectMaskBoundaryMask(
    ObjectBoundaryMaskResult& result,
    const ObjectDetectionResult& detection_result, int thickness,
    bool use_as_feature_detection_mask) {
  if (detection_result.num() == 0) {
    return;
  }

  computeObjectMaskBoundaryMaskHelper(result, detection_result.labelled_mask,
                                      thickness, use_as_feature_detection_mask,
                                      [&detection_result]() -> ObjectIds {
                                        return detection_result.objectIds();
                                      });
}

gtsam::FastMap<ObjectId, Histogram> makeTrackletLengthHistorgram(
    const Frame::Ptr frame, const std::vector<size_t>& bins) {
  // one for every object + 1 for static points
  gtsam::FastMap<ObjectId, Histogram> histograms;

  const auto& dyamic_features = frame->dynamic_features_;
  auto itr = dyamic_features.beginObjectIterator();
  for (; itr != dyamic_features.endObjectIterator(); itr++) {
    auto [object_id, feature_per_object] = *itr;

    Histogram hist(bh::make_histogram(bh::axis::variable<>(bins)));
    hist.name_ = "tacklet-length-" + std::to_string(object_id);

    for (auto feature : feature_per_object) {
      // const Feature::Ptr feature =
      // dyamic_features.getByTrackletId(tracklet_id);
      CHECK(feature);
      CHECK_EQ(feature->objectId(), object_id);
      if (feature->usable()) {
        hist.histogram_(feature->age());
      }
    }

    histograms.insert2(object_id, hist);
  }

  // collect dynamic features
  // for (const auto& [object_id, observations] :
  // frame->getObjectObservations()) {
  //   Histogram hist(bh::make_histogram(bh::axis::variable<>(bins)));
  //   hist.name_ = "tacklet-length-" + std::to_string(object_id);

  //   // for (auto tracklet_id : observations.object_features) {
  //   //   // const Feature::Ptr feature = frame->at(tracklet_id);
  //   //   // CHECK(feature);
  //   //   // if (feature->usable()) {
  //   //   //   hist.histogram_(feature->age());
  //   //   // }
  //   // }

  //   histograms.insert2(object_id, hist);
  // }

  // collect static features
  Histogram static_hist(bh::make_histogram(bh::axis::variable<>(bins)));
  static_hist.name_ = "tacklet-length-0";
  for (const auto& static_feature : frame->static_features_.usableIterator()) {
    static_hist.histogram_(static_feature->age());
  }
  histograms.insert2(background_label, static_hist);
  return histograms;
}

cv::Mat depthTo3D(const ImageWrapper<ImageType::Depth>& depth_image,
                  const cv::Mat& K) {
  const cv::Mat& depth_map = depth_image;
  int H = depth_map.rows;
  int W = depth_map.cols;

  // Camera intrinsic parameters
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  // Generate pixel grid
  cv::Mat u_grid, v_grid;
  cv::Mat u = cv::Mat::zeros(H, W, CV_64F);
  cv::Mat v = cv::Mat::zeros(H, W, CV_64F);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      u.at<double>(y, x) = static_cast<double>(x);
      v.at<double>(y, x) = static_cast<double>(y);
    }
  }

  // Normalize pixel coordinates by the intrinsic matrix
  cv::Mat X_norm = (u - cx) / fx;
  cv::Mat Y_norm = (v - cy) / fy;

  // Depth map scaling for 3D coordinates
  cv::Mat X_3D = X_norm.mul(depth_map);
  cv::Mat Y_3D = Y_norm.mul(depth_map);

  cv::Mat Z_3D = depth_map.clone();

  std::vector<cv::Mat> channels = {X_3D, Y_3D, Z_3D};
  cv::Mat point_cloud;
  cv::merge(channels, point_cloud);

  return point_cloud;  // 3-channel float matrix (H x W x 3)
}

void writeOutProjectMaskAndDepthMap(
    const ImageWrapper<ImageType::Depth>& depth_image,
    const ImageWrapper<ImageType::SemanticMask>& mask_image,
    const Camera& camera, FrameId frame_id) {
  cv::Mat point_cloud =
      depthTo3D(depth_image, camera.getParams().getCameraMatrix());

  const cv::Mat& mask = mask_image;

  // new mask of double type to match the type of the output cloud
  cv::Mat mask_double;
  mask.copyTo(mask_double);
  mask_double.convertTo(mask_double, CV_64F);

  std::vector<cv::Mat> channels = {point_cloud, mask_double};

  cv::Mat projected_cloud;
  cv::merge(channels, projected_cloud);

  static const auto folder_name = "project_mask";
  const std::string output_folder = getOutputFilePath(folder_name);

  // create write out directly if it does not exist
  createDirectory(output_folder);

  const std::string file_name =
      output_folder + "/" + std::to_string(frame_id) + ".yml";
  cv::FileStorage file(file_name, cv::FileStorage::WRITE);
  file << "matrix" << projected_cloud;
  file.release();
}

// TODO: specifically this is one type of noise using a specific RGBD
// measurement model
std::pair<gtsam::Vector3, gtsam::Matrix33> backProjectAndCovariance(
    const Feature& feature, const Camera& camera, double pixel_sigma,
    double depth_sigma) {
  const auto gtsam_camera = camera.getImplCamera();
  const auto keypoint = feature.keypoint();
  const auto u = keypoint(0);
  const auto v = keypoint(1);

  CHECK(feature.hasDepth());
  const auto depth = feature.depth();
  const auto& cam_params = camera.getParams();
  const auto fx = cam_params.fx();
  const auto fy = cam_params.fy();
  const auto cx = cam_params.cu();
  const auto cy = cam_params.cv();

  // Jacobian J of backprojection w.r.t. (u, v, d) assuming pinhole camera
  gtsam::Matrix33 J;
  J << depth / fx, 0, (u - cx) / fx, 0, depth / fy, (v - cy) / fy, 0, 0, 1;

  double pixel_sigma2 = pixel_sigma * pixel_sigma;
  double depth_sigma2 = depth_sigma * depth_sigma;
  gtsam::Matrix33 sigma_uvd =
      (Eigen::Vector3d(pixel_sigma2, pixel_sigma2, depth_sigma2)).asDiagonal();

  // Propagate to 3D covariance
  gtsam::Matrix33 sigma_3d = J * sigma_uvd * J.transpose();

  // Back project point
  gtsam::Point3 landmark = gtsam_camera->backproject(keypoint, depth);
  return {landmark, sigma_3d};
}

// void writeOutProjectMaskAndDepthMap(const ImageWrapper<ImageType::Depth>&
// depth_image, const ImageWrapper<ImageType::MotionMask>& mask_image, const
// Camera& camera, FrameId frame_id) {
//   writeOutProjectMaskAndDepthMap(depth_image,
//   ImageWrapper<ImageType::SemanticMask>(static_cast<const
//   cv::Mat&>(mask_image)), camera, frame_id);
// }

}  // namespace vision_tools

// void RGBDProcessor::updateMovingObjects(const Frame& previous_frame,
// Frame::Ptr current_frame,  cv::Mat& debug) const {
//   const cv::Mat& rgb =
//   current_frame->tracking_images_.get<ImageType::RGBMono>();

//   rgb.copyTo(debug);

//   const gtsam::Pose3& previous_pose = previous_frame.T_world_camera_;
//   const gtsam::Pose3& current_pose = current_frame->T_world_camera_;

//   const auto previous_dynamic_feature_container =
//   previous_frame.dynamic_features_; const auto
//   current_dynamic_feature_container = current_frame->dynamic_features_;

//   //iterate over each object seen in the previous frame and collect features
//   in current and previous frames to determine scene flow for(auto&
//   [object_id, current_object_observation] :
//   current_frame->object_observations_) {

//     int object_track_count = 0; //number of tracked points on the object
//     int sf_count = 0; //number of points on the object with a sufficient
//     scene flow thresh

//     const TrackletIds& object_features =
//     current_object_observation.object_features_; for(const auto tracklet_id :
//     object_features) {
//       if(previous_dynamic_feature_container.exists(tracklet_id)) {
//         CHECK(current_dynamic_feature_container.exists(tracklet_id));

//         Feature::Ptr current_feature =
//         current_dynamic_feature_container.getByTrackletId(tracklet_id);
//         Feature::Ptr previous_feature =
//         previous_dynamic_feature_container.getByTrackletId(tracklet_id);

//         if(!previous_feature->usable()) {
//           current_feature->markInvalid();
//           continue;
//         }

//         Landmark lmk_previous, lmk_current;
//         camera_->backProject(previous_feature->keypoint_,
//         previous_feature->depth_, &lmk_previous, previous_pose);
//         camera_->backProject(current_feature->keypoint_,
//         current_feature->depth_, &lmk_current, current_pose);

//         Landmark flow_world = lmk_previous - lmk_current;
//         double sf_norm = flow_world.norm();

//         if(sf_norm > params_.scene_flow_magnitude) {
//           sf_count++;
//         }

//         object_track_count++;
//       }
//     }

//     if(sf_count < 50) {
//       continue;
//     }
//     double average_flow_count = (double)sf_count /
//     (double)object_track_count;

//     LOG(INFO) << "Num points that are dynamic " << average_flow_count << "/"
//     << params_.scene_flow_percentage << " for object " << object_id;
//     if(average_flow_count > params_.scene_flow_percentage) {
//       current_object_observation.marked_as_moving_ = true;

//       static const cv::Scalar blue(255, 0, 0);

//       for(TrackletId track : object_features) {
//         Feature::Ptr current_feature =
//         current_dynamic_feature_container.getByTrackletId(track); const
//         Keypoint& px = current_feature->keypoint_; cv::circle(debug,
//         utils::gtsamPointToCV(px), 6, blue, 1);
//       }

//       //only debug stuff

//     }

//   }

// }

void determineOutlierIds(const TrackletIds& inliers,
                         const TrackletIds& tracklets, TrackletIds& outliers) {
  VLOG_IF(1, inliers.size() > tracklets.size())
      << "Usage warning: inlier size (" << inliers.size()
      << ") > tracklets size (" << tracklets.size()
      << "). Are you parsing inliers as tracklets incorrectly?";
  outliers.clear();
  TrackletIds inliers_sorted(inliers.size()),
      tracklets_sorted(tracklets.size());
  std::copy(inliers.begin(), inliers.end(), inliers_sorted.begin());
  std::copy(tracklets.begin(), tracklets.end(), tracklets_sorted.begin());

  std::sort(inliers_sorted.begin(), inliers_sorted.end());
  std::sort(tracklets_sorted.begin(), tracklets_sorted.end());

  // full set A (tracklets) must be first and inliers MUST be a subset of A for
  // the set_difference function to work
  std::set_difference(tracklets_sorted.begin(), tracklets_sorted.end(),
                      inliers_sorted.begin(), inliers_sorted.end(),
                      std::inserter(outliers, outliers.begin()));
}

}  // namespace dyno
