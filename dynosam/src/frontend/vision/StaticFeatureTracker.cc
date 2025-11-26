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

#include "dynosam/frontend/vision/StaticFeatureTracker.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno {

StaticFeatureTracker::StaticFeatureTracker(const TrackerParams& params,
                                           Camera::Ptr camera,
                                           ImageDisplayQueue* display_queue)
    : FeatureTrackerBase(params, camera, display_queue) {}

ExternalFlowFeatureTracker::ExternalFlowFeatureTracker(
    const TrackerParams& params, Camera::Ptr camera,
    ImageDisplayQueue* display_queue)
    : StaticFeatureTracker(params, camera, display_queue),
      static_grid_(
          static_cell_size,
          std::ceil(static_cast<double>(camera->getParams().ImageWidth()) /
                    static_cell_size),
          std::ceil(static_cast<double>(camera->getParams().ImageHeight()) /
                    static_cell_size)) {
  const auto orb_params = params.orb_params;
  orb_detector_ = std::make_unique<ORBextractor>(
      params.max_nr_keypoints_before_anms,
      static_cast<float>(orb_params.scale_factor), orb_params.n_levels,
      orb_params.init_threshold_fast, orb_params.min_threshold_fast);

  CHECK(!img_size_.empty());
}

// if previous frame is null, assume that it is the first frame... and that
// frames are always processed in order!
FeatureContainer ExternalFlowFeatureTracker::trackStatic(
    Frame::Ptr previous_frame, const ImageContainer& image_container,
    FeatureTrackerInfo& tracker_info, const cv::Mat&,
    const std::optional<gtsam::Rot3>&) {
  const ImageWrapper<ImageType::RGBMono>& rgb_wrapper = image_container.rgb();
  const cv::Mat& rgb = rgb_wrapper.toRGB();
  cv::Mat mono = ImageType::RGBMono::toMono(rgb_wrapper);
  CHECK(!mono.empty());

  const cv::Mat& motion_mask = image_container.objectMotionMask();
  CHECK(!motion_mask.empty());

  cv::Mat descriptors;
  KeypointsCV detected_keypoints;
  (*orb_detector_)(mono, cv::Mat(), detected_keypoints, descriptors);

  // assign tracked features to grid and add to static features
  FeatureContainer static_features;

  const size_t& min_tracks =
      static_cast<size_t>(params_.max_features_per_frame);
  const FrameId frame_k = image_container.frameId();

  // appy tracking (ie get correspondences)
  // TODO: only track frames that have been tracked for some time?
  if (previous_frame) {
    // TODO: for now assume consequative frames
    const FrameId frame_k_1 = previous_frame->getFrameId();
    CHECK_EQ(frame_k_1 + 1u, frame_k);

    for (Feature::Ptr previous_feature : previous_frame->static_features_) {
      const size_t tracklet_id = previous_feature->trackletId();
      const size_t age = previous_feature->age();
      const Keypoint kp = previous_feature->predictedKeypoint();

      // check kp contained before we do a static grid look up to ensure we
      // don't go out of bounds
      if (!camera_->isKeypointContained(kp)) {
        continue;
      }

      const int x = functional_keypoint::u(kp);
      const int y = functional_keypoint::v(kp);
      const size_t cell_idx = static_grid_.getCellIndex(kp);
      const ObjectId instance_label = motion_mask.at<ObjectId>(y, x);

      if (static_grid_.isOccupied(cell_idx)) continue;

      if (previous_feature->usable() && instance_label == background_label) {
        size_t new_age = age + 1;
        Feature::Ptr feature = constructStaticFeature(
            image_container, kp, new_age, tracklet_id, frame_k);
        if (feature) {
          static_features.add(feature);
          static_grid_.occupancy_[cell_idx] = true;
        }
      }
    }
  }

  // number features tracked with optical flow
  const auto n_optical_flow = static_features.size();
  tracker_info.static_track_optical_flow = n_optical_flow;

  TrackletIdManager& tracked_id_manager = TrackletIdManager::instance();

  if (static_features.size() < min_tracks) {
    // iterate over new observations
    for (size_t i = 0; i < detected_keypoints.size(); i++) {
      if (static_features.size() >= min_tracks) {
        break;
      }

      const KeypointCV& kp_cv = detected_keypoints[i];
      const int& x = kp_cv.pt.x;
      const int& y = kp_cv.pt.y;

      // if not already tracked with optical flow
      if (motion_mask.at<int>(y, x) != background_label) {
        continue;
      }

      Keypoint kp(x, y);
      const size_t cell_idx = static_grid_.getCellIndex(kp);
      if (!static_grid_.isOccupied(cell_idx)) {
        const size_t age = 0;
        size_t tracklet_id = tracked_id_manager.getTrackletIdCount();
        Feature::Ptr feature = constructStaticFeature(image_container, kp, age,
                                                      tracklet_id, frame_k);
        if (feature) {
          tracked_id_manager.incrementTrackletIdCount();
          static_grid_.occupancy_[cell_idx] = true;
          static_features.add(feature);
        }
      }
    }
  }

  static_grid_.reset();

  size_t total_tracks = static_features.size();
  tracker_info.static_track_detections = total_tracks - n_optical_flow;
  return static_features;
}

Feature::Ptr ExternalFlowFeatureTracker::constructStaticFeature(
    const ImageContainer& image_container, const Keypoint& kp, size_t age,
    TrackletId tracklet_id, FrameId frame_id) const {
  // implicit double -> int cast for pixel location
  const int x = functional_keypoint::u(kp);
  const int y = functional_keypoint::v(kp);

  const cv::Mat& rgb = image_container.rgb();
  const cv::Mat& motion_mask = image_container.objectMotionMask();
  const cv::Mat& optical_flow = image_container.opticalFlow();

  CHECK(!optical_flow.empty());
  CHECK(!motion_mask.empty());

  if (motion_mask.at<int>(y, x) != background_label) {
    return nullptr;
  }

  // check flow
  double flow_xe = static_cast<double>(optical_flow.at<cv::Vec2f>(y, x)[0]);
  double flow_ye = static_cast<double>(optical_flow.at<cv::Vec2f>(y, x)[1]);

  if (!(flow_xe != 0 && flow_ye != 0)) {
    return nullptr;
  }

  OpticalFlow flow(flow_xe, flow_ye);

  // check predicted flow is within image
  Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(kp, flow);
  if (!camera_->isKeypointContained(predicted_kp)) {
    return nullptr;
  }

  Feature::Ptr feature = std::make_shared<Feature>();
  (*feature)
      .objectId(background_label)
      .frameId(frame_id)
      .keypointType(KeyPointType::STATIC)
      .age(age)
      .trackletId(tracklet_id)
      .keypoint(kp)
      .measuredFlow(flow)
      .predictedKeypoint(predicted_kp);
  return feature;
}

KltFeatureTracker::KltFeatureTracker(const TrackerParams& params,
                                     Camera::Ptr camera,
                                     ImageDisplayQueue* display_queue)
    : StaticFeatureTracker(params, camera, display_queue) {
  detector_ = std::make_shared<SparseFeatureDetector>(
      params, FunctionalDetector::FactoryCreate(params));

  static const cv::Size klt_window_size(21, 21);  // Window size for KLT
  static const int klt_max_level = 3;             // Max pyramid levels for KLT
  static const cv::TermCriteria klt_criteria = cv::TermCriteria(
      cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.03);

  // used as flags argument for calcOpticalFlowPyrLK - initially starts as
  // default (0) flag
  int klt_flags = 0;

  lk_cuda_tracker_ = cv::cuda::SparsePyrLKOpticalFlow::create(
      klt_window_size, klt_max_level, klt_criteria.maxCount);

  CHECK_NOTNULL(detector_);
}

FeatureContainer KltFeatureTracker::trackStatic(
    Frame::Ptr previous_frame, const ImageContainer& image_container,
    FeatureTrackerInfo& tracker_info, const cv::Mat& detection_mask,
    const std::optional<gtsam::Rot3>& R_km1_k) {
  // tracked features and new features
  FeatureContainer new_tracks_and_detections;

  cv::Mat current_equialized_greyscale;
  equalizeImage(image_container, current_equialized_greyscale);

  if (!previous_frame) {
    FeatureContainer previous_inliers;
    detectFeatures(current_equialized_greyscale, image_container,
                   previous_inliers, new_tracks_and_detections, detection_mask);

    tracker_info.static_track_detections = new_tracks_and_detections.size();

    return new_tracks_and_detections;
  } else {
    // we have previous tracks
    // we should have already calculated the processed rgb image from the
    // previous frame
    cv::Mat previous_equialized_greyscale;
    equalizeImage(previous_frame->image_container_,
                  previous_equialized_greyscale);

    FeatureContainer previous_inliers;
    auto iter = previous_frame->static_features_.beginUsable();
    for (const auto& inlier_feature : iter) {
      previous_inliers.add(inlier_feature);
    }

    // if we dont actually have any previous tracks
    // this may be in cases where we have an IMU and so we have some odometry
    // but no feature tracks from the previous frame!
    if (previous_inliers.empty()) {
      FeatureContainer previous_inliers;
      detectFeatures(current_equialized_greyscale, image_container,
                     previous_inliers, new_tracks_and_detections,
                     detection_mask);
      tracker_info.static_track_detections = new_tracks_and_detections.size();
      return new_tracks_and_detections;
    }

    // Tracklet ids associated with the set of previous inliers that are now
    // outliers
    TrackletIds previous_outliers;

    // track features from the previous frame and detect new ones if necessary
    CHECK(trackPoints(
        current_equialized_greyscale, previous_equialized_greyscale,
        image_container, previous_inliers, new_tracks_and_detections,
        previous_outliers, tracker_info, detection_mask, R_km1_k));

    // after tracking, mark features in the older frame as outliers
    // TODO: (jesse) actually not sure we HAVE to do this, but better to keep
    // things as consisent as possible
    previous_frame->static_features_.markOutliers(previous_outliers);

    return new_tracks_and_detections;
  }
}

void KltFeatureTracker::equalizeImage(const ImageContainer& image_container,
                                      cv::Mat& equialized_greyscale) const {
  const ImageWrapper<ImageType::RGBMono>& rgb_wrapper = image_container.rgb();
  const cv::Mat& rgb = rgb_wrapper.toRGB();
  cv::Mat mono = ImageType::RGBMono::toMono(rgb_wrapper);
  CHECK(!mono.empty());

  mono.copyTo(equialized_greyscale);
  // CHECK(clahe_);

  // clahe_->apply(mono, equialized_greyscale);
}

std::vector<cv::Point2f> KltFeatureTracker::detectRawFeatures(
    const cv::Mat& processed_img, int number_tracked, const cv::Mat& mask) {
  KeypointsCV keypoints;
  detector_->detect(processed_img, keypoints, number_tracked, mask);

  std::vector<cv::Point2f> points;
  cv::KeyPoint::convert(keypoints, points);
  return points;
}

bool KltFeatureTracker::detectFeatures(const cv::Mat& processed_img,
                                       const ImageContainer& image_container,
                                       const FeatureContainer& current_features,
                                       FeatureContainer& new_features,
                                       const cv::Mat& detection_mask) {
  const FrameId frame_k = image_container.frameId();
  const cv::Mat& motion_mask = image_container.objectMotionMask();

  // internal detection mask that is appended with new invalid pixels
  // this builds the static detection mask over the existing input mask
  cv::Mat detection_mask_impl;
  // If we are provided with an external detection/feature mask, initalise the
  // detection mask with this and add more invalid sections to it
  if (!detection_mask.empty()) {
    CHECK_EQ(motion_mask.rows, detection_mask.rows);
    CHECK_EQ(motion_mask.cols, detection_mask.cols);
    detection_mask_impl = detection_mask.clone();
  } else {
    detection_mask_impl = cv::Mat(motion_mask.size(), CV_8U, cv::Scalar(255));
  }
  CHECK_EQ(detection_mask_impl.type(), CV_8U);
  // create mask from object mask so that all pixels > 0 are ignored (by setting
  // the value in the new mask to 0 at these locations) start with an invalid
  // mask
  cv::Mat object_feature_mask = cv::Mat::zeros(motion_mask.size(), CV_8U);
  // Set background pixels (0 in motion_mask) -> valid (1 in
  // object_feature_mask)
  object_feature_mask.setTo(255, motion_mask == 0);
  // combine with existing mask information (from current features)
  cv::bitwise_and(detection_mask_impl, object_feature_mask,
                  detection_mask_impl);

  // slow
  // add mask over objects detected in the scene
  // TODO: should just be a masking operation but treating all non-zero pixels
  // as 1 (ie make binary) and then inveverting the mask so that object pixels
  // (originally 1) become 0, indicating they should not be used! for (int i =
  // 0; i < motion_mask.rows; i++) {
  //   for (int j = 0; j < motion_mask.cols; j++) {
  //     const ObjectId label = motion_mask.at<ObjectId>(i, j);

  //     if (label != background_label) {
  //       cv::circle(
  //           detection_mask_impl, cv::Point2f(j, i),
  //           params_.min_distance_btw_tracked_and_detected_static_features,
  //           cv::Scalar(0), cv::FILLED);
  //     }
  //   }
  // }

  // add mask over current static features
  for (const auto& feature : current_features) {
    const Keypoint kp = feature->keypoint();
    CHECK(feature->usable());
    cv::circle(detection_mask_impl, cv::Point2f(kp(0), kp(1)),
               params_.min_distance_btw_tracked_and_detected_static_features,
               cv::Scalar(0), cv::FILLED);
  }

  std::vector<cv::Point2f> detected_points;
  {
    utils::TimingStatsCollector timer("static_feature_track.detect_raw");
    detected_points = detectRawFeatures(processed_img, current_features.size(),
                                        detection_mask_impl);
  }

  for (const cv::Point2f& detected_point : detected_points) {
    Keypoint kp(static_cast<double>(detected_point.x),
                static_cast<double>(detected_point.y));
    const int x = functional_keypoint::u(kp);
    const int y = functional_keypoint::v(kp);

    if (!(camera_->isKeypointContained(kp) && isWithinShrunkenImage(kp))) {
      continue;
    }

    // with the detection mask this should never happen
    if (motion_mask.at<int>(y, x) != background_label) {
      continue;
    }

    Feature::Ptr feature = constructNewStaticFeature(kp, frame_k);
    if (feature) {
      new_features.add(feature);
    }
  }

  return true;
}

bool KltFeatureTracker::trackPoints(const cv::Mat& current_processed_img,
                                    const cv::Mat& previous_processed_img,
                                    const ImageContainer& image_container,
                                    const FeatureContainer& previous_features,
                                    FeatureContainer& tracked_features,
                                    TrackletIds& outlier_previous_features,
                                    FeatureTrackerInfo& tracker_info,
                                    const cv::Mat& detection_mask,
                                    const std::optional<gtsam::Rot3>& R_km1_k) {
  if (current_processed_img.empty() || previous_processed_img.empty() ||
      previous_features.empty()) {
    return false;
  }

  outlier_previous_features.clear();

  const cv::Mat& motion_mask = image_container.objectMotionMask();
  const FrameId frame_k = image_container.frameId();

  std::vector<uchar> klt_status;
  std::vector<float> err;
  // All tracklet ids from the set of previous features to track
  TrackletIds tracklet_ids;

  // cannot just get inliers (becuase in reality this is)
  std::vector<cv::Point2f> previous_pts =
      previous_features.toOpenCV(&tracklet_ids, true);
  CHECK_EQ(previous_pts.size(), previous_features.size());
  CHECK_EQ(previous_pts.size(), tracklet_ids.size());

  static const cv::Size klt_window_size(21, 21);  // Window size for KLT
  static const int klt_max_level = 3;             // Max pyramid levels for KLT
  static const cv::TermCriteria klt_criteria = cv::TermCriteria(
      cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.03);

  // used as flags argument for calcOpticalFlowPyrLK - initially starts as
  // default (0) flag
  int klt_flags = 0;
  std::vector<cv::Point2f> current_points;
  if (R_km1_k) {
    predictKeypointsGivenRotation(current_points, previous_pts, *R_km1_k);
    klt_flags = cv::OPTFLOW_USE_INITIAL_FLOW;
  } else {
    // as per documentation the vector must have the same size as the input
    current_points.resize(previous_pts.size());
  }
  CHECK_EQ(current_points.size(), previous_pts.size());

  {
    // utils::TimingStatsCollector timer("static_feature_track.calc_LK");
    // cv::cuda::GpuMat gpu_prev_img(previous_processed_img);
    // cv::cuda::GpuMat gpu_current_img(current_processed_img);

    // cv::cuda::GpuMat d_points1(previous_pts);    // upload points
    // cv::cuda::GpuMat d_points2(current_points);  // output points
    // cv::cuda::GpuMat d_status;                   // status of each point
    // cv::cuda::GpuMat d_err;                      // error for each point

    // lk_cuda_tracker_->calc(gpu_prev_img, gpu_current_img, d_points1,
    // d_points2,
    //                        d_status, d_err);

    // // Download results back to CPU
    // d_points2.download(current_points);
    // d_status.download(status);

    cv::calcOpticalFlowPyrLK(previous_processed_img, current_processed_img,
                             previous_pts, current_points, klt_status, err,
                             klt_window_size, klt_max_level, klt_criteria,
                             klt_flags);

    // if we used OPTFLOW_USE_INITIAL_FLOW check that we actually got good flow
    if (klt_flags == cv::OPTFLOW_USE_INITIAL_FLOW) {
      static constexpr int kMinSuccessTracks = 10;
      int succ_num = 0;
      for (size_t i = 0; i < klt_status.size(); i++) {
        if (klt_status[i]) succ_num++;
      }
      if (succ_num < kMinSuccessTracks) {
        LOG(WARNING) << "Using initial flow for KLT tracking failed: only "
                     << succ_num << " tracked!";
        cv::calcOpticalFlowPyrLK(previous_processed_img, current_processed_img,
                                 previous_pts, current_points, klt_status, err,
                                 klt_window_size, klt_max_level, klt_criteria);
      }
    }

    // check flow back
    std::vector<cv::Point2f> reverse_previous_feature_points = current_points;
    std::vector<uchar> klt_reverse_status;
    cv::calcOpticalFlowPyrLK(current_processed_img, previous_processed_img,
                             current_points, reverse_previous_feature_points,
                             klt_reverse_status, err, cv::Size(21, 21), 5);
    CHECK_EQ(klt_reverse_status.size(), tracklet_ids.size());

    auto distance = [](const cv::Point2f& pt1,
                       const cv::Point2f& pt2) -> float {
      float dx = pt1.x - pt2.x;
      float dy = pt1.y - pt2.y;
      return std::sqrt(dx * dx + dy * dy);
    };
    // update klt status based on result from flow
    for (size_t i = 0; i < klt_status.size(); i++) {
      const bool both_status_good =
          klt_status.at(i) && klt_reverse_status.at(i);
      const bool within_distance =
          distance(previous_pts.at(i), reverse_previous_feature_points.at(i)) <=
          0.5;

      if (both_status_good && within_distance) {
        klt_status.at(i) = 1;
      } else {
        klt_status.at(i) = 0;
      }
    }
  }

  CHECK_EQ(previous_pts.size(), current_points.size());
  CHECK_EQ(klt_status.size(), current_points.size());

  std::vector<cv::Point2f> good_current, good_previous;
  TrackletIds good_tracklets;
  // can also look at the err?
  for (size_t i = 0; i < klt_status.size(); i++) {
    if (klt_status[i]) {
      good_current.push_back(current_points.at(i));
      good_previous.push_back(previous_pts.at(i));
      good_tracklets.push_back(tracklet_ids.at(i));
    }
  }

  // Geometric verification using RANSAC
  const cv::Mat geometric_verification_mask =
      geometricVerification(good_previous, good_current);
  std::vector<cv::Point2f> verified_current, verified_previous;
  TrackletIds verified_tracklets;
  for (int i = 0; i < geometric_verification_mask.rows; ++i) {
    if (geometric_verification_mask.at<uchar>(i)) {
      verified_current.push_back(good_current.at(i));
      verified_previous.push_back(good_previous.at(i));
      verified_tracklets.push_back(good_tracklets.at(i));
    }
  }

  CHECK_EQ(verified_tracklets.size(), verified_current.size());

  // add to tracked features
  for (size_t i = 0; i < verified_tracklets.size(); i++) {
    TrackletId tracklet_id = verified_tracklets.at(i);

    const Feature::Ptr previous_feature =
        previous_features.getByTrackletId(tracklet_id);
    // TODO: check this is the same as the previos kp to guarnatee order?

    CHECK(previous_feature->usable());

    const cv::Point2f kp_cv = verified_current.at(i);
    Keypoint kp(static_cast<double>(kp_cv.x), static_cast<double>(kp_cv.y));

    const int x = functional_keypoint::u(kp);
    const int y = functional_keypoint::v(kp);

    if (motion_mask.at<int>(y, x) != background_label) {
      continue;
    }

    if (!(camera_->isKeypointContained(kp) && isWithinShrunkenImage(kp))) {
      continue;
    }
    Feature::Ptr feature = constructStaticFeatureFromPrevious(
        kp, previous_feature, tracklet_id, frame_k);
    if (feature) {
      tracked_features.add(feature);
    }
  }

  // Get the outliers associated with the previous_features container by taking
  // the set difference between the verified and total tracklets NOTE: verified
  // tracklets are not necessary the same as the tracklets in tracked_features
  // as tracked features may excluse some features (e.g. if not in the shrunken
  // image) or (will eventually) have new tracklets after a new detection takes
  // place we just want the set difference between the original features and
  // ones we KNOW are outliers
  {
    utils::TimingStatsCollector timer("static_feature_track.find_outliers");
    determineOutlierIds(verified_tracklets, tracklet_ids,
                        outlier_previous_features);
  }

  const auto& n_tracked = tracked_features.size();
  tracker_info.static_track_optical_flow = n_tracked;

  if (tracked_features.size() <
      static_cast<size_t>(params_.min_features_per_frame)) {
    utils::TimingStatsCollector timer("static_feature_track.detect");
    // if we do not have enough features, detect more on the current image
    detectFeatures(current_processed_img, image_container, tracked_features,
                   tracked_features, detection_mask);
    tracker_info.new_static_detections = true;

    const auto n_detected = tracked_features.size() - n_tracked;
    tracker_info.static_track_detections += n_detected;
  }

  return true;
}

cv::Mat KltFeatureTracker::geometricVerification(
    const std::vector<cv::Point2f>& good_old,
    const std::vector<cv::Point2f>& good_new) const {
  if (good_old.size() >= 4) {  // Minimum number of points required for RANSAC
    cv::Mat mask;
    cv::findHomography(good_old, good_new, cv::RANSAC, 5.0, mask);
    return mask;
  } else {
    return cv::Mat::ones(
        good_old.size(), 1,
        CV_8U);  // If not enough points, assume all are inliers
  }
}

Feature::Ptr KltFeatureTracker::constructStaticFeatureFromPrevious(
    const Keypoint& kp_current, Feature::Ptr previous_feature,
    const TrackletId tracklet_id, const FrameId frame_id) const {
  CHECK(previous_feature);
  CHECK_EQ(previous_feature->trackletId(), tracklet_id);

  size_t age = previous_feature->age();
  age++;

  TrackletId tracklet_to_use = tracklet_id;
  // if age is too large, or age is zero, retrieve new tracklet id
  if (age > params_.max_feature_track_age) {
    // TrackletIdManager& tracked_id_manager = TrackletIdManager::instance();
    // tracklet_to_use = tracked_id_manager.getTrackletIdCount();
    // tracked_id_manager.incrementTrackletIdCount();
    // age = 0u;
    return nullptr;
  }

  // update previous keypoint
  previous_feature->measuredFlow(kp_current - previous_feature->keypoint());
  // This is so awful, but happens becuase the way the code was originally
  // written, we expect flow from k to k+1 (grrrr)
  previous_feature->predictedKeypoint(kp_current);

  Feature::Ptr feature = std::make_shared<Feature>();
  (*feature)
      .objectId(background_label)
      .frameId(frame_id)
      .keypointType(KeyPointType::STATIC)
      .age(age)
      .markInlier()
      .trackletId(tracklet_to_use)
      .keypoint(kp_current);

  return feature;
}

Feature::Ptr KltFeatureTracker::constructNewStaticFeature(
    const Keypoint& kp_current, const FrameId frame_id) const {
  static const auto kAge = 0u;

  TrackletIdManager& tracked_id_manager = TrackletIdManager::instance();
  TrackletId tracklet_to_use = tracked_id_manager.getAndIncrementTrackletId();

  Feature::Ptr feature = std::make_shared<Feature>();
  (*feature)
      .objectId(background_label)
      .frameId(frame_id)
      .keypointType(KeyPointType::STATIC)
      .age(kAge)
      .markInlier()
      .trackletId(tracklet_to_use)
      .keypoint(kp_current);
  return feature;
}

}  // namespace dyno
