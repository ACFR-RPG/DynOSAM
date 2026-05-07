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

#include "dynosam/frontend/vision/FeatureTracker.hpp"

#include <glog/logging.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for_each.h>

#include <mutex>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/frontend/anms/NonMaximumSuppression.h"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_cv/RGBDCamera.hpp"
#include "dynosam_nn/YoloV8ObjectDetector.hpp"

namespace dyno {

// FeatureTracker(
//     const FrontendParams& params,
//     Camera::Ptr camera,
//     StaticFeatureTracker::UniquePtr static_feature_tracker,
//     ImageDisplayQueue* display_queue = nullptr);

FeatureTracker::FeatureTracker(const FrontendParams& params, Camera::Ptr camera,
                               ImageDisplayQueue* display_queue)
    : FeatureTrackerBase(params.tracker_params, camera, display_queue),
      frontend_params_(params) {
  static_feature_tracker_ = std::make_unique<KltFeatureTracker>(
      params.tracker_params, camera, display_queue);
  CHECK(!img_size_.empty());

  LOG(INFO) << "Creating cv::cuda::SparsePyrLKOpticalFlow";
  lk_cuda_tracker_ = cv::cuda::SparsePyrLKOpticalFlow::create(
      klt_window_size_, klt_max_level_, 30);

  // max level is going to depend on number of objects we see
  // aalocate for 6 dynamic objects
  lk_tracker_dynamic_ = std::make_unique<SparseLKTracker>(
      klt_window_size_, klt_max_level_,
      6 * params.tracker_params.max_dynamic_features_per_frame,
      ImageContainer::kRGB, ImageContainer::kRGB);

  if (!params_.prefer_provided_object_detection) {
    LOG(INFO) << "Creating object detection engine";
    dyno::YoloConfig yolo_config;
    dyno::ModelConfig model_config;
    model_config.model_file = "yolov8n-seg.pt";
    object_detection_ =
        std::make_shared<dyno::YoloV8ObjectDetector>(model_config, yolo_config);
  }
}

Frame::Ptr FeatureTracker::track(FrameId frame_id, Timestamp timestamp,
                                 const ImageContainer& image_container,
                                 const std::optional<gtsam::Rot3>& R_km1_k) {
  // take "copy" of tracking_images which is then given to the frame
  // this will mean that the tracking images (input) are not necessarily the
  // same as the ones inside the returned frame
  utils::ChronoTimingStats tracking_timer("tracking_timer");
  ImageContainer input_images = image_container;

  info_ = FeatureTrackerInfo();  // clear the info
  info_.frame_id = frame_id;
  info_.timestamp = timestamp;

  if (initial_computation_) {
    // intitial computation
    const cv::Size& other_size =
        static_cast<const cv::Mat&>(input_images.rgb()).size();
    CHECK(!previous_frame_);
    CHECK_EQ(img_size_.width, other_size.width);
    CHECK_EQ(img_size_.height, other_size.height);
    initial_computation_ = false;
  } else {
    CHECK(previous_frame_);
    CHECK_EQ(previous_frame_->frame_id_, frame_id - 1u)
        << "Incoming frame id must be consequative";
  }

  // compute the ObjectBoundaryMaskResult AND detect/track object on the image
  // if required using the ObjectDetectionEngine (currently we throw away the
  // result after the function call) ObjectDetectionEngine is used if
  // params_.prefer_provided_object_detection is false
  vision_tools::ObjectBoundaryMaskResult boundary_mask_result;
  objectDetection(boundary_mask_result, input_images);

  if (input_images.hasDepth()) {
    // if we have depth just ignore all places where we have invalid depth
    // TODO: ideally we we split this based on the static/dynamic max threshold
    // too so this is account for directly in the tracking
    cv::Mat& binary_detection_mask = boundary_mask_result.boundary_mask;

    const cv::Mat& depth_image = input_images.depth();
    binary_detection_mask.setTo(0, depth_image == 0);
  }

  if (!initial_computation_ && params_.use_propogate_mask) {
    utils::ChronoTimingStats timer("propogate_mask");
    propogateMask(input_images);
  }

  // data-structure to handle which objects required re-tracking/sampling
  std::set<ObjectId> object_keyframes;
  // which dynamic features (are new and) were retroactively tracked on the
  // previous frame these features require initial depth estimation which will
  // be done by the frontend
  TrackletIds retroactive_tracks;

  auto static_track = [&](FeatureContainer& static_features) {
    VLOG(60) << "Starting static track";
    utils::ChronoTimingStats static_track_timer("static_feature_track");
    static_features = static_feature_tracker_->trackStatic(
        previous_frame_, input_images, info_,
        boundary_mask_result.boundary_mask, R_km1_k);
  };

  auto dynamic_track = [&](FeatureContainer& dynamic_features,
                           cv::Mat& dynamic_detection_mask) {
    VLOG(60) << "Starting dynamic track";
    if (params_.prefer_provided_optical_flow && input_images.hasOpticalFlow()) {
      VLOG(60) << "Starting dense object feature tracking";
      utils::ChronoTimingStats dynamic_track_timer("dynamic_feature_track");
      trackDynamic(frame_id, input_images, dynamic_features, object_keyframes,
                   dynamic_detection_mask, boundary_mask_result);
    } else {
      if (params_.prefer_provided_optical_flow &&
          !input_images.hasOpticalFlow()) {
        LOG(WARNING) << "Params specify prefer provided optical flow but input "
                        "is missing! Falling back to KLT";
      }
      VLOG(60) << "Starting KLT object feature tracking";
      utils::ChronoTimingStats dynamic_track_timer("dynamic_feature_track_klt");
      trackDynamicKLT(frame_id, input_images, dynamic_features,
                      object_keyframes, retroactive_tracks,
                      dynamic_detection_mask, boundary_mask_result);
    }
  };

  // start tracking threads since we can do this independantly
  FeatureContainer static_features, dynamic_features;
  std::thread static_track_thread(static_track, std::ref(static_features));
  std::thread dynamic_track_thread(dynamic_track, std::ref(dynamic_features),
                                   std::ref(dynamic_detection_mask_));

  static_track_thread.join();
  dynamic_track_thread.join();

  previous_tracked_frame_ = previous_frame_;  // Update previous frame (previous
                                              // to the newly created frame)

  // calculate dynamic observations for existing data
  // TODO: SingleDetectionResult really does not need the tracklet ids they
  // are never actually used!! this prevents the frame from needing to do the
  // same calculations we've alrady done
  gtsam::FastMap<ObjectId, SingleDetectionResult> object_observations;
  for (size_t i = 0; i < boundary_mask_result.objects_detected.size(); i++) {
    ObjectId object_id = boundary_mask_result.objects_detected.at(i);
    const cv::Rect& bb_detection =
        boundary_mask_result.object_bounding_boxes.at(i);

    SingleDetectionResult observation;
    observation.object_id = object_id;
    // bit of a hacky way to get the object masks as actually they should be
    // provided by the detection directly but also we want the dilated mask
    // TODO: should clone? Really this should all be const
    observation.mask = boundary_mask_result.labelled_boundary_mask == object_id;
    // observation.object_features = dynamic_features.getByObject(object_id);
    observation.bounding_box = bb_detection;

    object_observations[object_id] = observation;
  }

  utils::ChronoTimingStats f_timer("tracking_timer.frame_construction");
  auto new_frame = std::make_shared<Frame>(
      frame_id, timestamp, camera_, input_images, static_features,
      dynamic_features, object_observations, info_);

  // update tracking/sampling information for dynamic obejcts
  new_frame->retracked_objects_ =
      ObjectIds(object_keyframes.begin(), object_keyframes.end());
  new_frame->retroactive_tracks = retroactive_tracks;

  // update depth threshold information
  new_frame->setMaxBackgroundDepth(frontend_params_.max_background_depth);
  new_frame->setMaxObjectDepth(frontend_params_.max_object_depth);

  VLOG(1) << "Tracked on frame " << frame_id << " t= " << std::setprecision(15)
          << timestamp << ", object ids "
          << container_to_string(new_frame->getObjectIds());
  previous_frame_ = new_frame;
  boarder_detection_mask_ = boundary_mask_result.boundary_mask;

  return new_frame;
}

bool FeatureTracker::stereoTrack(FeatureContainer& left_features,
                                 const ImageContainer& image_container) const {
  if (!image_container.hasRightRgb()) {
    return false;
  }

  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();
  CHECK(rgbd_camera) << "Stereo imagery provided at k= "
                     << image_container.frameId()
                     << " but rgbd camera is null!";

  // TODO: reuse image pyramids from before!
  utils::ChronoTimingStats timing("stereo_track_timer");
  TrackletIds tracklets_ids;
  std::vector<cv::Point2f> left_feature_points =
      left_features.toOpenCV(&tracklets_ids, true);

  if (left_feature_points.size() < 8) {
    LOG(WARNING) << "Not enough left feature points for stereo matching...";
    return false;
  }

  // should share left image pyramid between trackers...!
  // specify tracking between left and right images
  SparseLKTracker stereo_lk_tracker(klt_window_size_, klt_max_level_,
                                    left_features.size(), ImageContainer::kRGB,
                                    ImageContainer::kRightRgb);

  const LKWorkspace& lk_result = stereo_lk_tracker.track(
      image_container, image_container, left_feature_points);

  const auto& klt_status = lk_result.status;
  const auto& klt_err = lk_result.error;
  const auto& right_feature_points = lk_result.pts;

  CHECK_EQ(left_feature_points.size(), right_feature_points.size());
  CHECK_EQ(tracklets_ids.size(), right_feature_points.size());
  CHECK_EQ(tracklets_ids.size(), klt_status.size());

  // apply fundamental matrix calc to each feature track with different id since
  // they have independant motions
  struct Tracklet2DVectors {
    std::vector<cv::Point2f> left;
    std::vector<cv::Point2f> right;
    TrackletIds tracklets;
  };
  gtsam::FastMap<ObjectId, Tracklet2DVectors> good_tracks_per_object;

  // collect points per object for outlier rejection with homography
  for (size_t i = 0; i < klt_status.size(); i++) {
    if (!klt_status[i]) {
      continue;
    }

    TrackletId tracklet_id = tracklets_ids.at(i);
    const Feature::Ptr feature = left_features.getByTrackletId(tracklet_id);

    const ObjectId object_id = feature->objectId();
    if (!good_tracks_per_object.exists(object_id)) {
      good_tracks_per_object.insert2(object_id, Tracklet2DVectors{});
    }

    Tracklet2DVectors& tracklet_vectors = good_tracks_per_object.at(object_id);
    tracklet_vectors.left.push_back(left_feature_points.at(i));
    tracklet_vectors.right.push_back(right_feature_points.at(i));
    tracklet_vectors.tracklets.push_back(tracklet_id);
  }

  // geometrically verified feature tracks and tracklets for all objects
  std::vector<cv::Point2f> verified_left;
  std::vector<cv::Point2f> verified_right;
  TrackletIds verified_tracklets;
  // apply outlier rejection via fundamental matrix for each object
  for (const auto& [object_id, tracklet_vectors] : good_tracks_per_object) {
    // need more than 8 points for fundamental matrix calc with ransac
    // points will not be marked as inliers (verified) and threfore will
    // be marked as outliers later
    if (tracklet_vectors.tracklets.size() < 8) {
      continue;
    }

    std::vector<uchar> epipolar_inliers;
    cv::findFundamentalMat(tracklet_vectors.left, tracklet_vectors.right,
                           cv::FM_RANSAC, 1.0, 0.99, epipolar_inliers);

    for (size_t i = 0; i < epipolar_inliers.size(); ++i) {
      auto tracklet_id = tracklet_vectors.tracklets.at(i);
      if (epipolar_inliers[i]) {
        verified_left.push_back(tracklet_vectors.left[i]);
        verified_right.push_back(tracklet_vectors.right[i]);

        CHECK(left_features.getByTrackletId(tracklet_id))
            << "Somehow tracklet id " << tracklet_id << " is missing!";
        verified_tracklets.push_back(tracklet_id);
      }
    }
  }

  const auto& fx = camera_->getParams().fx();
  for (size_t i = 0; i < verified_tracklets.size(); i++) {
    auto inlier_stereo_track = verified_tracklets.at(i);
    Feature::Ptr feature = left_features.getByTrackletId(inlier_stereo_track);
    CHECK(feature);
    CHECK(feature->usable());

    double uL = static_cast<double>(verified_left[i].x);
    double v = static_cast<double>(verified_left[i].y);
    double uR = static_cast<double>(verified_right[i].x);

    double disparity = uL - uR;
    // Reject near-zero disparity
    // this will also mean far away points.... multi-view triangulation
    // across frames is needed here... fall back on depth map...
    if (disparity <= 1.0 || uR < 0.0f) {
      feature->markOutlier();
      continue;
    }

    Keypoint right_kp(uR, v);
    if (!(camera_->isKeypointContained(right_kp) &&
          isWithinShrunkenImage(right_kp))) {
      feature->markOutlier();
      continue;
    }

    // TODO: should use RGBDCamera class!
    double depth = rgbd_camera->depthFromDisparity(disparity);
    // for testing
    // TODO: no max depth
    feature->depth(depth);
    feature->rightKeypoint(right_kp);
  }

  TrackletIds outlier_tracklets;
  determineOutlierIds(verified_tracklets, tracklets_ids, outlier_tracklets);

  left_features.markOutliers(outlier_tracklets);
  return true;
}

void FeatureTracker::trackDynamic(
    FrameId frame_id, const ImageContainer& image_container,
    FeatureContainer& dynamic_features, std::set<ObjectId>& object_keyframes,
    cv::Mat& dynamic_detection_mask,
    const vision_tools::ObjectBoundaryMaskResult& boundary_mask_result) {
  // first dectect dynamic points
  const cv::Mat& rgb = image_container.rgb();
  // flow is going to take us from THIS frame to the next frame (which does not
  // make sense for a realtime system)
  const cv::Mat& flow = image_container.opticalFlow();
  const cv::Mat& motion_mask = image_container.objectMotionMask();

  TrackletIdManager& tracked_id_manager = TrackletIdManager::instance();

  std::set<ObjectId> instance_labels;
  dynamic_features.clear();

  gtsam::FastMap<ObjectId, FeatureContainer> tracks_per_object;
  const cv::Mat& detection_mask = boundary_mask_result.boundary_mask;
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

  // creating tracking mask, pixel level indicator (1....N) of dynamic feature
  // location this is different to the detection_mask_impl which is a binary
  // mask (0/255) and indicates the location of all features (static and
  // dynamic) and is used to avoid detecting features near existing ones
  cv::Mat dynamic_tracking_mask =
      cv::Mat(detection_mask_impl.size(), CV_8U, cv::Scalar(0));

  if (previous_frame_) {
    const cv::Mat& previous_motion_mask =
        previous_frame_->image_container_.objectMotionMask();
    utils::ChronoTimingStats tracked_dynamic_features(
        "tracked_dynamic_features");
    for (Feature::Ptr previous_dynamic_feature :
         previous_frame_->usableDynamicIterator()) {
      const TrackletId tracklet_id = previous_dynamic_feature->trackletId();
      const size_t age = previous_dynamic_feature->age();

      const Keypoint kp = previous_dynamic_feature->predictedKeypoint();
      const int x = functional_keypoint::u(kp);
      const int y = functional_keypoint::v(kp);
      const ObjectId predicted_label = motion_mask.at<ObjectId>(y, x);

      if (!detection_mask_impl.empty()) {
        const unsigned char valid_detection =
            detection_mask_impl.at<unsigned char>(y, x);
        if (valid_detection == 0) {
          continue;
        }
      }

      ObjectId previous_label = previous_dynamic_feature->objectId();
      CHECK_NE(previous_label, background_label);
      CHECK_GT(previous_label, 0);

      PerObjectStatus& object_tracking_info =
          info_.getObjectStatus(predicted_label);
      object_tracking_info.num_previous_track++;

      // true if predicted label not on the background
      const bool is_predicted_object_label =
          predicted_label != background_label;
      // true if predicted label the same as the previous label of the tracked
      // point
      const bool is_precited_same_as_previous =
          predicted_label == previous_label;

      // update stats
      if (!is_predicted_object_label)
        object_tracking_info.num_tracked_with_background_label++;
      if (!is_precited_same_as_previous)
        object_tracking_info.num_tracked_with_different_label++;

      // only include point if it is contained, it is not static and the
      // previous label is the same as the predicted label
      if (camera_->isKeypointContained(kp) && is_predicted_object_label &&
          is_precited_same_as_previous) {
        size_t new_age = age + 1;
        double flow_xe = static_cast<double>(flow.at<cv::Vec2f>(y, x)[0]);
        double flow_ye = static_cast<double>(flow.at<cv::Vec2f>(y, x)[1]);

        OpticalFlow flow(flow_xe, flow_ye);
        const Keypoint predicted_kp =
            Feature::CalculatePredictedKeypoint(kp, flow);

        if (!isWithinShrunkenImage(predicted_kp)) {
          object_tracking_info.num_outside_shrunken_image++;
          continue;
        }

        if (flow_xe == 0 || flow_ye == 0) {
          object_tracking_info.num_zero_flow++;
          continue;
        }

        // limit point tracking of a certain age
        TrackletId tracklet_to_use = tracklet_id;
        if (new_age > params_.max_dynamic_feature_age) {
          tracklet_to_use = tracked_id_manager.getAndIncrementTrackletId();
          new_age = 0;
        }

        Feature::Ptr feature = std::make_shared<Feature>();
        (*feature)
            .objectId(predicted_label)
            .frameId(frame_id)
            .keypointType(KeyPointType::DYNAMIC)
            .age(new_age)
            .trackletId(tracklet_to_use)
            .keypoint(kp)
            .measuredFlow(flow)
            .predictedKeypoint(predicted_kp);

        if (!tracks_per_object.exists(feature->objectId())) {
          tracks_per_object.insert2(feature->objectId(), FeatureContainer{});
        }
        tracks_per_object.at(feature->objectId()).add(feature);
        instance_labels.insert(feature->objectId());
        object_tracking_info.num_track++;

        // add zero fill to detection mask to indicate the existance of a
        // tracked point at this feature location
        cv::circle(
            detection_mask_impl, cv::Point2f(x, y),
            params_.min_distance_btw_tracked_and_detected_dynamic_features,
            cv::Scalar(0), cv::FILLED);

        // fill tracking mask with tracked points, labelled with the object
        // label (j) to indicate places on object with keypoints
        cv::circle(
            dynamic_tracking_mask, cv::Point2f(x, y),
            params_.min_distance_btw_tracked_and_detected_dynamic_features,
            cv::Scalar(predicted_label), cv::FILLED);
      }
    }
    // merge features
    for (const auto& [_, features_j] : tracks_per_object) {
      dynamic_features += features_j;
    }
  }

  requiresSampling(object_keyframes, info_, image_container, tracks_per_object,
                   boundary_mask_result, dynamic_tracking_mask);

  std::set<ObjectId> objects_sampled;
  sampleDynamic(frame_id, image_container,
                object_keyframes,  // indicates which objects to sample!!
                dynamic_features, objects_sampled, detection_mask_impl);

  dynamic_detection_mask = detection_mask_impl;
}

void FeatureTracker::trackDynamicKLT(
    FrameId frame_id, const ImageContainer& image_container,
    FeatureContainer& dynamic_features, std::set<ObjectId>& object_keyframes,
    TrackletIds& retroactive_trackslet_ids, cv::Mat& dynamic_detection_mask,
    const vision_tools::ObjectBoundaryMaskResult& boundary_mask_result) {
  const cv::Mat& rgb = image_container.rgb();
  cv::Mat mono = ImageType::RGBMono::toMono(image_container.rgb());
  const cv::Mat& motion_mask = image_container.objectMotionMask();

  TrackletIdManager& tracked_id_manager = TrackletIdManager::instance();

  gtsam::FastMap<ObjectId, FeatureContainer> tracks_per_object;
  const cv::Mat& detection_mask = boundary_mask_result.boundary_mask;
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

  // creating tracking mask, pixel level indicator (1....N) of dynamic feature
  // location this is different to the detection_mask_impl which is a binary
  // mask (0/255) and indicates the location of all features (static and
  // dynamic) and is used to avoid detecting features near existing ones
  cv::Mat dynamic_tracking_mask =
      cv::Mat(detection_mask_impl.size(), CV_8U, cv::Scalar(0));

  auto constructNewDynamicFeature =
      [&](const Keypoint& kp_current, const ObjectId object_id,
          const FrameId frame_id) -> Feature::Ptr {
    static const auto kAge = 0u;

    if (!isWithinShrunkenImage(kp_current)) {
      return nullptr;
    }

    TrackletId tracklet_to_use = tracked_id_manager.getAndIncrementTrackletId();

    Feature::Ptr feature = std::make_shared<Feature>();
    (*feature)
        .objectId(object_id)
        .frameId(frame_id)
        .keypointType(KeyPointType::DYNAMIC)
        .age(kAge)
        .markInlier()
        .trackletId(tracklet_to_use)
        .keypoint(kp_current);

    CHECK(feature->usable());

    return feature;
  };

  auto constructDynamicFeatureFromPrevious =
      [&](const Keypoint& kp_current, Feature::Ptr previous_feature,
          const TrackletId tracklet_id, const ObjectId object_id,
          const FrameId frame_id) -> Feature::Ptr {
    if (!isWithinShrunkenImage(kp_current)) {
      return nullptr;
    }

    CHECK(previous_feature);
    CHECK_EQ(previous_feature->trackletId(), tracklet_id);
    CHECK_EQ(previous_feature->objectId(), object_id);

    size_t age = previous_feature->age();
    age++;

    TrackletId tracklet_to_use = tracklet_id;
    // if age is too large, or age is zero, retrieve new tracklet id
    if (age > params_.max_dynamic_feature_age) {
      return nullptr;
    }

    // update previous keypoint
    previous_feature->measuredFlow(kp_current - previous_feature->keypoint());
    // This is so awful, but happens becuase the way the code was originally
    // written, we expect flow from k to k+1 (grrrr)
    previous_feature->predictedKeypoint(kp_current);

    Feature::Ptr feature = std::make_shared<Feature>();
    (*feature)
        .objectId(object_id)
        .frameId(frame_id)
        .keypointType(KeyPointType::DYNAMIC)
        .age(age)
        .markInlier()
        .trackletId(tracklet_to_use)
        .keypoint(kp_current);

    CHECK(feature->usable());

    return feature;
  };

  FeatureContainer previous_inliers;
  if (previous_frame_) {
    FeatureContainer previous_inliers(
        previous_frame_->dynamic_features_.usableIterator());

    // All tracklet ids from the set of previous features to track
    TrackletIds tracklet_ids;
    std::vector<cv::Point2f> previous_pts =
        previous_inliers.toOpenCV(&tracklet_ids);
    CHECK_EQ(previous_pts.size(), previous_inliers.size());
    CHECK_EQ(previous_pts.size(), tracklet_ids.size());

    CHECK_EQ(previous_pts.size(), previous_inliers.size());
    CHECK_EQ(previous_pts.size(), tracklet_ids.size());

    if (tracklet_ids.size() > 0) {
      utils::ChronoTimingStats tracking_t("dynamic_feature_track_klt.tracking");

      // track from previous to current on the dynamic feature points
      const LKWorkspace& lk_result = lk_tracker_dynamic_->track(
          previous_frame_->imageContainer(), image_container, previous_pts);

      const auto& klt_status = lk_result.status;
      const auto& klt_err = lk_result.error;
      const auto& current_points = lk_result.pts;

      CHECK_EQ(previous_pts.size(), current_points.size());
      CHECK_EQ(klt_status.size(), current_points.size());

      struct Tracklet2DVectors {
        std::vector<cv::Point2f> current;
        std::vector<cv::Point2f> previous;
        TrackletIds tracklets;
      };
      gtsam::FastMap<ObjectId, Tracklet2DVectors> good_tracks_per_object;
      // collect points per object for outlier rejection with homography
      // can also look at the err?
      for (size_t i = 0; i < klt_status.size(); i++) {
        if (!klt_status[i]) {
          continue;
        }

        TrackletId tracklet_id = tracklet_ids.at(i);
        const Feature::Ptr previous_feature =
            previous_inliers.getByTrackletId(tracklet_id);

        const ObjectId object_id = previous_feature->objectId();
        if (!good_tracks_per_object.exists(object_id)) {
          good_tracks_per_object.insert2(object_id, Tracklet2DVectors{});
        }

        Tracklet2DVectors& tracklet_vectors =
            good_tracks_per_object.at(object_id);
        tracklet_vectors.current.push_back(current_points.at(i));
        tracklet_vectors.previous.push_back(previous_pts.at(i));
        tracklet_vectors.tracklets.push_back(tracklet_id);
      }

      // geometrically verified feature tracks and tracklets for all objects
      std::vector<cv::Point2f> verified_current;
      TrackletIds verified_tracklets;
      // perform outlier rejection per object
      for (const auto& [object_id, tracklet_vectors] : good_tracks_per_object) {
        std::vector<cv::Point2f> verified_current_j, verified_previous_j;
        TrackletIds verified_tracklets_j;

        vision_tools::outlierRejectHomography(
            tracklet_vectors.previous, tracklet_vectors.current,
            tracklet_vectors.tracklets, verified_previous_j, verified_current_j,
            verified_tracklets_j);

        verified_current.insert(verified_current.begin(),
                                verified_current_j.begin(),
                                verified_current_j.end());
        verified_tracklets.insert(verified_tracklets.begin(),
                                  verified_tracklets_j.begin(),
                                  verified_tracklets_j.end());
      }

      CHECK_EQ(verified_tracklets.size(), verified_current.size());

      for (size_t i = 0; i < verified_tracklets.size(); i++) {
        TrackletId tracklet_id = verified_tracklets.at(i);

        const Feature::Ptr previous_feature =
            previous_inliers.getByTrackletId(tracklet_id);
        CHECK(previous_feature->usable());

        const Keypoint kp = utils::cvPointToGtsam(verified_current.at(i));
        if (!isWithinShrunkenImage(kp)) {
          continue;
        }

        const int x = functional_keypoint::u(kp);
        const int y = functional_keypoint::v(kp);
        const ObjectId predicted_label =
            functional_keypoint::at<ObjectId>(kp, motion_mask);

        if (!detection_mask_impl.empty()) {
          const unsigned char valid_detection =
              detection_mask_impl.at<unsigned char>(y, x);
          if (valid_detection == 0) {
            continue;
          }
        }

        ObjectId previous_label = previous_feature->objectId();
        CHECK_NE(previous_label, background_label);
        CHECK_GT(previous_label, 0);

        PerObjectStatus& object_tracking_info =
            info_.getObjectStatus(predicted_label);
        object_tracking_info.num_previous_track++;

        // true if predicted label not on the background
        const bool is_predicted_object_label =
            predicted_label != background_label;
        // true if predicted label the same as the previous label of the tracked
        // point
        const bool is_precited_same_as_previous =
            predicted_label == previous_label;

        // update stats
        if (!is_predicted_object_label)
          object_tracking_info.num_tracked_with_background_label++;
        if (!is_precited_same_as_previous)
          object_tracking_info.num_tracked_with_different_label++;

        if (camera_->isKeypointContained(kp) && is_predicted_object_label &&
            is_precited_same_as_previous) {
          if (!isWithinShrunkenImage(kp)) {
            object_tracking_info.num_outside_shrunken_image++;
            continue;
          }

          Feature::Ptr feature = constructDynamicFeatureFromPrevious(
              kp, previous_feature, tracklet_id, predicted_label, frame_id);

          if (feature) {
            if (!tracks_per_object.exists(feature->objectId())) {
              tracks_per_object.insert2(feature->objectId(),
                                        FeatureContainer{});
            }
            tracks_per_object.at(feature->objectId()).add(feature);
            object_tracking_info.num_track++;

            // add zero fill to detection mask to indicate the existance of a
            // tracked point at this feature location
            cv::circle(
                detection_mask_impl, cv::Point2f(x, y),
                params_.min_distance_btw_tracked_and_detected_dynamic_features,
                cv::Scalar(0), cv::FILLED);

            // fill tracking mask with tracked points, labelled with the object
            // label (j) to indicate places on object with keypoints
            cv::circle(
                dynamic_tracking_mask, cv::Point2f(x, y),
                params_.min_distance_btw_tracked_and_detected_dynamic_features,
                cv::Scalar(predicted_label), cv::FILLED);
          }
        }
      }
    }

    for (const auto& [object_id, features_j] : tracks_per_object) {
      dynamic_features += features_j;
    }
  }

  {
    utils::ChronoTimingStats ts_t(
        "dynamic_feature_track_klt.requires_sampling");
    requiresSampling(object_keyframes, info_, image_container,
                     tracks_per_object, boundary_mask_result,
                     dynamic_tracking_mask);
  }

  double qualityLevel = 0.01;

  static constexpr float tolerance = 0.01;
  Eigen::MatrixXd binning_mask;

  const int& max_features_to_track = params_.max_dynamic_features_per_frame;
  const int& min_feature_distance =
      params_.min_distance_btw_tracked_and_detected_dynamic_features;

  std::mutex mutex;
  auto& tracking_info = info_;
  // thread safe function to calculate number of corners needed for each object
  auto getNumberCornersNeeded =
      [&tracking_info, &mutex,
       &max_features_to_track](ObjectId object_id) -> int {
    std::unique_lock<std::mutex> lock(mutex);
    const PerObjectStatus& object_tracking_info =
        tracking_info.getObjectStatus(object_id);
    const int& number_tracked = object_tracking_info.num_track;
    int nr_corners_needed = std::max(max_features_to_track - number_tracked, 0);
    return nr_corners_needed;
  };

  // new detections + retroactive tracks if provided
  struct DetectionsWithTrack {
    std::vector<KeypointCV> detections_current;
    std::vector<KeypointCV> retroactive_tracks;
    bool retroactively_tracked = false;
  };

  tbb::concurrent_unordered_map<ObjectId, DetectionsWithTrack>
      keypoints_per_object;

  cv::Mat previous_mono;
  cv::Mat previous_motion_mask;

  if (previous_frame_) {
    previous_mono =
        ImageType::RGBMono::toMono(previous_frame_->image_container_.rgb());
    previous_motion_mask = previous_frame_->image_container_.objectMotionMask();
  }

  utils::ChronoTimingStats detection_t("dynamic_feature_track_klt.detection");
  // tbb::parallel_for_each(
  //     object_keyframes.begin(), object_keyframes.end(), [&](auto& object_id)
  //     {
  for (ObjectId object_id : object_keyframes) {
    cv::Mat obj_mask = (motion_mask == object_id);
    // ignore additonal features from the tracking mask
    cv::Mat combined_mask;
    cv::bitwise_and(obj_mask, detection_mask_impl, combined_mask);

    std::vector<cv::Point2f> detected_points;
    cv::goodFeaturesToTrack(mono, detected_points, max_features_to_track,
                            qualityLevel, min_feature_distance, combined_mask);

    // the actual set of keypoints to use for ANMS
    // if we have previous frame and therefore previous tracks
    // this will only be the set of keypoints with flow
    std::vector<KeypointCV> keypoints;
    std::vector<KeypointCV> indexed_retroactive_keypoints;
    // do track back so new features already have two observations
    // so we can immediately begin tracking!
    const bool object_exists_in_previous =
        previous_frame_
            ? previous_frame_->getObjectObservations().exists(object_id)
            : false;
    const bool retroactively_tracked = !previous_mono.empty() &&
                                       object_exists_in_previous &&
                                       !detected_points.empty();
    if (retroactively_tracked) {
      // specific tracker to track from current to previous
      // TODO: ideally reuse the pyramids... between objects...
      SparseLKTracker klt_tracker(klt_window_size_, klt_max_level_,
                                  detected_points.size(), ImageContainer::kRGB,
                                  ImageContainer::kRGB);

      // track from newly detected points to points on the previous frame
      const LKWorkspace& lk_result = klt_tracker.track(
          image_container, previous_frame_->imageContainer(), detected_points);

      const auto& klt_status = lk_result.status;
      const auto& previous_points = lk_result.pts;
      const auto& current_points = detected_points;

      std::vector<cv::Point2f> good_current_j, good_previous_j;
      for (size_t i = 0; i < klt_status.size(); i++) {
        if (!klt_status[i]) {
          continue;
        }

        const cv::Point2f& kp_previous = previous_points.at(i);
        const cv::Point2f& kp_current = current_points.at(i);

        // check image region
        if (!isWithinShrunkenImage(kp_previous)) {
          continue;
        }

        if (!isWithinShrunkenImage(kp_current)) {
          continue;
        }

        // check same object in both images
        const ObjectId previous_label =
            previous_motion_mask.at<ObjectId>(kp_previous);
        if (previous_label != object_id) {
          continue;
        }

        // check a a valid image patch on the previous image
        bool valid_detection = true;
        // check detection mask on the previous frame
        if (!dynamic_detection_mask_.empty()) {
          // 0 is invalid and therefore will cast to false
          valid_detection = static_cast<bool>(
              dynamic_detection_mask_.at<unsigned char>(kp_previous));
        }

        if (!valid_detection) {
          continue;
        }

        good_current_j.push_back(kp_current);
        good_previous_j.push_back(kp_previous);
      }

      std::vector<cv::Point2f> verified_current_j, verified_previous_j;
      vision_tools::outlierRejectHomography(good_previous_j, good_current_j,
                                            verified_previous_j,
                                            verified_current_j);

      // fill keypoints with only good tracks
      keypoints.reserve(verified_previous_j.size());
      indexed_retroactive_keypoints.resize(verified_previous_j.size());
      for (size_t i = 0; i < verified_previous_j.size(); i++) {
        // use the index is as correspondance so we can recover the
        // correct previous flow after ANMS
        KeypointCV kp_curr;
        kp_curr.pt = verified_current_j.at(i);
        kp_curr.class_id = static_cast<int>(i);
        keypoints.push_back(kp_curr);

        KeypointCV kp_prev;
        kp_prev.pt = verified_previous_j.at(i);
        kp_prev.class_id = static_cast<int>(i);
        indexed_retroactive_keypoints[i] = kp_prev;
      }
    } else {
      // no previous tracks so just use the keypoints as is
      cv::KeyPoint::convert(detected_points, keypoints);
    }

    // keypoints are either direct detections or detections+retroactive
    // tracking
    std::vector<KeypointCV>& max_keypoints = keypoints;
    const size_t detected_size = max_keypoints.size();

    const int nr_corners_needed = getNumberCornersNeeded(object_id);

    AdaptiveNonMaximumSuppression non_maximum_supression(
        AnmsAlgorithmType::RangeTree);
    max_keypoints = non_maximum_supression.suppressNonMax(
        keypoints, nr_corners_needed, tolerance, img_size_.width,
        img_size_.height, 5, 5, binning_mask);

    VLOG(10) << "Kps: " << max_keypoints.size() << " for j=" << object_id
             << " after ANMS (originally " << detected_size << ", requested "
             << nr_corners_needed << ")";

    std::vector<KeypointCV> max_previous_keypoints;
    if (retroactively_tracked) {
      max_previous_keypoints.reserve(max_keypoints.size());
      CHECK_EQ(indexed_retroactive_keypoints.size(), detected_size);
      // go through and re associate tracks on previous image
      // using the class id as the cache index
      for (const auto& kp_cv : max_keypoints) {
        KeypointCV previous_kp = indexed_retroactive_keypoints[kp_cv.class_id];
        max_previous_keypoints.push_back(previous_kp);
      }
    }

    keypoints_per_object.insert(
        {object_id, DetectionsWithTrack{max_keypoints, max_previous_keypoints,
                                        retroactively_tracked}});
    // });
  }

  for (const auto& [object_id, detections_with_track] : keypoints_per_object) {
    const auto& new_keypoints = detections_with_track.detections_current;
    const auto& retroactive_tracks = detections_with_track.retroactive_tracks;

    info_.getObjectStatus(object_id).num_sampled = new_keypoints.size();

    if (detections_with_track.retroactively_tracked) {
      CHECK_EQ(new_keypoints.size(), retroactive_tracks.size());
    }

    for (size_t i = 0; i < new_keypoints.size(); i++) {
      const auto& cv_keypoint = new_keypoints.at(i);
      const Keypoint keypoint = utils::cvKeypointToGtsam(cv_keypoint);

      if (!isWithinShrunkenImage(keypoint)) {
        continue;
      }

      const ObjectId predicted_label =
          functional_keypoint::at<ObjectId>(keypoint, motion_mask);
      CHECK_EQ(predicted_label, object_id);

      // if we have a previous track create two new features
      Feature::Ptr feature_current = nullptr;
      if (detections_with_track.retroactively_tracked) {
        const auto& cv_keypoint_previous = retroactive_tracks.at(i);
        // check we got the right associations
        CHECK_EQ(cv_keypoint_previous.class_id, cv_keypoint.class_id);
        Keypoint keypoint_previous =
            utils::cvKeypointToGtsam(cv_keypoint_previous);

        auto feature_previous = constructNewDynamicFeature(
            keypoint_previous, object_id, previous_frame_->getFrameId());

        if (!feature_previous) {
          continue;
        }
        const TrackletId tracklet_id = feature_previous->trackletId();

        feature_current = constructDynamicFeatureFromPrevious(
            keypoint, feature_previous, tracklet_id, object_id, frame_id);

        if (feature_current && feature_previous) {
          previous_frame_->dynamic_features_.add(feature_previous);
          retroactive_trackslet_ids.push_back(tracklet_id);
        }
      } else {
        // assume we only have new detections no track
        feature_current =
            constructNewDynamicFeature(keypoint, object_id, frame_id);
      }

      if (feature_current) {
        // only fill detection mask not tracking mask
        cv::circle(
            detection_mask_impl, utils::gtsamPointToCv(keypoint),
            params_.min_distance_btw_tracked_and_detected_dynamic_features,
            cv::Scalar(0), cv::FILLED);
        dynamic_features.add(feature_current);
      }
    }
  }

  dynamic_detection_mask = detection_mask_impl;
}

void FeatureTracker::sampleDynamic(FrameId frame_id,
                                   const ImageContainer& image_container,
                                   const std::set<ObjectId>& objects_to_sample,
                                   FeatureContainer& dynamic_features,
                                   std::set<ObjectId>& objects_sampled,
                                   const cv::Mat& detection_mask) {
  VLOG(20) << "Begin sample dynamic";
  struct KeypointData {
    OpticalFlow flow;
    Keypoint predicted_kp;
  };

  const cv::Mat& rgb = image_container.rgb();
  // flow is going to take us from THIS frame to the next frame (which does not
  // make sense for a realtime system)
  const cv::Mat& flow = image_container.opticalFlow();
  const cv::Mat& motion_mask = image_container.objectMotionMask();

  TrackletIdManager& tracked_id_manager = TrackletIdManager::instance();

  // container to store keypoints per object
  tbb::concurrent_hash_map<ObjectId, KeypointsCV> sampled_keypoints;
  const int rows = rgb.rows;
  const int cols = rgb.cols;

  std::vector<KeypointData> cached_keypoint_data;
  cached_keypoint_data.resize(rows * cols);
  // TODO: since we're looping over the whole image here anyway why dont we also
  // use this to create the dense point cloud image and then pass it to the
  // frame!!!
  VLOG(20) << "Begin parallel dynamic sample";
  std::mutex mutex;
  tbb::parallel_for(0, rows, [&](int i) {
    const unsigned char* detection_ptr = detection_mask.ptr<unsigned char>(i);
    const ObjectId* motion_ptr = motion_mask.ptr<ObjectId>(i);
    const cv::Vec2f* flow_ptr = flow.ptr<cv::Vec2f>(i);

    for (int j = 0; j < cols; j++) {
      if (detection_ptr[j] == 0) continue;  // Skip invalid pixels

      ObjectId object_id = motion_ptr[j];

      // skip if this object does not need to be sampled
      if (objects_to_sample.find(object_id) == objects_to_sample.end()) {
        continue;
      }

      if (object_id == background_label) continue;

      double flow_xe = static_cast<double>(flow_ptr[j][0]);
      double flow_ye = static_cast<double>(flow_ptr[j][1]);
      if (flow_xe == 0 || flow_ye == 0) {
        const std::lock_guard<std::mutex> lock(mutex);
        info_.getObjectStatus(object_id).num_zero_flow++;
        continue;
      }

      OpticalFlow flow(flow_xe, flow_ye);
      Keypoint keypoint(j, i);
      Keypoint predicted_kp =
          Feature::CalculatePredictedKeypoint(keypoint, flow);

      if (isWithinShrunkenImage(keypoint)) {
        int cache_index = i * cols + j;

        // Directly assign instead of creating a new object
        cached_keypoint_data[cache_index] = {flow, predicted_kp};

        KeypointCV opencv_keypoint = utils::gtsamPointToKeyPoint(keypoint);
        opencv_keypoint.class_id = cache_index;

        tbb::concurrent_hash_map<ObjectId, KeypointsCV>::accessor acc;
        if (sampled_keypoints.insert(acc, object_id)) {
          acc->second = KeypointsCV{};  // Initialize with an empty vector
        }
        acc->second.push_back(opencv_keypoint);  // Add keypoint safely

        // is this going to be slow?
        // need the statusObject to exit by the time we get to the next tbb loop
        // so we can get the number of tracks
        const std::lock_guard<std::mutex> lock(mutex);
        info_.getObjectStatus(object_id).num_sampled++;
        // object_tracking_info.num_sampled++;

      } else {
        // const std::lock_guard<std::mutex> lock(mutex);
        // PerObjectStatus& object_tracking_info =
        // info_.getObjectStatus(object_id);
        // object_tracking_info.num_outside_shrunken_image++;
      }
    }
  });

  VLOG(20) << "End parallel dynamic sample";

  const int& max_features_to_track = params_.max_dynamic_features_per_frame;
  static constexpr float tolerance = 0.01;
  Eigen::MatrixXd binning_mask;

  VLOG(20) << "Begin parallel dynamic ANMS";
  // for(const auto& [object_id, opencv_keypoints] : sampled_keypoints) {
  tbb::parallel_for_each(
      sampled_keypoints.begin(), sampled_keypoints.end(), [&](auto& entry) {
        auto& [object_id, opencv_keypoints] = entry;

        const PerObjectStatus& object_tracking_info =
            info_.getObjectStatus(object_id);
        const int& number_tracked = object_tracking_info.num_track;

        int nr_corners_needed =
            std::max(max_features_to_track - number_tracked, 0);

        std::vector<KeypointCV>& max_keypoints = opencv_keypoints;

        const size_t sampled_size = max_keypoints.size();

        // TODO: Ssc better but maybe bad alloc????
        AdaptiveNonMaximumSuppression non_maximum_supression(
            AnmsAlgorithmType::RangeTree);
        max_keypoints = non_maximum_supression.suppressNonMax(
            opencv_keypoints, nr_corners_needed, tolerance, img_size_.width,
            img_size_.height, 5, 5, binning_mask);

        VLOG(10) << "Kps: " << max_keypoints.size() << " for j=" << object_id
                 << " after ANMS (originally " << sampled_size << ")";
        {
          const std::lock_guard<std::mutex> lock(mutex);
          info_.getObjectStatus(object_id).num_sampled = max_keypoints.size();
        }

        for (const KeypointCV& cv_keypoint : max_keypoints) {
          Keypoint keypoint = utils::cvKeypointToGtsam(cv_keypoint);
          int cache_index = cv_keypoint.class_id;
          // recover cached data
          const KeypointData& cached_data = cached_keypoint_data[cache_index];

          CHECK(isWithinShrunkenImage(keypoint));
          TrackletId tracklet_id =
              tracked_id_manager.getAndIncrementTrackletId();
          Feature::Ptr feature = std::make_shared<Feature>();
          (*feature)
              .objectId(object_id)
              .frameId(frame_id)
              .keypointType(KeyPointType::DYNAMIC)
              .age(0)
              .trackletId(tracklet_id)
              .keypoint(keypoint)
              .measuredFlow(cached_data.flow)
              .predictedKeypoint(cached_data.predicted_kp);

          {
            const std::lock_guard<std::mutex> lock(mutex);
            dynamic_features.add(feature);
            objects_sampled.insert(feature->objectId());
          }
        }
      });
  VLOG(20) << "End parallel dynamic ANMS";
}

// TODO: this should really be covarage based somehow...
void FeatureTracker::requiresSampling(
    std::set<ObjectId>& objects_to_sample, FeatureTrackerInfo& info,
    const ImageContainer& image_container,
    const gtsam::FastMap<ObjectId, FeatureContainer>& features_per_object,
    const vision_tools::ObjectBoundaryMaskResult& boundary_mask_result,
    const cv::Mat& dynamic_tracking_mask) const {
  // VLOG(20) << "Starting sampling check";
  ObjectIds detected_objects = boundary_mask_result.objects_detected;

  {
    // sanity check assert
    CHECK(equals_with_abs_tol(detected_objects,
                              boundary_mask_result.objects_detected))
        << "Explicit detected objects " << container_to_string(detected_objects)
        << " != boundary mask result: "
        << container_to_string(boundary_mask_result.objects_detected)
        << " this could happen if the object mask changes dramatically...!!";
  }

  const int& max_dynamic_point_age = params_.max_dynamic_feature_age;
  // bascially how early we want to retrack points based on their expiry
  // it takes a few frames for the feature to end up in the backend (ie. at
  // least twice, to ensure a valid track) so we want to track new points
  // earlier than that to ensure we dont have a frame with NO points
  const auto age_buffer = std::max(3, params_.dynamic_feature_age_buffer);
  const auto& min_dynamic_tracks = params_.min_dynamic_tracks;
  const auto& min_iou = params_.min_dynamic_mask_iou;
  const size_t expiry_age =
      static_cast<size_t>(max_dynamic_point_age - age_buffer);
  CHECK_GT(expiry_age, 0u);

  for (size_t i = 0; i < detected_objects.size(); i++) {
    const ObjectId object_id = detected_objects.at(i);

    // object is tracked and therefore should exist in the previous frame!
    if (info.dynamic_track.exists(object_id)) {
      auto& per_object_status = info.dynamic_track.at(object_id);

      if (!features_per_object.exists(object_id)) {
        LOG(WARNING) << "Object " << object_id
                     << " found in mask and info at k=" << info.frame_id
                     << " but missing tracked features. Skipping...";
        continue;
      }
      const FeatureContainer& per_object_tracks =
          features_per_object.at(object_id);
      const size_t num_tracked = per_object_status.num_track;
      const size_t num_previous = per_object_status.num_previous_track;
      const double survival_ratio =
          num_previous > 0 ? (double)num_tracked / (double)num_previous : 0.0;

      CHECK_EQ(num_tracked, per_object_tracks.size());
      // if more than 80% of points on the object are going to expire within the
      // next (at least 3) frames
      size_t are_geriatric = 0u;
      for (const auto& feature : per_object_tracks) {
        size_t age = feature->age();
        if (age > expiry_age) {
          are_geriatric++;
        }
      }
      // TODO: this seems wrong.... should it not be the other way around!
      const bool many_old_points =
          ((double)are_geriatric / (double)num_tracked) > 0.7;
      // if we have less than N tracks
      const bool too_few_tracks = num_tracked < min_dynamic_tracks;
      // eventually also area based tings

      // bounding box of the whole mask, representing the object detected in the
      // actual image
      const cv::Rect& detection_bb =
          boundary_mask_result.inner_boarder_object_bounding_boxes.at(i);

      // bounding box of the tracked feature points on the object
      const cv::Rect tracked_bb =
          cv::boundingRect(per_object_tracks.toOpenCV(nullptr, true));

      const double iou = utils::calculateIoU(detection_bb, tracked_bb);
      const bool small_iou = iou < min_iou;

      const bool poor_tracking = survival_ratio < 0.4;

      const bool needs_sampling =
          many_old_points || too_few_tracks || small_iou || poor_tracking;

      if (needs_sampling) {
        objects_to_sample.insert(object_id);
        per_object_status.object_resampled = true;

        VLOG(5) << "Object " << info_string(info.frame_id, object_id)
                << " requires sampling";

        VLOG_IF(5, many_old_points) << "Sampling reason: too many old points";
        VLOG_IF(5, too_few_tracks) << "Sampling reason: too few points";
        VLOG_IF(5, small_iou) << "Sampling reason: IoU too small";
        VLOG_IF(5, poor_tracking) << "Sampling reason: Poor tracking";
      }
    } else {
      objects_to_sample.insert(object_id);
      VLOG(5) << "Object " << info_string(info.frame_id, object_id)
              << " requires sampling. Sampling reason: new object";
      // this will make a new object status
      auto& per_object_status = info.getObjectStatus(object_id);
      per_object_status.object_new = true;
      per_object_status.object_resampled = true;
    }
  }
}

bool FeatureTracker::objectDetection(
    vision_tools::ObjectBoundaryMaskResult& boundary_mask_result,
    ImageContainer& image_container) {
  // from some experimental testing 10 pixles is a good boarder to add around
  // objects when the image is 640x480 assuming we have some scaling factor r,
  // width/height * r = 10 and for 640/480, r = (approx) 7.51 for images not
  // this size we will try and keep the same ratio as this seemed to work well
  // NOTE: assumes img_size_ has been set
  double image_ratio =
      static_cast<double>(img_size_.width * img_size_.height) / (640.0 * 480.0);
  static constexpr double kScalingFactorR = 7.51;
  // desired boarder thickness in pixels for a 640 x 480 image
  const int scaled_boarder_thickness =
      std::round(image_ratio * 640.0 / 480.0 * kScalingFactorR);
  // create detection mask around the boarder of each dynamic object with some
  // thickness this prevents static and dynamic points being detected around the
  // edge of the dynamic object as there are lots of inconsistencies here the
  // detection mask is in the opencv mask form: CV_8UC1 where white pixels (255)
  // are valid and black pixels (0) should not be detected on
  static constexpr bool kUseAsFeatureDetectionMask = true;
  // else run etection
  if (params_.prefer_provided_object_detection) {
    if (image_container.hasObjectMask()) {
      cv::Mat object_mask = image_container.objectMotionMask();
      // NOTE: importantly this will calculate the observed objects in this
      // frame so we dont need to recalculate!
      VLOG(50) << "Using provided object detection mask k="
               << image_container.frameId();
      vision_tools::computeObjectMaskBoundaryMask(
          boundary_mask_result, object_mask, scaled_boarder_thickness,
          kUseAsFeatureDetectionMask);
      return false;
    } else {
      LOG(FATAL) << "Params specify prefer provided object mask but input "
                    "is missing!";
    }
  } else {
    CHECK(object_detection_);
    VLOG(50) << "Running object detection and tracking inference k="
             << image_container.frameId();
    ObjectDetectionResult detection_result;
    {
      utils::ChronoTimingStats timing("tracking_timer.detection_inference");
      detection_result = object_detection_->process(image_container.rgb());
    }
    cv::Mat object_mask = detection_result.labelled_mask;

    {
      utils::ChronoTimingStats timing("tracking_timer.compute_boundary_mask");
      vision_tools::computeObjectMaskBoundaryMask(
          boundary_mask_result, detection_result, scaled_boarder_thickness,
          kUseAsFeatureDetectionMask);
    }

    // update or insert image container with object mask
    image_container.replace<ImageType::MotionMask>(ImageContainer::kObjectMask,
                                                   object_mask);
    return true;
  }
}

void FeatureTracker::propogateMask(ImageContainer& image_container) {
  if (!previous_frame_) return;

  const cv::Mat& previous_rgb = previous_frame_->image_container_.rgb();
  const cv::Mat& previous_mask =
      previous_frame_->image_container_.objectMotionMask();
  const cv::Mat& previous_flow =
      previous_frame_->image_container_.opticalFlow();

  // note reference
  cv::Mat& current_mask = image_container.objectMotionMask();

  ObjectIds instance_labels;
  for (const Feature::Ptr& dynamic_feature :
       previous_frame_->usableDynamicIterator()) {
    CHECK(dynamic_feature->objectId() != background_label);
    instance_labels.push_back(dynamic_feature->objectId());
  }

  CHECK_EQ(instance_labels.size(), previous_frame_->numDynamicUsableFeatures());
  std::sort(instance_labels.begin(), instance_labels.end());
  instance_labels.erase(
      std::unique(instance_labels.begin(), instance_labels.end()),
      instance_labels.end());
  // each row is correlated with a specific instance label and each column is
  // the tracklet id associated with that label
  std::vector<TrackletIds> object_features(instance_labels.size());

  // collect the predicted labels and semantic labels in vector

  // TODO: inliers?
  for (const Feature::Ptr& dynamic_feature :
       previous_frame_->usableDynamicIterator()) {
    CHECK(Feature::IsNotNull(dynamic_feature));
    for (size_t j = 0; j < instance_labels.size(); j++) {
      // save object label for object j with feature i
      if (dynamic_feature->objectId() == instance_labels[j]) {
        object_features[j].push_back(dynamic_feature->trackletId());
        CHECK(dynamic_feature->objectId() != background_label);
        break;
      }
    }
  }

  // check each object label distribution in the coming frame
  for (size_t i = 0; i < object_features.size(); i++) {
    // labels at the current mask using the predicted keypoint from the previous
    // frame each iteration is per label so temp_label should correspond to
    // features within the same object
    ObjectIds temp_label;
    for (size_t j = 0; j < object_features[i].size(); j++) {
      // feature at k-1
      Feature::Ptr feature = previous_frame_->dynamic_features_.getByTrackletId(
          object_features[i][j]);
      CHECK(Feature::IsNotNull(feature));
      // kp at k
      const Keypoint& predicted_kp = feature->predictedKeypoint();
      const int u = functional_keypoint::u(predicted_kp);
      const int v = functional_keypoint::v(predicted_kp);
      // ensure u and v are sitll inside the CURRENT frame
      if (u < previous_rgb.cols && u > 0 && v < previous_rgb.rows && v > 0) {
        // add instance label at predicted keypoint
        temp_label.push_back(current_mask.at<ObjectId>(v, u));
      }
    }

    // this is a lovely magic number inherited from some old code :)
    if (temp_label.size() < 20) {
      LOG(WARNING) << "not enoug points to track object " << instance_labels[i]
                   << " points size - " << temp_label.size();
      // TODO:mark has static!!???
      continue;
    }

    // find label that appears most in LabTmp()
    // (1) count duplicates
    std::map<int, int> label_duplicates;
    // k is object label
    for (int k : temp_label) {
      if (label_duplicates.find(k) == label_duplicates.end()) {
        label_duplicates.insert({k, 0});
      } else {
        label_duplicates.at(k)++;
      }
    }
    // (2) and sort them by descending order by number of times an object
    // appeared (ie. by pair.second)
    std::vector<std::pair<int, int>> sorted;
    for (auto k : label_duplicates) {
      sorted.push_back(std::make_pair(k.first, k.second));
    }

    auto sort_pair_int = [](const std::pair<int, int>& a,
                            const std::pair<int, int>& b) -> bool {
      return (a.second > b.second);
    };
    std::sort(sorted.begin(), sorted.end(), sort_pair_int);

    // recover the missing mask (time consuming!)
    // LOG(INFO) << sorted[0].first << " " << sorted[0].second << " " <<
    // instance_labels[i];
    //  if (sorted[0].second < 30)
    // {
    //   LOG(WARNING) << "not enoug points to track object " <<
    //   instance_labels[i] << " points size - "
    //                << sorted[0].second;
    //   //TODO:mark has static!!
    //   continue;
    // }
    if (sorted[0].first == 0)  //?
    // if (sorted[0].first == instance_labels[i])  //?
    {
      for (int j = 0; j < previous_rgb.rows; j++) {
        for (int k = 0; k < previous_rgb.cols; k++) {
          if (previous_mask.at<ObjectId>(j, k) == instance_labels[i]) {
            const double flow_xe =
                static_cast<double>(previous_flow.at<cv::Vec2f>(j, k)[0]);
            const double flow_ye =
                static_cast<double>(previous_flow.at<cv::Vec2f>(j, k)[1]);

            if (flow_xe == 0 || flow_ye == 0) {
              continue;
            }

            OpticalFlow flow(flow_xe, flow_ye);
            // x, y
            Keypoint kp(k, j);
            const Keypoint predicted_kp =
                Feature::CalculatePredictedKeypoint(kp, flow);

            if (!isWithinShrunkenImage(predicted_kp)) {
              continue;
            }

            if ((predicted_kp(0) < previous_rgb.cols && predicted_kp(0) > 0 &&
                 predicted_kp(1) < previous_rgb.rows && predicted_kp(1) > 0)) {
              current_mask.at<ObjectId>(functional_keypoint::v(predicted_kp),
                                        functional_keypoint::u(predicted_kp)) =
                  instance_labels[i];
              //  current_rgb
              // updated_mask_points++;
            }
          }
        }
      }
    }
  }
}

}  // namespace dyno
