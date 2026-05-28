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

#include "dynosam/frontend/vision/Frame.hpp"

#include <tbb/parallel_for.h>

#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_sensors/RGBDCamera.hpp"

namespace dyno {

Frame::Frame(
    FrameId frame_id, Timestamp timestamp, Camera::Ptr camera,
    const ImageContainer& image_container,
    const FeatureContainer& static_features,
    const FeatureContainer& dynamic_features,
    const gtsam::FastMap<ObjectId, SingleDetectionResult>& object_observations,
    std::optional<FeatureTrackerInfo> tracking_info)
    : frame_id_(frame_id),
      timestamp_(timestamp),
      camera_(camera),
      image_container_(image_container),
      static_features_(static_features),
      dynamic_features_(dynamic_features),
      object_observations_(object_observations),
      tracking_info_(tracking_info) {
  // NOTE: no rectification, use camera matrix as P for cv::undistortPoints
  // see
  // https://stackoverflow.com/questions/22027419/bad-results-when-undistorting-points-using-opencv-in-python
  // i mean, this could just be shared between frames?
  const CameraParams& cam_params = camera->getParams();
  cv::Mat P = cam_params.getCameraMatrix();
  cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1);
  undistorter_ = std::make_shared<UndistorterRectifier>(P, cam_params, R);
}

Frame::Frame(FrameId frame_id, Timestamp timestamp, Camera::Ptr camera,
             const ImageContainer& image_container,
             const FeatureContainer& static_features,
             const FeatureContainer& dynamic_features,
             std::optional<FeatureTrackerInfo> tracking_info)
    : Frame(frame_id, timestamp, camera, image_container, static_features,
            dynamic_features, {}, tracking_info) {
  // dynamic info is not pre-calculated so calculate it here! SLOW!
  constructDynamicObservations();
}

std::optional<SingleDetectionResult> Frame::objectDetection(
    ObjectId object_id) const {
  std::optional<SingleDetectionResult> result;
  if (object_observations_.exists(object_id)) {
    result.emplace(object_observations_.at(object_id));
  }
  return result;
}

bool Frame::objectResampled(ObjectId object_id) const {
  return std::find(retracked_objects_.begin(), retracked_objects_.end(),
                   object_id) != retracked_objects_.end();
}

bool Frame::exists(TrackletId tracklet_id) const {
  const bool result = static_features_.exists(tracklet_id) ||
                      dynamic_features_.exists(tracklet_id);

  // debug checking -> should only be in one feature container
  if (result) {
    CHECK(!(static_features_.exists(tracklet_id) &&
            dynamic_features_.exists(tracklet_id)))
        << "Tracklet Id " << tracklet_id
        << " exists in both static and dynamic feature sets. Should be unique!";
  }
  return result;
}

size_t Frame::numStaticUsableFeatures() const {
  auto iter = usableStaticIterator();
  return static_cast<size_t>(std::distance(iter.begin(), iter.end()));
}

size_t Frame::numDynamicUsableFeatures() const {
  auto iter = usableDynamicIterator();
  return static_cast<size_t>(std::distance(iter.begin(), iter.end()));
}

size_t Frame::numTotalFeatures() const {
  return numStaticFeatures() + numDynamicFeatures();
}

Feature::Ptr Frame::at(TrackletId tracklet_id) const {
  if (!exists(tracklet_id)) {
    return nullptr;
  }

  if (static_features_.exists(tracklet_id)) {
    CHECK(!dynamic_features_.exists(tracklet_id));
    return static_features_.getByTrackletId(tracklet_id);
  } else {
    CHECK(dynamic_features_.exists(tracklet_id));
    return dynamic_features_.getByTrackletId(tracklet_id);
  }
}

bool Frame::isFeatureUsable(TrackletId tracklet_id) const {
  const auto& feature = at(tracklet_id);
  if (!feature) {
    throw std::runtime_error(
        "Failed to check feature usability - tracklet id " +
        std::to_string(tracklet_id) + " does not exist");
  }

  return feature->usable();
}

FeaturePtrs Frame::collectFeatures(TrackletIds tracklet_ids) const {
  FeaturePtrs features;
  for (const auto tracklet_id : tracklet_ids) {
    Feature::Ptr feature = at(tracklet_id);
    if (!feature) {
      throw std::runtime_error("Failed to collectFeatures - tracklet id " +
                               std::to_string(tracklet_id) + " does not exist");
    }

    features.push_back(feature);
  }

  return features;
}

Landmark Frame::backProjectToCamera(TrackletId tracklet_id) const {
  Feature::Ptr feature = at(tracklet_id);
  if (!feature) {
    throw std::runtime_error("Failed to backProjectToCamera - tracklet id " +
                             std::to_string(tracklet_id) + " does not exist");
  }

  // if no depth, project to unitsphere?
  CHECK(feature->hasDepth());

  // Landmark lmk;
  return getLandmarkFromCache(landmark_in_camera_cache_, feature,
                              gtsam::Pose3::Identity());
  // return lmk;
}

Landmark Frame::backProjectToWorld(TrackletId tracklet_id) const {
  Feature::Ptr feature = at(tracklet_id);
  if (!feature) {
    throw std::runtime_error("Failed to backProjectToWorld - tracklet id " +
                             std::to_string(tracklet_id) + " does not exist");
  }

  // if no depth, project to unitsphere?
  CHECK(feature->hasDepth());

  // Landmark lmk;
  // camera_->backProject(feature->keypoint_, feature->depth_, &lmk,
  // T_world_camera_);
  return getLandmarkFromCache(landmark_in_world_cache_, feature,
                              T_world_camera_);
}

Camera::CameraImpl Frame::getFrameCamera() const {
  const CameraParams& camera_params = camera_->getParams();
  return Camera::CameraImpl(
      T_world_camera_,
      camera_params.constructGtsamCalibration<Camera::CalibrationType>());
}

PointCloudLabelRGB::Ptr Frame::projectToDenseCloud(
    const cv::Mat* detection_mask) const {
  if (!image_container_.hasDepth()) {
    return nullptr;
  }

  PointCloudLabelRGB::Ptr cloud = pcl::make_shared<PointCloudLabelRGB>();
  const cv::Mat& depth_image = image_container_.depth();
  const cv::Mat& motion_mask = image_container_.objectMotionMask();

  if (detection_mask) {
    CHECK(utils::cvSizeEqual(detection_mask->size(), depth_image.size()));
  }

  const int rows = depth_image.rows;
  const int cols = depth_image.cols;

  // Reserve memory for the cloud (approximate max size)
  // cloud->points.reserve(rows * cols);
  cloud->points.resize(rows * cols);

  // api needs image ref
  // in this function we will not actually change the depth image
  cv::Mat& depth_image_ref = const_cast<cv::Mat&>(depth_image);
  FunctionalParallelOpenCVMat process(
      depth_image_ref, [&](cv::Mat& depth_image, int i, int j) {
        const unsigned char* detection_ptr =
            detection_mask ? detection_mask->ptr<unsigned char>(i) : nullptr;
        const ObjectId* motion_mask_ptr = motion_mask.ptr<ObjectId>(i);
        const Depth* depth_ptr = depth_image.ptr<Depth>(i);

        if (detection_mask && detection_ptr[j] == 0) return;

        const ObjectId object_id = motion_mask_ptr[j];
        const Depth depth = depth_ptr[j];

        // double depth_thresh;
        Color colour;
        if (object_id == background_label) {
          // depth_thresh = max_background_threshold_;
          colour = Color::black();
        } else {
          // depth_thresh = max_object_threshold_;
          colour = Color::uniqueId(object_id);
        }

        // if (depth > depth_thresh || depth <= 0 || !std::isfinite(depth))
        // return;
        if (depth <= 0 || !std::isfinite(depth)) return;

        // Back-projection
        const Keypoint kp(j, i);
        Landmark point;
        // this call is probably very slow
        camera_->backProject(kp, depth, &point);

        cloud->points[i * cols + j] = PointLabelRGB(
            static_cast<float>(point(0)), static_cast<float>(point(1)),
            static_cast<float>(point(2)), static_cast<std::uint8_t>(colour.r),
            static_cast<std::uint8_t>(colour.g),
            static_cast<std::uint8_t>(colour.b),
            static_cast<std::uint32_t>(object_id));
      });
  process.run();

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = false;
  return cloud;
}

bool Frame::getCorrespondences(FeaturePairs& correspondences,
                               const Frame& previous_frame,
                               KeyPointType kp_type) const {
  if (kp_type == KeyPointType::STATIC) {
    return getStaticCorrespondences(correspondences, previous_frame);
  } else {
    return getDynamicCorrespondences(correspondences, previous_frame);
  }
}

Frame::ConstructCorrespondanceFunc<Landmark, Keypoint>
Frame::landmarkWorldKeypointCorrespondance() const {
  auto func = [&](const Frame& previous_frame,
                  const Feature::Ptr& previous_feature,
                  const Feature::Ptr& current_feature) {
    if (!previous_feature->hasDepth()) {
      DYNO_THROW_MSG(DynosamException)
          << "Frame::landmarkWorldKeypointCorrespondance error in constructing "
             " (w) -> keypoint correspondences previous feature does not have "
             "depth! (j= "
          << previous_feature->objectId() << ")";
    }

    // eventuall map?
    Landmark lmk_w =
        previous_frame.backProjectToWorld(previous_feature->trackletId());
    return TrackletCorrespondance(previous_feature->trackletId(), lmk_w,
                                  current_feature->keypoint());
  };

  return std::bind(func, std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3);
}

Frame::ConstructCorrespondanceFunc<Landmark, Keypoint>
Frame::landmarkLocalKeypointCorrespondance() const {
  auto func = [&](const Frame& previous_frame,
                  const Feature::Ptr& previous_feature,
                  const Feature::Ptr& current_feature) {
    if (!previous_feature->hasDepth()) {
      DYNO_THROW_MSG(DynosamException)
          << "Frame::landmarkWorldKeypointCorrespondance error in constructing "
             " (w) -> keypoint correspondences previous feature does not have "
             "depth! (j= "
          << previous_feature->objectId() << ")";
    }

    Landmark lmk_c =
        previous_frame.backProjectToCamera(previous_feature->trackletId());
    return TrackletCorrespondance(previous_feature->trackletId(), lmk_c,
                                  current_feature->keypoint());
  };

  return std::bind(func, std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3);
}

Frame::ConstructCorrespondanceFunc<Keypoint, Keypoint>
Frame::imageKeypointCorrespondance() const {
  auto func = [&](const Frame&, const Feature::Ptr& previous_feature,
                  const Feature::Ptr& current_feature) {
    return TrackletCorrespondance(previous_feature->trackletId(),
                                  previous_feature->keypoint(),
                                  current_feature->keypoint());
  };

  return std::bind(func, std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3);
}

Frame::ConstructCorrespondanceFunc<Landmark, gtsam::Vector3>
Frame::landmarkWorldProjectedBearingCorrespondance() const {
  auto func = [&](const Frame& previous_frame,
                  const Feature::Ptr& previous_feature,
                  const Feature::Ptr& current_feature) {
    if (!previous_feature->hasDepth()) {
      DYNO_THROW_MSG(DynosamException)
          << "Frame::landmarkWorldProjectedBearingCorrespondance error in "
             "constructing "
             " (w) -> keypoint correspondences previous feature does not have "
             "depth! (j= "
          << previous_feature->objectId() << ")";
    }

    // eventuall map?
    Landmark lmk_w =
        previous_frame.backProjectToWorld(previous_feature->trackletId());

    // TODO: what if image is already undistorted
    gtsam::Vector3 projected_versor =
        undistorter_->undistortKeypointAndGetProjectedVersor(
            current_feature->keypoint());
    return TrackletCorrespondance(previous_feature->trackletId(), lmk_w,
                                  projected_versor);
  };

  return std::bind(func, std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3);
}

Frame::ConstructCorrespondanceFunc<Landmark, Landmark>
Frame::landmarkWorldPointCloudCorrespondance() const {
  auto func = [&](const Frame& previous_frame,
                  const Feature::Ptr& previous_feature,
                  const Feature::Ptr& current_feature) {
    if (!previous_feature->hasDepth()) {
      DYNO_THROW_MSG(DynosamException)
          << "Frame::landmarkWorldPointCloudCorrespondance error in "
             "constructing "
             " (w) -> keypoint correspondences previous feature does not have "
             "depth! (j= "
          << previous_feature->objectId() << ")";
    }

    // eventuall map?
    Landmark lmk_w_k_1 =
        previous_frame.backProjectToWorld(previous_feature->trackletId());
    // eventuall map?
    Landmark lmk_w_k = backProjectToWorld(current_feature->trackletId());

    return TrackletCorrespondance(previous_feature->trackletId(), lmk_w_k_1,
                                  lmk_w_k);
  };

  return std::bind(func, std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3);
}

bool Frame::getDynamicCorrespondences(FeaturePairs& correspondences,
                                      const Frame& previous_frame,
                                      ObjectId object_id) const {
  if (object_observations_.find(object_id) == object_observations_.end()) {
    LOG(WARNING) << "Object object instance id " << object_id
                 << " not found for frame " << frame_id_;
    return false;
  }

  // get the correspondences from these two iterators
  // we iterate over the current feature container which should only contain
  // features on the object and compare against the container
  vision_tools::getCorrespondences(
      correspondences, previous_frame.dynamic_features_,
      // we iterate over the current feature container which should only contain
      // features on the object
      this->dynamic_features_, UsableObjectLabelPredicate(object_id));

  return correspondences.size() > 0u;
}

bool Frame::getStaticCorrespondences(FeaturePairs& correspondences,
                                     const Frame& previous_frame) const {
  // vision_tools::getCorrespondences(
  //     correspondences, previous_frame.static_features_.usableIterator(),
  //     static_features_.usableIterator());

  vision_tools::getCorrespondences(correspondences,
                                   previous_frame.static_features_,
                                   static_features_, UsableFeaturePredicate());

  return correspondences.size() > 0u;
}

bool Frame::getDynamicCorrespondences(FeaturePairs& correspondences,
                                      const Frame& previous_frame) const {
  vision_tools::getCorrespondences(correspondences,
                                   previous_frame.dynamic_features_,
                                   dynamic_features_, UsableFeaturePredicate());
  return correspondences.size() > 0u;
}

void Frame::constructDynamicObservations() {
  object_observations_.clear();
  // assumes that the mask gets updated with the tracking label
  const ObjectIds instance_labels =
      vision_tools::getObjectLabels(image_container_.objectMotionMask());

  auto inlier_iterator = dynamic_features_.usableIterator();
  for (const Feature::Ptr& dynamic_feature : inlier_iterator) {
    CHECK(!dynamic_feature->isStatic());
    CHECK(dynamic_feature->usable());

    const ObjectId object_id = dynamic_feature->objectId();
    // this check is just for sanity!
    CHECK(std::find(instance_labels.begin(), instance_labels.end(),
                    object_id) != instance_labels.end())
        << "Missing " << object_id << " in "
        << container_to_string(instance_labels);

    if (object_observations_.find(object_id) == object_observations_.end()) {
      SingleDetectionResult observation;
      observation.object_id = object_id;
      object_observations_[object_id] = observation;
    }

    // object_observations_[object_id].object_features.push_back(
    //     dynamic_feature->trackletId());
  }

  // now construct image masks from tracking mask
  // For each tracked object, find its id in the mask
  // and draw it.
  // We apply some eroding/dilation on it to make the resulting submask smoother
  // so that we can more easily fit an rectangle to it
  const cv::Mat& mask = image_container_.objectMotionMask();
  for (auto& object_observation_pair : object_observations_) {
    const ObjectId object_id = object_observation_pair.first;
    SingleDetectionResult& obs = object_observation_pair.second;
    vision_tools::findObjectBoundingBox(mask, object_id, obs.bounding_box);
  }
}

Landmark Frame::getLandmarkFromCache(LandmarkMap& cache, Feature::Ptr feature,
                                     const gtsam::Pose3& X_world) const {
  // TODO: dont cache as we now update the optical flow and the depth in the
  // frontend and cacheing it will not use the right values!!!
  //  const auto& it = cache.find(feature->tracklet_id_);
  //  if(it != cache.end()) {
  //      return it->second;
  //  }

  Landmark lmk;
  camera_->backProject(feature->keypoint(), feature->depth(), &lmk, X_world);
  // cache.insert({feature->tracklet_id_, lmk});
  return lmk;
}

DepthUpdater::DepthUpdater(FeatureTracker* tracker)
    : tracker_(CHECK_NOTNULL(tracker)) {
  const auto& params = tracker_->frontendParams();
  max_background_threshold_ = params.max_background_depth;
  max_object_threshold_ = params.max_object_depth;
}

bool DepthUpdater::update(Frame::Ptr frame,
                          const std::optional<TrackletIds>& tracklets) const {
  const ImageContainer& container = frame->imageContainer();

  if (container.hasRightRgb()) {
    return updateFromStereo(frame, tracklets);
  } else if (container.hasDepth()) {
    return updateFromDepth(frame, tracklets);
  }
  return false;
}

bool DepthUpdater::updateFromDepth(
    Frame::Ptr frame, const std::optional<TrackletIds>& tracklets) const {
  FeatureContainer features = collectFeatures(frame, tracklets);

  const ImageContainer& container = frame->imageContainer();
  const cv::Mat& depth = container.depth();

  const auto& camera_ptr = CHECK_NOTNULL(frame->getCamera());
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_ptr->safeGetRGBDCamera();
  CHECK_NOTNULL(rgbd_camera);

  for (Feature::Ptr feature : features) {
    const int x = functional_keypoint::u(feature->keypoint());
    const int y = functional_keypoint::v(feature->keypoint());
    const ObjectId j = feature->objectId();
    const Depth d = depth.at<Depth>(y, x);

    const Depth max_depth = (j == background_label) ? max_background_threshold_
                                                    : max_object_threshold_;

    if (d > max_depth || d <= 0) {
      feature->markInvalid();
      feature->depth(Feature::invalid_depth);
    } else {
      feature->depth(d);
      // right projected right keypoint with virtual camera
      //  if projection fails, set as outlier
      if (!rgbd_camera->projectRight(feature)) {
        feature->depth(Feature::invalid_depth);
        feature->markOutlier();
      }
    }
  }

  return true;
}

bool DepthUpdater::updateFromStereo(
    Frame::Ptr frame, const std::optional<TrackletIds>& tracklets) const {
  const ImageContainer& container = frame->imageContainer();

  if (!container.hasRightRgb()) {
    return false;
  }

  FeatureContainer features = collectFeatures(frame, tracklets);
  return tracker_->stereoTrack(features, container);
}

FeatureContainer DepthUpdater::collectFeatures(
    Frame::Ptr frame, const std::optional<TrackletIds>& tracklets) const {
  FeatureContainer features;
  if (tracklets) {
    for (TrackletId tracklet_id : *tracklets) {
      Feature::Ptr feature = frame->at(tracklet_id);
      CHECK_NOTNULL(feature);

      features.add(feature);
    }
  } else {
    // collect all features
    for (const auto& f : frame->static_features_.usableIterator()) {
      features.add(f);
    }

    for (const auto& f : frame->dynamic_features_.usableIterator()) {
      features.add(f);
    }
  }
  return features;
}

}  // namespace dyno
