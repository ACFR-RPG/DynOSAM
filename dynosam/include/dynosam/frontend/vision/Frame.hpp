/*
 *   Copyright (c) 2023 Jesse Morris (jesse.morris@sydney.edu.au)
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

#include <functional>

#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_common/PointCloudProcess.hpp"
#include "dynosam_common/StructuredContainers.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_cv/Camera.hpp"
#include "dynosam_cv/Feature.hpp"
#include "dynosam_cv/ImageContainer.hpp"
#include "dynosam_cv/UndistortRectifier.hpp"

namespace dyno {

class Frame {
 public:
  DYNO_POINTER_TYPEDEFS(Frame)

  const FrameId frame_id_;
  const Timestamp timestamp_;
  Camera::Ptr camera_;
  const ImageContainer image_container_;

  gtsam::Pose3 T_world_camera_ = gtsam::Pose3::Identity();

  FeatureContainer static_features_;
  FeatureContainer dynamic_features_;

  std::optional<FeatureTrackerInfo>
      tracking_info_;  //! information from the tracker that was used to created
                       //! this frame

  //! Depricate as unused!
  static ObjectId global_object_id;

  // semantic instance label to object observation (by the actual observations
  // in the image) set in constructor
  std::map<ObjectId, DynamicObjectObservation> object_observations_;
  MotionEstimateMap
      motion_estimates_;  // map of object ids to object motions that take the
                          // object from k-1 to k in W. Updated in the frontend
                          // and will not initially have a value

  Frame(FrameId frame_id, Timestamp timestamp, Camera::Ptr camera,
        const ImageContainer& image_container,
        const FeatureContainer& static_features,
        const FeatureContainer& dynamic_features,
        const std::map<ObjectId, DynamicObjectObservation>& object_observations,
        std::optional<FeatureTrackerInfo> tracking_info = {});

  Frame(FrameId frame_id, Timestamp timestamp, Camera::Ptr camera,
        const ImageContainer& image_container,
        const FeatureContainer& static_features,
        const FeatureContainer& dynamic_features,
        std::optional<FeatureTrackerInfo> tracking_info = {});

  inline FrameId getFrameId() const { return frame_id_; }
  inline Timestamp getTimestamp() const { return timestamp_; }
  inline const Camera::Ptr& getCamera() const { return camera_; }

  /**
   * @brief Returns the pose of the sensor at this frame w.r.t the world frame
   * i.e ^wT_c such that a point, p measured in the camera frame can be put in
   * the world frame using this pose e.g. ^wp = ^wT_c * ^cp
   *
   * @return const gtsam::Pose3&
   */
  const gtsam::Pose3& getPose() const { return T_world_camera_; }

  inline const std::map<ObjectId, DynamicObjectObservation>&
  getObjectObservations() const {
    return object_observations_;
  }
  inline std::map<ObjectId, DynamicObjectObservation>& getObjectObservations() {
    return object_observations_;
  }

  // note: this doesnt mean inliers/outliers in the current frame (as this
  // happens after tracking)
  inline const std::optional<FeatureTrackerInfo>& getTrackingInfo() const {
    return tracking_info_;
  }

  /**
   * @brief Gets the total number of static features observed at this frame.
   * Includes usables (inliers) and outliers.
   *
   * @return size_t
   */
  inline size_t numStaticFeatures() const { return static_features_.size(); }

  // NOTE: this is not the number of tracked features, just the ones that are
  // outliers and therefore include all the newly detected ones!!
  /**
   * @brief Gets the total number of static inlier features
   *
   * @return size_t
   */
  inline size_t numStaticUsableFeatures() {
    auto iter = usableStaticFeaturesBegin();
    // we have defined the std::distance operator for this iterator type using
    // std::iterator_traits.
    return static_cast<size_t>(std::distance(iter.begin(), iter.end()));
  }

  /**
   * @brief Gets the total number of dynamic features observed at this frame.
   * Includes usables (inliers) and outliers.
   *
   * @return size_t
   */
  inline size_t numDynamicFeatures() const { return dynamic_features_.size(); }

  /**
   * @brief Gets the total number of dynamic inlier features
   *
   * @return size_t
   */
  inline size_t numDynamicUsableFeatures() {
    auto iter = usableDynamicFeaturesBegin();
    return static_cast<size_t>(std::distance(iter.begin(), iter.end()));
  }

  /**
   * @brief Gets the total number of features (static + dynamic) in this frame.
   * Includes usables (inliers) and outliers.
   *
   * @return size_t
   */
  inline size_t numTotalFeatures() const {
    return numStaticFeatures() + numDynamicFeatures();
  }

  /**
   * @brief Checks if a feature exists within this frame. Could be static or
   * dynamic.
   *
   * @param tracklet_id TrackletId
   * @return true
   * @return false
   */
  bool exists(TrackletId tracklet_id) const;

  /**
   * @brief Gets a feature from its tracklet ID. Could be static or dynamic.
   * Nullptr is returned if the feature does not exist.
   *
   * @param tracklet_id TrackletId
   * @return Feature::Ptr
   */
  Feature::Ptr at(TrackletId tracklet_id) const;

  /**
   * @brief Returns true if the feature is usable (inlier).
   * Throws std::runtime_error if the feature does not exist. Use in conjunction
   * with Frame::exists.
   *
   * @param tracklet_id TrackletId
   * @return true
   * @return false
   */
  bool isFeatureUsable(TrackletId tracklet_id) const;

  /**
   * @brief Collects a vector of feature pointers given a vector of tracklet
   * ids. Features returned could be inliers or outliers. If a feature does not
   * exist std::runtime_error
   *
   * @param tracklet_ids TrackletIds
   * @return FeaturePtrs
   */
  FeaturePtrs collectFeatures(TrackletIds tracklet_ids) const;

  /**
   * @brief Project the feature into the camera frame and returns the 3D
   * landmark. Currently requires the feature to have depth associated with it
   *
   * @param tracklet_id TrackletId
   * @return Landmark
   */
  Landmark backProjectToCamera(TrackletId tracklet_id) const;

  /**
   * @brief Project the feature into the world frame using the frames pose
   * (T_world_camera_) and returns the 3D landmark. Currently requires the
   * feature to have depth associated with it
   *
   * @param tracklet_id TrackletId
   * @return Landmark
   */
  Landmark backProjectToWorld(TrackletId tracklet_id) const;

  /**
   * @brief Constructs a GTSAM camera for this frame using the current pose of
   * this camera (T_world_camera)
   *
   * @return Camera::CameraImpl
   */
  Camera::CameraImpl getFrameCamera() const;

  inline ObjectIds getObjectIds() const {
    ObjectIds object_ids;
    for (const auto& kv : object_observations_) {
      object_ids.push_back(kv.first);
    }
    return object_ids;
  }

  PointCloudLabelRGB::Ptr projectToDenseCloud(
      const cv::Mat* detection_mask = nullptr) const;

  /**
   * @brief Update the depth values on all contained features.
   * Clips the static and dynamic features (marking them invalid) accordingly.
   *
   * Returns false if the internal image container does not have a depth mat.
   *
   * @return true
   * @return false
   */
  bool updateDepths();

  Frame& setMaxBackgroundDepth(double thresh);
  Frame& setMaxObjectDepth(double thresh);

  // // TODO: this really needs testing
  // void moveObjectToStatic(ObjectId instance_label);
  // // TODO: testing
  // // also updates all the tracking_labels of the features associated with
  // this
  // // object
  // void updateObjectTrackingLabel(const DynamicObjectObservation& observation,
  //                                ObjectId new_tracking_label);

  /**
   * @brief Gets the set of tracked pairs between this frame and the previous
   * frame. All correspondences will be usable (inliers). The correspondences
   * will be in the form of feature pairs where first = feature in the previous
   * frame and second = feature in this frame.
   *
   * True is returned if the number of correspondences > 0
   *
   * @param correspondences FeaturePairs& vector of feature pairs
   * @param previous_frame const Frame&
   * @param kp_type KeyPointType
   * @return true
   * @return false
   */
  bool getCorrespondences(FeaturePairs& correspondences,
                          const Frame& previous_frame,
                          KeyPointType kp_type) const;

  /**
   * @brief Gets the set of tracked pairs between this frame and the previous
   * frame for a tracked dynamic object seen in both frames.
   *
   * All correspondences will be usable (inliers).
   * The correspondences will be in the form of feature pairs where first =
   * feature in the previous frame and second = feature in this frame.
   *
   * The object id is checked via the this frames Frame::object_observations_
   * map and is currently implemented as the object instance label (not tracking
   * label).
   *
   * True is returned if the number of correspondences > 0.
   *
   * @param correspondences FeaturePairs& vector of feature pairs
   * @param previous_frame const Frame&
   * @param object_id ObjectId
   * @return true
   * @return false
   */
  bool getDynamicCorrespondences(FeaturePairs& correspondences,
                                 const Frame& previous_frame,
                                 ObjectId object_id) const;

  /**
   * @brief Functional alias that constructs a TrackletCorrespondance from a
   * pair of features (tracked features in the previous and current frame
   * respectively) and the previous frame. The function arguments are in the
   * form previous frame, previous feature, current feature.
   *
   * We functionalise this operation so that we can define any type of
   * TrackletCorrespondance for a set of tracked features via
   * Frame::getCorrespondences which takes a ConstructCorrespondanceFunc as an
   * argument.
   *
   * @tparam RefType Type to be used as the reference type in the
   * TrackletCorrespondance
   * @tparam CurType Type to be used as the current type in the
   * TrackletCorrespondance
   */
  template <typename RefType, typename CurType>
  using ConstructCorrespondanceFunc =
      std::function<TrackletCorrespondance<RefType, CurType>(
          const Frame&, const Feature::Ptr&, const Feature::Ptr&)>;

  /**
   * @brief Helper function that creates a ConstructCorrespondanceFunc which
   * operates to return a TrackletCorrespondance where the ref type is a
   * Landmark in the world frame and the cur type is a Keypoint in the current
   * frame.
   *
   * This is useful for 3D-2D correspondence types.
   *
   * @return ConstructCorrespondanceFunc<Landmark, Keypoint>
   */
  ConstructCorrespondanceFunc<Landmark, Keypoint>
  landmarkWorldKeypointCorrespondance() const;

  /**
   * @brief Helper function that creates a ConstructCorrespondanceFunc which
   * operates to return a TrackletCorrespondance where both types are keypoints
   * (in the ref and cur frame respectively).
   *
   *  This is useful for 2D-2D correspondence types.
   *
   * @return ConstructCorrespondanceFunc<Keypoint, Keypoint>
   */
  ConstructCorrespondanceFunc<Keypoint, Keypoint> imageKeypointCorrespondance()
      const;

  /**
   * @brief Helper function that creates a ConstructCorrespondanceFunc which
   * operates to return a TrackletCorrespondance where the ref type is a
   * Landmark in the world frame and the curr type is a normalized bearing
   * vector constructed from the corresponding Keypoint in the current frame.
   *
   * This is useful for 3D-2D correspondence types for the motion solver which
   * expected correctly projected bearing vectors.
   *
   * @return ConstructCorrespondanceFunc<Landmark, gtsam::Vector3>
   */
  ConstructCorrespondanceFunc<Landmark, gtsam::Vector3>
  landmarkWorldProjectedBearingCorrespondance() const;

  /**
   * @brief Helper function ConstructCorrespondanceFunc which operates to return
   * a TrackletCorrespondance where both types are Landmarks in the world frame
   * (in the ref and cur frame respectively).
   *
   *  This is useful for 3D-3D correspondence types.
   *
   * @return ConstructCorrespondanceFunc<Landmark, Landmark>
   */
  ConstructCorrespondanceFunc<Landmark, Landmark>
  landmarkWorldPointCloudCorrespondance() const;

  template <typename RefType, typename CurType>
  bool getCorrespondences(
      GenericCorrespondences<RefType, CurType>& correspondences,
      const Frame& previous_frame, KeyPointType kp_type,
      const ConstructCorrespondanceFunc<RefType, CurType>& func) const;

  template <typename RefType, typename CurType>
  bool getDynamicCorrespondences(
      GenericCorrespondences<RefType, CurType>& correspondences,
      const Frame& previous_frame, ObjectId object_id,
      const ConstructCorrespondanceFunc<RefType, CurType>& func) const;

  // special iterator types
  FeatureFilterIterator usableStaticFeaturesBegin();
  FeatureFilterIterator usableStaticFeaturesBegin() const;

  FeatureFilterIterator usableDynamicFeaturesBegin();
  FeatureFilterIterator usableDynamicFeaturesBegin() const;

 protected:
  // these do not do distortion or projection along the ray
  bool getStaticCorrespondences(FeaturePairs& correspondences,
                                const Frame& previous_frame) const;
  bool getDynamicCorrespondences(FeaturePairs& correspondences,
                                 const Frame& previous_frame) const;

 private:
  static void updateDepthsFeatureContainer(
      FeatureContainer& container, const ImageWrapper<ImageType::Depth>& depth,
      double max_depth);

  // based on the current set of dynamic features
  //  populates object_observations_
  void constructDynamicObservations();

  Landmark getLandmarkFromCache(LandmarkMap& cache, Feature::Ptr feature,
                                const gtsam::Pose3& X_world) const;

 private:
  UndistorterRectifier::Ptr undistorter_;

  //! cached projections of each landmark depending on the frame to save speed
  //! Based off the initial 2d observation and depth from the feature
  mutable LandmarkMap landmark_in_camera_cache_;
  mutable LandmarkMap landmark_in_world_cache_;

  //! Background points greater than this depth will be discarded
  double max_background_threshold_ = 40.0;
  // ! Object points greater than this depth will be discarded
  double max_object_threshold_ = 25.0;
};

}  // namespace dyno

#include "dynosam/frontend/vision/Frame-inl.hpp"
