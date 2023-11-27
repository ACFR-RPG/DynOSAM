/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"

#include "dynosam/utils/TimingStats.hpp"

namespace dyno {

ObjectId Frame::global_object_id{1};

Frame::Frame(
        FrameId frame_id,
        Timestamp timestamp,
        Camera::Ptr camera,
        const TrackingInputImages& tracking_images,
        const FeatureContainer& static_features,
        const FeatureContainer& dynamic_features)
        :   frame_id_(frame_id),
            timestamp_(timestamp),
            camera_(camera),
            tracking_images_(tracking_images),
            static_features_(static_features),
            dynamic_features_(dynamic_features)
        {
            constructDynamicObservations();

            // NOTE: no rectification, use camera matrix as P for cv::undistortPoints
            // see https://stackoverflow.com/questions/22027419/bad-results-when-undistorting-points-using-opencv-in-python
            //i mean, this could just be shared between frames?
            const CameraParams& cam_params = camera->getParams();
            cv::Mat P = cam_params.getCameraMatrix();
            cv::Mat R = cv::Mat::eye(3,3,CV_32FC1);
            undistorter_ = std::make_shared<UndistorterRectifier>(P, cam_params, R);
        }

bool Frame::exists(TrackletId tracklet_id) const {
    const bool result = static_features_.exists(tracklet_id) || dynamic_features_.exists(tracklet_id);

    //debug checking -> should only be in one feature container
    if(result) {
        CHECK(!(static_features_.exists(tracklet_id) &&  dynamic_features_.exists(tracklet_id)))
            << "Tracklet Id " <<  tracklet_id << " exists in both static and dynamic feature sets. Should be unique!";
    }
    return result;
}

Feature::Ptr Frame::at(TrackletId tracklet_id) const {
    if(!exists(tracklet_id)) {
        return nullptr;
    }

    if(static_features_.exists(tracklet_id)) {
        CHECK(!dynamic_features_.exists(tracklet_id));
        return static_features_.getByTrackletId(tracklet_id);
    }
    else {
        CHECK(dynamic_features_.exists(tracklet_id));
        return dynamic_features_.getByTrackletId(tracklet_id);
    }
}


bool Frame::isFeatureUsable(TrackletId tracklet_id) const {
    const auto& feature = at(tracklet_id);
    if(!feature) {
        throw std::runtime_error("Failed to check feature usability - tracklet id " + std::to_string(tracklet_id) + " does not exist");
    }

    return feature->usable();
}


FeaturePtrs Frame::collectFeatures(TrackletIds tracklet_ids) const {
    FeaturePtrs features;
    for(const auto tracklet_id : tracklet_ids) {
        Feature::Ptr feature = at(tracklet_id);
        if(!feature) {
            throw std::runtime_error("Failed to collectFeatures - tracklet id " + std::to_string(tracklet_id) + " does not exist");
        }

        features.push_back(feature);

    }

    return features;
}


Landmark Frame::backProjectToCamera(TrackletId tracklet_id) const {
    Feature::Ptr feature = at(tracklet_id);
    if(!feature) {
        throw std::runtime_error("Failed to backProjectToCamera - tracklet id " + std::to_string(tracklet_id) + " does not exist");
    }

    //if no depth, project to unitsphere?
    CHECK(feature->hasDepth());

    // Landmark lmk;
    return getLandmarkFromCache(landmark_in_camera_cache_, feature, gtsam::Pose3::Identity());
    // return lmk;
}

Landmark Frame::backProjectToWorld(TrackletId tracklet_id) const {
    Feature::Ptr feature = at(tracklet_id);
    if(!feature) {
        throw std::runtime_error("Failed to backProjectToWorld - tracklet id " + std::to_string(tracklet_id) + " does not exist");
    }

    //if no depth, project to unitsphere?
    CHECK(feature->hasDepth());

    // Landmark lmk;
    // camera_->backProject(feature->keypoint_, feature->depth_, &lmk, T_world_camera_);
    return getLandmarkFromCache(landmark_in_world_cache_, feature, T_world_camera_);
}


Camera::CameraImpl Frame::getFrameCamera() const {
    const CameraParams& camera_params = camera_->getParams();
    return Camera::CameraImpl(T_world_camera_, camera_params.constructGtsamCalibration<Camera::CalibrationType>());
}

void Frame::updateDepths(const ImageWrapper<ImageType::Depth>& depth, double max_static_depth, double max_dynamic_depth) {
    updateDepthsFeatureContainer(static_features_, depth, max_static_depth);
    updateDepthsFeatureContainer(dynamic_features_, depth, max_dynamic_depth);
}



bool Frame::getCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, KeyPointType kp_type) const {
    if(kp_type == KeyPointType::STATIC) {
        return getStaticCorrespondences(correspondences, previous_frame);
    }
    else {
        return getDynamicCorrespondences(correspondences, previous_frame);
    }
}

Frame::ConstructCorrespondanceFunc<Landmark, Keypoint> Frame::landmarkWorldKeypointCorrespondance() const {
    auto func = [&](const Frame& previous_frame, const Feature::Ptr& previous_feature, const Feature::Ptr& current_feature) {
        if(!previous_feature->hasDepth()) {
            throw std::runtime_error("Error in constructing Landmark (w) -> keypoint correspondences - previous feature does not have depth!");
        }

        //eventuall map?
        Landmark lmk_w = previous_frame.backProjectToWorld(previous_feature->tracklet_id_);
        return TrackletCorrespondance(previous_feature->tracklet_id_, lmk_w, current_feature->keypoint_);
    };

    return std::bind(func, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
}


Frame::ConstructCorrespondanceFunc<Keypoint, Keypoint> Frame::imageKeypointCorrespondance() const {
    auto func = [&](const Frame&, const Feature::Ptr& previous_feature, const Feature::Ptr& current_feature) {
        return TrackletCorrespondance(previous_feature->tracklet_id_, previous_feature->keypoint_, current_feature->keypoint_);
    };

    return std::bind(func, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

}

Frame::ConstructCorrespondanceFunc<Landmark, gtsam::Vector3> Frame::landmarkWorldProjectedBearingCorrespondance() const {
    auto func = [&](const Frame& previous_frame, const Feature::Ptr& previous_feature, const Feature::Ptr& current_feature) {
        if(!previous_feature->hasDepth()) {
            throw std::runtime_error("Error in constructing Landmark (w) -> keypoint correspondences - previous feature does not have depth!");
        }

        //eventuall map?
        Landmark lmk_w = previous_frame.backProjectToWorld(previous_feature->tracklet_id_);

        gtsam::Vector3 projected_versor = undistorter_->undistortKeypointAndGetProjectedVersor(current_feature->keypoint_);
        return TrackletCorrespondance(previous_feature->tracklet_id_, lmk_w, projected_versor);
    };

    return std::bind(func, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
}


bool Frame::getDynamicCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, ObjectId object_id) const {

    if(object_observations_.find(object_id) == object_observations_.end()) {
        LOG(WARNING) << "Object object instance id " << object_id << " not found for frame " << frame_id_;
        return false;
    }

    const DynamicObjectObservation& observation = object_observations_.at(object_id);
    // TODO: need to put back on - if we have motion mask, we should just mark all objects as moving CHECK(observation.marked_as_moving_);
    const TrackletIds& tracklets = observation.object_features_;

    FeatureContainer feature_container;
    for(const TrackletId tracklet : tracklets) {
        if(isFeatureUsable(tracklet)) {
            feature_container.add(this->at(tracklet));
        }
    }

    //make iterator for the previous dynamic features that ensure each feature is usable and has a matching instance label
    auto previous_dynamic_features_iterator = FeatureFilterIterator(
        const_cast<FeatureContainer&>(previous_frame.dynamic_features_),
        [object_id](const Feature::Ptr& f) -> bool {
            return Feature::IsUsable(f) && f->instance_label_ == object_id;
        });

    //get the correspondences from these two iterators
    //we iterate over the current feature container which should only contain features on the object
    //and compare against the container
    vision_tools::getCorrespondences(
        correspondences,
        previous_dynamic_features_iterator,
        //we iterate over the current feature container which should only contain features on the object
        feature_container.beginUsable()
    );

    LOG(INFO) << "Found " << correspondences.size() << " correspondences for object instance " << object_id << " " << (correspondences.size() > 0u);

    return correspondences.size() > 0u;
}


bool Frame::getStaticCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame) const {
    vision_tools::getCorrespondences(
        correspondences,
        previous_frame.static_features_.beginUsable(),
        static_features_.beginUsable()
    );

    return correspondences.size() > 0u;
}

bool Frame::getDynamicCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame) const {
    vision_tools::getCorrespondences(
        correspondences,
        previous_frame.dynamic_features_.beginUsable(),
        dynamic_features_.beginUsable()
    );
    return correspondences.size() > 0u;
}


void Frame::updateDepthsFeatureContainer(FeatureContainer& container, const ImageWrapper<ImageType::Depth>& depth, double max_depth) {
    const cv::Mat& depth_mat = depth;
    auto iter = container.beginUsable();

    for(Feature::Ptr feature : iter) {
        CHECK(feature->usable());
        // const Feature::Ptr& feature = *iter;
        const int x = functional_keypoint::u(feature->keypoint_);
        const int y = functional_keypoint::v(feature->keypoint_);
        const Depth d = depth_mat.at<Depth>(y, x);

        if(d > max_depth || d <= 0) {
            feature->markInvalid();
        }

         //if now invalid or happens to be invalid from a previous frame, make depth invalid too
        if(!feature->usable()) {
            feature->depth_ = Feature::invalid_depth;
        }
        else {
            feature->depth_ = d;
        }
    }

}


void Frame::constructDynamicObservations() {
    // CHECK_GT(dynamic_features_.size(), 0u);
    object_observations_.clear();
    // utils::TimingStatsCollector construct_obs_timer("construct_dynamic_obs_timer");

    const ObjectIds instance_labels = vision_tools::getObjectLabels(tracking_images_.get<ImageType::MotionMask>());

    auto inlier_iterator = dynamic_features_.beginUsable();
    for(const Feature::Ptr& dynamic_feature : inlier_iterator) {
        CHECK(!dynamic_feature->isStatic());
        CHECK(dynamic_feature->usable());

        const ObjectId instance_label = dynamic_feature->instance_label_;
        //this check is just for sanity!
        CHECK(std::find(instance_labels.begin(), instance_labels.end(), instance_label) != instance_labels.end());

        if(object_observations_.find(instance_label) == object_observations_.end()) {
            DynamicObjectObservation observation;
            observation.tracking_label_ = -1;
            observation.instance_label_ = instance_label;
            object_observations_[instance_label] = observation;
        }

        object_observations_[instance_label].object_features_.push_back(dynamic_feature->tracklet_id_);
    }
}

void Frame::moveObjectToStatic(ObjectId instance_label) {
    auto it = object_observations_.find(instance_label);
    CHECK(it != object_observations_.end());


    DynamicObjectObservation& observation = it->second;
    observation.marked_as_moving_ = false;
    CHECK(observation.instance_label_ == instance_label);
    //go through all features, move them to from dynamic structure and add them to static
    for(TrackletId tracklet_id : observation.object_features_) {
        CHECK(dynamic_features_.exists(tracklet_id));
        Feature::Ptr dynamic_feature = dynamic_features_.getByTrackletId(tracklet_id);

        if(!dynamic_feature->usable()) {continue;}

        CHECK(!dynamic_feature->isStatic());
        CHECK_EQ(dynamic_feature->tracklet_id_, tracklet_id);
        CHECK_EQ(dynamic_feature->instance_label_, instance_label);
        dynamic_feature->type_ = KeyPointType::STATIC;
        dynamic_feature->instance_label_ = background_label;
        dynamic_feature->tracking_label_ = background_label;

        dynamic_features_.remove(tracklet_id);
        //Jesse: no, do not move points (these are dense) to static - instrad we need to mark the AREA
        //around the object as static and then retrack all points in there!!
        // static_features_.add(dynamic_feature);
    }

    object_observations_.erase(it);

}

void Frame::updateObjectTrackingLabel(const DynamicObjectObservation& observation, ObjectId new_tracking_label) {
    auto it = object_observations_.find(observation.instance_label_);
    CHECK(it != object_observations_.end());

    auto& obs = it->second;
    obs.tracking_label_ = new_tracking_label;
    //update all features
    for(TrackletId tracklet_id : obs.object_features_) {
        Feature::Ptr feature = dynamic_features_.getByTrackletId(tracklet_id);
        CHECK(feature);
        feature->tracking_label_ = new_tracking_label;
    }
}


FeatureFilterIterator Frame::usableStaticFeaturesBegin() {
    return static_features_.beginUsable();
}

FeatureFilterIterator Frame::usableStaticFeaturesBegin() const {
    return static_features_.beginUsable();
}

FeatureFilterIterator Frame::usableDynamicFeaturesBegin() {
    return dynamic_features_.beginUsable();
}

FeatureFilterIterator Frame::usableDynamicFeaturesBegin() const {
    return dynamic_features_.beginUsable();
}



Landmark Frame::getLandmarkFromCache(LandmarkMap& cache, Feature::Ptr feature, const gtsam::Pose3& X_world) const {
    const auto& it = cache.find(feature->tracklet_id_);
    if(it != cache.end()) {
        return it->second;
    }

    Landmark lmk;
    camera_->backProject(feature->keypoint_, feature->depth_, &lmk, X_world);
    cache.insert({feature->tracklet_id_, lmk});
    return lmk;
}

// Frame::FeatureFilterIterator Frame::dynamicUsableBegin() {
//     return FeatureFilterIterator(dynamic_features_, [&](const Feature::Ptr& f) -> bool
//         {
//             return f->usable();
//         }
//     );
// }

} //dyno
