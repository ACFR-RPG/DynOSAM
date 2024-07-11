/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#include "dynosam/backend/RGBDBackendModule.hpp"
#include "dynosam/common/Flags.hpp"

#include "dynosam/common/PointCloudProcess.hpp"

#include "dynosam/factors/ObjectCentricMotionFactor.hpp"

#include <glog/logging.h>
#include <gflags/gflags.h>

namespace dyno {

StateQuery<gtsam::Pose3> RGBDBackendModule::ObjectCentricAccessor::getObjectMotion(FrameId frame_id, ObjectId object_id) const {
    const auto frame_node_k = getMap()->getFrame(frame_id);
    const auto frame_node_k_1 = getMap()->getFrame(frame_id - 1u);
    if (frame_node_k && frame_node_k_1) {
        const auto motion_key = frame_node_k->makeObjectMotionKey(object_id);
        StateQuery<gtsam::Pose3> motion_s0_k = this->query<gtsam::Pose3>(motion_key);

        StateQuery<gtsam::Pose3> motion_s0_k_1 = this->query<gtsam::Pose3>(
            frame_node_k_1->makeObjectMotionKey(object_id)
        );

        if (motion_s0_k && motion_s0_k_1) {
            //want a motion from k-1 to k, but we estimate s0 to k
            //^w_{k-1}H_k = ^w_{s0}H_k \: ^w_{s0}H_{k-1}^{-1}
            return StateQuery<gtsam::Pose3>(
                motion_key,
                motion_s0_k.get() * motion_s0_k_1.get().inverse()
            );
        }
        else {
            return StateQuery<gtsam::Pose3>::NotInMap(frame_node_k->makeObjectMotionKey(object_id));
        }
    }
    return StateQuery<gtsam::Pose3>::InvalidMap();
}

StateQuery<gtsam::Pose3> RGBDBackendModule::ObjectCentricAccessor::getObjectPose(FrameId frame_id, ObjectId object_id) const {
    //we estimate a motion ^w_{s0}H_k, so we can compute a pose ^wL_k = ^w_{s0}H_k * ^wL_{s0}
    const auto frame_node_k = getMap()->getFrame(frame_id);
    gtsam::Key motion_key = frame_node_k->makeObjectMotionKey(object_id);
    gtsam::Key pose_key = frame_node_k->makeObjectPoseKey(object_id);
    CHECK(frame_node_k);
    ///hmmm... if we do a query after we do an update but before an optimise then the motion will
    //be whatever we initalised it with
    //in the case of identity, the pose at k will just be L_s0 which we dont want?
    StateQuery<gtsam::Pose3> motion_s0_k = this->query<gtsam::Pose3>(motion_key);
    CHECK(false);

    if(motion_s0_k) {
        CHECK(L0_values_->exists(object_id));
        const gtsam::Pose3& L0 = L0_values_->at(object_id).first;

        //from Chirikjian equ. 32
        // gtsam::Rot3 pcg_R = motion_s0_k->rotation();
        // gtsam::Rot3 L_R = pcg_R * L0.rotation() * pcg_R.inverse();
        // gtsam::Point3 L_t = pcg_R * L0.translation();
        const gtsam::Pose3 L_k = motion_s0_k.get() * L0;
        // const gtsam::Pose3 L_k(L_R, L_t);

        return StateQuery<gtsam::Pose3>(
            pose_key,
            L_k
        );
    }
    else {
        return StateQuery<gtsam::Pose3>::NotInMap(pose_key);
    }

}

StateQuery<gtsam::Point3> RGBDBackendModule::ObjectCentricAccessor::getDynamicLandmark(FrameId frame_id, TrackletId tracklet_id) const {
    //we estimate a motion ^w_{s0}H_k, so we can compute a point ^wm_k = ^w_{s0}H_k * ^wL_{s0} * ^{L_{s0}}m
    const auto frame_node_k = getMap()->getFrame(frame_id);
    const auto lmk_node = getMap()->getLandmark(tracklet_id);
    CHECK(frame_node_k);
    CHECK_NOTNULL(lmk_node);
    const auto object_id = lmk_node->object_id;
    //point in L_{s0}
    //NOTE: we use STATIC point key here
    gtsam::Key point_key = this->makeDynamicKey(tracklet_id);
    StateQuery<gtsam::Point3> point_local = this->query<gtsam::Point3>(point_key);

    //get motion from S0 to k
    gtsam::Key motion_key = frame_node_k->makeObjectMotionKey(object_id);
    StateQuery<gtsam::Pose3> motion_s0_k = this->query<gtsam::Pose3>(motion_key);

    if(point_local) CHECK(motion_s0_k) << "We have a point " << DynoLikeKeyFormatter(point_key) << " but no motion at frame " << frame_id;
    if(point_local && motion_s0_k) {

        CHECK(L0_values_->exists(object_id));
        const gtsam::Pose3& L0 = L0_values_->at(object_id).first;
        //point in world at k
        const gtsam::Point3 m_k = motion_s0_k.get() * L0 * point_local.get();
        return StateQuery<gtsam::Point3>(
            point_key,
            m_k
        );
    }
    else {
        return StateQuery<gtsam::Point3>::NotInMap(point_key);
    }

}

StatusLandmarkEstimates RGBDBackendModule::ObjectCentricAccessor::getDynamicLandmarkEstimates(FrameId frame_id, ObjectId object_id) const {
    const auto frame_node = getMap()->getFrame(frame_id);
    const auto object_node = getMap()->getObject(object_id);
    CHECK_NOTNULL(frame_node);
    CHECK_NOTNULL(object_node);

    if(!frame_node->objectObserved(object_id)) {
        return StatusLandmarkEstimates{};
    }

    StatusLandmarkEstimates estimates;
    //unlike in the base version, iterate over all points on the object (i.e all tracklets)
    //as we can propogate all of them!!!!
    const auto& dynamic_landmarks = object_node->dynamic_landmarks;
    for(auto lmk_node : dynamic_landmarks) {
        const auto tracklet_id = lmk_node->tracklet_id;

        CHECK_EQ(object_id, lmk_node->object_id);

        //user defined function should put point in the world frame
        StateQuery<gtsam::Point3> lmk_query = this->getDynamicLandmark(
            frame_id,
            tracklet_id
        );
        if(lmk_query) {
            estimates.push_back(
                LandmarkStatus::DynamicInGLobal(
                    lmk_query.get(), //estimate
                    frame_id,
                    tracklet_id,
                    object_id,
                    LandmarkStatus::Method::OPTIMIZED //this may not be correct!!
                ) //status
            );
        }
    }
    return estimates;


}



void RGBDBackendModule::ObjectCentricUpdater::dynamicPointUpdateCallback(const PointUpdateContext& context, UpdateObservationResult& result, gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors) {

    const auto lmk_node = context.lmk_node;
    const auto frame_node_k_1 = context.frame_node_k_1;
    const auto frame_node_k = context.frame_node_k;

    auto dynamic_point_noise = parent_->dynamic_point_noise_;
    Accessor::Ptr theta_accessor = this->accessorFromTheta();

    gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

    const gtsam::Key object_motion_key_k = frame_node_k->makeObjectMotionKey(context.getObjectId());
    const gtsam::Key object_motion_key_k_1 = frame_node_k_1->makeObjectMotionKey(context.getObjectId());



    gtsam::Pose3 L_0;
    FrameId s0;
    std::tie(L_0, s0) = getL0OrInitalise(context.getObjectId());
    auto landmark_motion_noise = parent_->landmark_motion_noise_;

    //check that the first frame id is at least the initial frame for s0
    CHECK_GE(frame_node_k_1->getId(), s0);

    if(!isDynamicTrackletInMap(lmk_node)) {
        //TODO: this will not hold in the batch case as the first dynamic point we get will not be the first point
        //on the object (we will get the first point seen within the window)
        //so, where should be initalise the object pose!?
        // //this is a totally new tracklet so should be the first time we've seen it!
        // CHECK_EQ(lmk_node->getFirstSeenFrame(), frame_node_k_1->getId());

        //mark as now in map
        is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), true);
        CHECK(isDynamicTrackletInMap(lmk_node));

        //use first point as initalisation?
        //in this case k is k-1 as we use frame_node_k_1
        gtsam::Pose3 s0_H_k = computeInitialHFromFrontend(frame_node_k_1->getId(), context.getObjectId());
        //measured point in camera frame
        const gtsam::Point3 m_camera = lmk_node->getMeasurement(frame_node_k_1).landmark;
        Landmark lmk_L0_init = L_0.inverse() * s0_H_k.inverse() * context.X_k_1_measured * m_camera;

        //initalise value //cannot initalise again the same -> it depends where L_0 is created, no?
        Landmark lmk_L0;
        getSafeQuery(
            lmk_L0,
            theta_accessor->query<Landmark>(point_key),
            lmk_L0_init
        );
        new_values.insert(point_key, lmk_L0);
        result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());
    }

    if(context.is_starting_motion_frame) {
        //add factor at k-1
        new_factors.emplace_shared<ObjectCentricMotionFactor>(
            frame_node_k_1->makePoseKey(), //pose key at previous frames,
            object_motion_key_k_1,
            point_key,
            lmk_node->getMeasurement(frame_node_k_1).landmark,
            L_0,
            landmark_motion_noise
        );
        result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());

    }

    // add factor at k
    new_factors.emplace_shared<ObjectCentricMotionFactor>(
        frame_node_k->makePoseKey(), //pose key at previous frames,
        object_motion_key_k,
        point_key,
        lmk_node->getMeasurement(frame_node_k).landmark,
        L_0,
        landmark_motion_noise
    );
    result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());



}
void RGBDBackendModule::ObjectCentricUpdater::objectUpdateContext(const ObjectUpdateContext& context, UpdateObservationResult& result, gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors) {

    auto frame_node_k = context.frame_node_k;
    const gtsam::Key object_motion_key_k = frame_node_k->makeObjectMotionKey(context.getObjectId());

    Accessor::Ptr theta_accessor = this->accessorFromTheta();
    const auto frame_id = context.getFrameId();
    const auto object_id = context.getObjectId();

    if(!is_other_values_in_map.exists(object_motion_key_k)) {
        // gtsam::Pose3 motion;
        gtsam::Pose3 motion = computeInitialHFromFrontend(context.getFrameId(), context.getObjectId());
        new_values.insert(object_motion_key_k, motion);
        is_other_values_in_map.insert2(object_motion_key_k, true);
    }

    if(frame_id < 2) return;

    auto frame_node_k_1 = getMap()->getFrame(frame_id - 1u);
    if (!frame_node_k_1) { return; }

    if(FLAGS_use_smoothing_factor && frame_node_k_1->objectObserved(object_id)) {
        //motion key at previous frame
        const gtsam::Symbol object_motion_key_k_1 = frame_node_k_1->makeObjectMotionKey(object_id);

        auto object_smoothing_noise = parent_->object_smoothing_noise_;
        CHECK(object_smoothing_noise);
        CHECK_EQ(object_smoothing_noise->dim(), 6u);

        {
            ObjectId object_label_k_1, object_label_k;
            FrameId frame_id_k_1, frame_id_k;
            CHECK(reconstructMotionInfo(object_motion_key_k_1, object_label_k_1, frame_id_k_1));
            CHECK(reconstructMotionInfo(object_motion_key_k, object_label_k, frame_id_k));
            CHECK_EQ(object_label_k_1, object_label_k);
            CHECK_EQ(frame_id_k_1 + 1, frame_id_k); //assumes consequative frames
        }

        //if the motion key at k (motion from k-1 to k), and key at k-1 (motion from k-2 to k-1)
        //exists in the map or is about to exist via new values, add the smoothing factor
        if(is_other_values_in_map.exists(object_motion_key_k_1) && is_other_values_in_map.exists(object_motion_key_k)) {
            // new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            //     object_motion_key_k_1,
            //     object_motion_key_k,
            //     gtsam::Pose3::Identity(),
            //     object_smoothing_noise
            // );
            if(result.debug_info) result.debug_info->getObjectInfo(context.getObjectId()).smoothing_factor_added = true;

        }

        // if(smoothing_added) {
        //     //TODO: add back in
        //     // object_debug_info.smoothing_factor_added = true;
        // }

    }


}

 std::pair<gtsam::Pose3, FrameId> RGBDBackendModule::ObjectCentricUpdater::getL0OrInitalise(ObjectId object_id) {
    if(!L0_values_.exists(object_id)) {
        //init object centroid at first frame!!
        auto object_node = getMap()->getObject(object_id);
        CHECK(object_node);

        FrameId first_seen_frame = object_node->getFirstSeenFrame();
        auto frame_node = getMap()->getFrame(first_seen_frame);
        CHECK(frame_node);

        auto gt_packet_map = parent_->getGroundTruthPackets();
        if(gt_packet_map->exists(first_seen_frame)) {
            const GroundTruthInputPacket& gt_packet_k_1 = gt_packet_map->at(first_seen_frame);

            ObjectPoseGT object_pose_gt_k_1;
            if(!gt_packet_k_1.getObject(object_id, object_pose_gt_k_1)) {
                auto pose_k_1 = object_pose_gt_k_1.L_world_;
                L0_values_.insert2(object_id, std::make_pair(pose_k_1, first_seen_frame));
                return L0_values_.at(object_id);
            }
        }

        StatusLandmarkEstimates dynamic_landmarks;

        //measured/linearized camera pose at the first frame this object has been seen
        const gtsam::Pose3 X_world = getInitialOrLinearizedSensorPose(first_seen_frame);

        auto measurement_pairs = frame_node->getDynamicMeasurements(object_id);
        for(const auto&[lmk_node, measurement] : measurement_pairs) {
            CHECK(lmk_node->seenAtFrame(first_seen_frame));
            CHECK_EQ(lmk_node->object_id, object_id);

            const gtsam::Point3 landmark_measurement_local = measurement.landmark;
            const gtsam::Point3 landmark_measurement_world = X_world * landmark_measurement_local;

            dynamic_landmarks.push_back(LandmarkStatus::DynamicInGLobal(
                landmark_measurement_local,
                first_seen_frame,
                lmk_node->tracklet_id,
                object_id,
                LandmarkStatus::Method::MEASURED
            ));
        }

        //keep in local (we say its dynamic in global so the X_world is not used) TODO: clean up code and clarify!
        CloudPerObject object_clouds = groupObjectCloud(dynamic_landmarks, X_world);
        if(object_clouds.size() == 0) {
            //TODO: why does this happen so much!!!
            LOG(INFO) << "Cannot collect object clouds from dynamic landmarks of " << object_id << " and frame " << first_seen_frame << "!! "
                << " # Dynamic lmks in the map for this object at this frame was " << dynamic_landmarks.size(); //<< " but reocrded lmks was " << dynamic_landmarks.size();
            return {gtsam::Pose3{}, 0};
        }
        CHECK_EQ(object_clouds.size(), 1);
        CHECK(object_clouds.exists(object_id));

        const auto dynamic_point_cloud = object_clouds.at(object_id);
        pcl::PointXYZ centroid;
        pcl::computeCentroid(dynamic_point_cloud, centroid);
        //TODO: outlier reject?
        gtsam::Point3 translation = pclPointToGtsam(centroid);
        gtsam::Pose3 center(gtsam::Rot3::Identity(), X_world.transformTo(translation));

        L0_values_.insert2(object_id, std::make_pair(center, first_seen_frame));
        LOG(INFO) << "Initalising L0 for " << object_id << " at frame " << first_seen_frame << " with  " << center;

    }
    return L0_values_.at(object_id);
}

//if we have an optimised motion should use that instead!!!?
gtsam::Pose3 RGBDBackendModule::ObjectCentricUpdater::computeInitialHFromFrontend(FrameId k, ObjectId object_id) {
    gtsam::Pose3 L_0;
    FrameId s0;
    std::tie(L_0, s0) = getL0OrInitalise(object_id);

    CHECK_LE(s0, k);
    if(k == s0) {
        //same frame so motion between them should be identity!
        //except for rotation?
        return gtsam::Pose3::Identity();
    }
    if(k-1 == s0) {
        //a motion that takes us from k-1 to k where k-1 == s0
        Motion3 motion;
        CHECK(parent_->hasFrontendMotionEstimate(k, object_id, &motion));
        return motion;
    }
    else {
        Motion3 composed_motion;

        LOG(INFO) << "Computing initial motion from " << s0 << " to "  << k;

        //query from so+1 to k since we index backwards
        for(auto frame = s0+1; frame <=k; frame++) {

            Motion3 motion; //if fail just use identity?
            if(!parent_->hasFrontendMotionEstimate(frame, object_id, &motion)) {
                LOG(INFO) << "No frontend motion at frame " << frame << " object id " << object_id;
            }

            composed_motion = motion * composed_motion;
        }
        //after loop motion should be ^w_{s0}H_k
        return composed_motion;
    }

}

} //dyno
