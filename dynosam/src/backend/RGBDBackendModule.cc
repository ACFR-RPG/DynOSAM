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
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/utils/TimingStats.hpp"
#include "dynosam/logger/Logger.hpp"

#include "dynosam/common/Flags.hpp"

#include "dynosam/backend/IncrementalOptimizer.hpp"


#include "dynosam/factors/LandmarkQuadricFactor.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/ObjectKinematicFactor.hpp"
#include "dynosam/factors/LandmarkMotionPoseFactor.hpp"
#include "dynosam/factors/LandmarkPoseSmoothingFactor.hpp"

#include <gtsam/base/debug.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <glog/logging.h>
#include <gflags/gflags.h>

DEFINE_int32(opt_window_size,  10, "Sliding window size for optimisation");
DEFINE_int32(opt_window_overlap,  4, "Overlap for window size optimisation");

namespace dyno {

RGBDBackendModule::RGBDBackendModule(const BackendParams& backend_params, Map3d2d::Ptr map, Camera::Ptr camera, const UpdaterType& updater_type, ImageDisplayQueue* display_queue)
    : Base(backend_params, map, display_queue), camera_(CHECK_NOTNULL(camera)), updater_type_(updater_type)
{
    CHECK_NOTNULL(map);
    CHECK_NOTNULL(map_);

    //TODO: functioanlise and streamline with BackendModule
    static_point_noise_ = gtsam::noiseModel::Isotropic::Sigma(3u, backend_params.static_point_noise_sigma_);
    dynamic_point_noise_ = gtsam::noiseModel::Isotropic::Sigma(3u, backend_params.dynamic_point_noise_sigma_);
    //set in base!
    // landmark_motion_noise_ =  gtsam::noiseModel::Isotropic::Sigma(3u, backend_params.motion_ternary_factor_noise_sigma_);

    if(backend_params.use_robust_kernals_) {
        static_point_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(backend_params.k_huber_3d_points_), static_point_noise_);

        dynamic_point_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(backend_params.k_huber_3d_points_), dynamic_point_noise_);

        //TODO: not k_huber_3d_points_ not just used for 3d points
        landmark_motion_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(backend_params.k_huber_3d_points_), landmark_motion_noise_);

        CHECK_NOTNULL(dynamic_pixel_noise_);
        dynamic_pixel_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(backend_params.k_huber_3d_points_), dynamic_pixel_noise_);

    }

    CHECK_NOTNULL(static_point_noise_);
    CHECK_NOTNULL(dynamic_point_noise_);
    CHECK_NOTNULL(landmark_motion_noise_);

    new_updater_ = std::move(makeUpdater());

}

RGBDBackendModule::~RGBDBackendModule() {
    LOG(INFO) << "Destructing RGBDBackendModule";

    if(base_params_.use_logger_) {
        //hack to make sure things are updated!!
        new_updater_->accessorFromTheta()->postUpdateCallback();
        new_updater_->logBackendFromMap();
    }

}

RGBDBackendModule::SpinReturn
RGBDBackendModule::boostrapSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) {

    const FrameId frame_k = input->getFrameId();
    first_frame_id_ = frame_k;
    CHECK_EQ(spin_state_.frame_id, frame_k);
    LOG(INFO) << "Running backend " << frame_k;
    //estimate of pose from the frontend
    const gtsam::Pose3 T_world_cam_k_frontend = input->T_world_camera_;
    initial_camera_poses_.insert2(frame_k, T_world_cam_k_frontend);
    initial_object_motions_.insert2(frame_k, input->estimated_motions_);

    {
        // TIMING_STATS("map.update_observations");
        map_->updateObservations(input->collectStaticMeasurements());
        map_->updateObservations(input->collectDynamicMeasurements());
    }

    gtsam::Values new_values;
    gtsam::NonlinearFactorGraph new_factors;

    new_updater_->setInitialPose(T_world_cam_k_frontend, frame_k, new_values);
    new_updater_->setInitialPosePrior(T_world_cam_k_frontend, frame_k, new_factors);
    //must optimzie to update the map
    // optimizer_->update(spin_state_, new_values, new_factors, map_);
    return {State::Nominal, nullptr};
}

RGBDBackendModule::SpinReturn
RGBDBackendModule::nominalSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) {

    const FrameId frame_k = input->getFrameId();
    CHECK_EQ(spin_state_.frame_id, frame_k);

    LOG(INFO) << "Running backend " << frame_k;
    //estimate of pose from the frontend
    const gtsam::Pose3 T_world_cam_k_frontend = input->T_world_camera_;
    initial_camera_poses_.insert2(frame_k, T_world_cam_k_frontend);
    initial_object_motions_.insert2(frame_k, input->estimated_motions_);
    {
        // TIMING_STATS("map.update_observations");
        map_->updateObservations(input->collectStaticMeasurements());
        map_->updateObservations(input->collectDynamicMeasurements());
    }

    gtsam::Values new_values;
    gtsam::NonlinearFactorGraph new_factors;

    new_updater_->addOdometry(frame_k, T_world_cam_k_frontend, new_values, new_factors);

    UpdateObservationParams update_params;
    update_params.enable_debug_info = true;
    update_params.do_backtrack = true;
    new_updater_->updateStaticObservations(frame_k, new_values, new_factors, update_params);
    new_updater_->updateDynamicObservations(frame_k, new_values, new_factors, update_params);

    //do sliding window like batch from zero
    // if(frame_k % 20 ==0) {
    //     gtsam::Values values;
    //     gtsam::NonlinearFactorGraph graph;
    //     std::tie(values, graph) = constructGraph(first_frame_id_, frame_k, true);
    //     LOG(INFO) << " Finished graph construction";

    //     graph.error(values);
    //     gtsam::LevenbergMarquardtParams opt_params;
    //     if(VLOG_IS_ON(20))
    //         opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;

    //     gtsam::Values optimised_values_sliding = gtsam::LevenbergMarquardtOptimizer(graph, values, opt_params).optimize();
    //     new_updater_->updateTheta(optimised_values_sliding);

    // }


    gtsam::Values optimised_values;
    double error_before = 0;
    double error_after = 0;
    if(buildSlidingWindowOptimisation(frame_k, optimised_values, error_before, error_after)) {
        //update map with best results
        new_updater_->updateTheta(optimised_values);
        LOG(INFO) << "Sliding window error before: " << error_before << " error after: " << error_after;
    }

    new_updater_->accessorFromTheta()->postUpdateCallback(); //force update every time (slow! and just for testing)
    auto accessor = new_updater_->accessorFromTheta();

    auto backend_output = std::make_shared<BackendOutputPacket>();
    backend_output->timestamp_ = input->getTimestamp();
    backend_output->frame_id_ = input->getFrameId();
    backend_output->T_world_camera_ = accessor->getSensorPose(frame_k).get();
    backend_output->static_landmarks_ = accessor->getFullStaticMap();
    backend_output->dynamic_landmarks_ = accessor->getDynamicLandmarkEstimates(frame_k);

    for(FrameId frame_id : map_->getFrameIds()) {
        backend_output->optimized_poses_.push_back(accessor->getSensorPose(frame_id).get());
    }

    backend_output->composed_object_poses = accessor->getObjectPoses();

    debug_info_ = DebugInfo();

    return {State::Nominal, backend_output};
}

std::tuple<gtsam::Values, gtsam::NonlinearFactorGraph>
RGBDBackendModule::constructGraph(FrameId from_frame, FrameId to_frame, bool set_initial_camera_pose_prior) {
    CHECK_LT(from_frame, to_frame);
    gtsam::Values new_values;
    gtsam::NonlinearFactorGraph new_factors;

    auto updater = std::move(makeUpdater());

    UpdateObservationParams update_params;
    update_params.do_backtrack = false;
    update_params.enable_debug_info = true;

    UpdateObservationResult results;

    CHECK_GE(from_frame, map_->firstFrameId());
    CHECK_LE(to_frame, map_->lastFrameId());

    for(auto frame_id = from_frame; frame_id <= to_frame; frame_id++) {
        LOG(INFO) << "Constructing dynamic graph at frame " << frame_id << " in loop (" << from_frame << " -> " << to_frame << ")";

        // pose estimate from frontend
        CHECK(initial_camera_poses_.exists(frame_id));
        //TODO: or latest initalisation point!?
        const gtsam::Pose3& T_world_camera_k = initial_camera_poses_.at(frame_id);

        //if first frame
        if(frame_id == from_frame) {
            //add first pose
            updater->setInitialPose(T_world_camera_k, frame_id, new_values);

            if(set_initial_camera_pose_prior)
                updater->setInitialPosePrior(T_world_camera_k, frame_id, new_factors);
        }
        else {
            updater->addOdometry(frame_id, T_world_camera_k, new_values, new_factors);
            //no backtrack
            results += updater->updateDynamicObservations(frame_id, new_values, new_factors, update_params);
        }
        results += updater->updateStaticObservations(frame_id, new_values, new_factors, update_params);

    }

    return {new_values, new_factors};
}


StateQuery<gtsam::Point3>  RGBDBackendModule::Accessor::getStaticLandmark(TrackletId tracklet_id) const {
    const auto lmk = getMap()->getLandmark(tracklet_id);
    CHECK(lmk);

    return this->query<gtsam::Point3>(
        lmk->makeStaticKey()
    );
}


MotionEstimateMap RGBDBackendModule::Accessor::getObjectMotions(FrameId frame_id) const {
    MotionEstimateMap motion_estimates;

    const auto frame_node = getMap()->getFrame(frame_id);
    if(!frame_node) {
        return motion_estimates;
    }

    const auto object_seen = frame_node->objects_seen.template collectIds<ObjectId>();
    for(ObjectId object_id : object_seen) {
        StateQuery<Motion3> motion_query = this->getObjectMotion(frame_id, object_id);
        if(motion_query) {
            motion_estimates.insert2(
                object_id,
                ReferenceFrameValue<Motion3>(
                    motion_query.get(),
                    ReferenceFrame::GLOBAL
                ));
        }
    }
    return motion_estimates;
}

EstimateMap<ObjectId, gtsam::Pose3> RGBDBackendModule::Accessor::getObjectPoses(FrameId frame_id) const {
    EstimateMap<ObjectId, gtsam::Pose3> pose_estimates;

    const auto frame_node = getMap()->getFrame(frame_id);
    if(!frame_node) {
        return pose_estimates;
    }

    const auto object_seen = frame_node->objects_seen.template collectIds<ObjectId>();
    for(ObjectId object_id : object_seen) {
        StateQuery<gtsam::Pose3> object_pose = this->getObjectPose(frame_id, object_id);
        if(object_pose) {
            pose_estimates.insert2(
                object_id,
                ReferenceFrameValue<gtsam::Pose3>(
                    object_pose.get(),
                    ReferenceFrame::GLOBAL
                ));
        }
    }
    return pose_estimates;
}

ObjectPoseMap RGBDBackendModule::Accessor::getObjectPoses() const {

    ObjectPoseMap object_poses;
    for(FrameId frame_id : getMap()->getFrameIds()) {
        EstimateMap<ObjectId, gtsam::Pose3> per_object_pose = this->getObjectPoses(frame_id);

        for(const auto&[object_id, pose] : per_object_pose) {
            if(!object_poses.exists(object_id)) {
                object_poses.insert2(object_id, gtsam::FastMap<FrameId, gtsam::Pose3>{});
            }

            auto& per_frame_pose = object_poses.at(object_id);
            per_frame_pose.insert2(frame_id, pose);

        }
    }
    return object_poses;
}

StatusLandmarkEstimates RGBDBackendModule::Accessor::getDynamicLandmarkEstimates(FrameId frame_id) const {
    const auto frame_node = getMap()->getFrame(frame_id);
    CHECK_NOTNULL(frame_node);

    StatusLandmarkEstimates estimates;
    const auto object_seen = frame_node->objects_seen.template collectIds<ObjectId>();
    for(ObjectId object_id : object_seen) {
        estimates += this->getDynamicLandmarkEstimates(frame_id, object_id);
    }
    return estimates;

}
StatusLandmarkEstimates RGBDBackendModule::Accessor::getDynamicLandmarkEstimates(FrameId frame_id, ObjectId object_id) const {
    const auto frame_node = getMap()->getFrame(frame_id);
    CHECK_NOTNULL(frame_node);

    if(!frame_node->objectObserved(object_id)) {
        return StatusLandmarkEstimates{};
    }

    StatusLandmarkEstimates estimates;
    const auto& dynamic_landmarks = frame_node->dynamic_landmarks;
    for(auto lmk_node : dynamic_landmarks) {
        const auto tracklet_id = lmk_node->tracklet_id;

        if(object_id != lmk_node->object_id) {
            continue;
        }

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

StatusLandmarkEstimates RGBDBackendModule::Accessor::getStaticLandmarkEstimates(FrameId frame_id) const {
     //dont go over the frames as this contains references to the landmarks multiple times
    //e.g. the ones seen in that frame
    StatusLandmarkEstimates estimates;

    const auto frame_node = getMap()->getFrame(frame_id);
    CHECK_NOTNULL(frame_node);


    for(const auto& landmark_node : frame_node->static_landmarks) {
        if(landmark_node->isStatic()) {
            StateQuery<gtsam::Point3> lmk_query = getStaticLandmark(landmark_node->tracklet_id);
            if(lmk_query) {
                estimates.push_back(
                    LandmarkStatus::StaticInGlobal(
                        lmk_query.get(), //estimate
                        LandmarkStatus::MeaninglessFrame,
                        landmark_node->getId(), //tracklet id
                        LandmarkStatus::Method::OPTIMIZED
                    ) //status
                );
            }
        }
    }
    return estimates;
}

StatusLandmarkEstimates RGBDBackendModule::Accessor::getFullStaticMap() const {
     //dont go over the frames as this contains references to the landmarks multiple times
    //e.g. the ones seen in that frame
    StatusLandmarkEstimates estimates;

    const auto landmarks = getMap()->getLandmarks();

    for(const auto&[_, landmark_node] : landmarks) {
        if(landmark_node->isStatic()) {
            // StateQuery<gtsam::Point3> lmk_query = this->query<gtsam::Point3>(
            //     landmark_node->makeStaticKey()
            // );
            StateQuery<gtsam::Point3> lmk_query = getStaticLandmark(landmark_node->tracklet_id);
            if(lmk_query) {
                estimates.push_back(
                    LandmarkStatus::StaticInGlobal(
                        lmk_query.get(), //estimate
                        LandmarkStatus::MeaninglessFrame,
                        landmark_node->getId(), //tracklet id
                        LandmarkStatus::Method::OPTIMIZED
                    ) //status
                );
            }
        }
    }
    return estimates;
}

StatusLandmarkEstimates RGBDBackendModule::Accessor::getLandmarkEstimates(FrameId frame_id) const {
    StatusLandmarkEstimates estimates;
    estimates += getStaticLandmarkEstimates(frame_id);
    estimates += getDynamicLandmarkEstimates(frame_id);
    return estimates;
}


bool RGBDBackendModule::Accessor::hasObjectMotionEstimate(FrameId frame_id, ObjectId object_id, Motion3* motion) const {
    const auto frame_node = getMap()->getFrame(frame_id);
    StateQuery<Motion3> motion_query = this->getObjectMotion(frame_id, object_id);

    if(motion_query) {
        if(motion) {
            *motion = motion_query.get();
        }
        return true;
    }
    return false;

}
bool RGBDBackendModule::Accessor::hasObjectMotionEstimate(FrameId frame_id, ObjectId object_id, Motion3& motion) const {
    return hasObjectMotionEstimate(frame_id, object_id, &motion);
}

bool RGBDBackendModule::Accessor::hasObjectPoseEstimate(FrameId frame_id, ObjectId object_id, gtsam::Pose3* pose) const {
    const auto frame_node = getMap()->getFrame(frame_id);
    StateQuery<gtsam::Pose3> pose_query = this->getObjectPose(frame_id, object_id);

    if(pose_query) {
        if(pose) {
            *pose = pose_query.get();
        }
        return true;
    }
    return false;

}
bool RGBDBackendModule::Accessor::hasObjectPoseEstimate(FrameId frame_id, ObjectId object_id, gtsam::Pose3& pose) const {
    return hasObjectPoseEstimate(frame_id, object_id, &pose);
}


gtsam::FastMap<ObjectId, gtsam::Point3>  RGBDBackendModule::Accessor::computeObjectCentroids(FrameId frame_id) const {
    gtsam::FastMap<ObjectId, gtsam::Point3> centroids;

    const auto frame_node = getMap()->getFrame(frame_id);
    if(!frame_node) {
        return centroids;
    }

    const auto object_seen = frame_node->objects_seen.template collectIds<ObjectId>();
    for(ObjectId object_id : object_seen) {
        const auto[centroid, result] = computeObjectCentroid(frame_id, object_id);

        if(result) {
            centroids.insert2(object_id, centroid);
        }
    }
    return centroids;
}


std::tuple<gtsam::Point3, bool> RGBDBackendModule::Accessor::computeObjectCentroid(FrameId frame_id, ObjectId object_id) const {
    const StatusLandmarkEstimates& dynamic_lmks = this->getDynamicLandmarkEstimates(frame_id, object_id);

    //convert to point cloud - should be a map with only one map in it
    CloudPerObject object_clouds = groupObjectCloud(dynamic_lmks, this->getSensorPose(frame_id).get());
    if(object_clouds.size() == 0) {
        //TODO: why does this happen so much!!!
        LOG(INFO) << "Cannot collect object clouds from dynamic landmarks of " << object_id << " and frame " << frame_id << "!! "
            << " # Dynamic lmks in the map for this object at this frame was " << dynamic_lmks.size(); //<< " but reocrded lmks was " << dynamic_landmarks.size();
        return {gtsam::Point3{}, false};
    }
    CHECK_EQ(object_clouds.size(), 1);
    CHECK(object_clouds.exists(object_id));

    const auto dynamic_point_cloud = object_clouds.at(object_id);
    pcl::PointXYZ centroid;
    pcl::computeCentroid(dynamic_point_cloud, centroid);
    //TODO: outlier reject?
    gtsam::Point3 translation = pclPointToGtsam(centroid);
    return {translation, true};
}

gtsam::Pose3 RGBDBackendModule::Updater::getInitialOrLinearizedSensorPose(FrameId frame_id) const {
    const auto accessor = this->accessorFromTheta();
    // sensor pose from a previous/current linearisation point
    StateQuery<gtsam::Pose3> X_k_theta = accessor->getSensorPose(frame_id);

    CHECK(parent_->initial_camera_poses_.exists(frame_id));
    gtsam::Pose3 X_k_initial = parent_->initial_camera_poses_.at(frame_id);
    //take either the query value from the map (if we have a previous initalisation), or the estimate from the camera
    gtsam::Pose3 X_k;
    getSafeQuery(X_k, X_k_theta, X_k_initial);
    return X_k;
}


void RGBDBackendModule::Updater::setInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values) {
    new_values.insert(CameraPoseSymbol(frame_id_k), T_world_camera);
    theta_.insert_or_assign(new_values);
}

void RGBDBackendModule::Updater::setInitialPosePrior(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::NonlinearFactorGraph& new_factors) {
    auto initial_pose_prior = parent_->initial_pose_prior_;
    new_factors.addPrior(CameraPoseSymbol(frame_id_k), T_world_camera, initial_pose_prior);
    factors_ += new_factors;
}

void RGBDBackendModule::Updater::addOdometry(FrameId from_frame, FrameId to_frame, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
    CHECK_LT(from_frame, to_frame);
    CHECK_GT(from_frame, getMap()->firstFrameId());

    for(auto frame_id = from_frame; frame_id <= to_frame; frame_id++) {
        gtsam::Values values;
        gtsam::NonlinearFactorGraph graph;

        // pose estimate from frontend
        CHECK(parent_->initial_camera_poses_.exists(frame_id));
        const gtsam::Pose3& T_world_camera_k = parent_->initial_camera_poses_.at(frame_id);

        addOdometry(frame_id, T_world_camera_k, values, graph);

        new_values.insert(values);
        new_factors += graph;
    }
}

void RGBDBackendModule::Updater::addOdometry(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
    new_values.insert(CameraPoseSymbol(frame_id_k), T_world_camera);
    theta_.insert_or_assign(new_values);
    auto odometry_noise = parent_->odometry_noise_;

    auto map = getMap();

    CHECK_GT(frame_id_k, map->firstFrameId());
    const FrameId frame_id_k_1 = frame_id_k - 1u;

    // pose estimate from frontend
    CHECK(parent_->initial_camera_poses_.exists(frame_id_k));
    const gtsam::Pose3& T_world_camera_k_1_frontend = parent_->initial_camera_poses_.at(frame_id_k_1);

    LOG(INFO) << "Adding odom between " << frame_id_k_1 << " and " << frame_id_k;
    const gtsam::Pose3 odom = T_world_camera_k_1_frontend.inverse() * T_world_camera;

    factor_graph_tools::addBetweenFactor(frame_id_k_1, frame_id_k, odom, odometry_noise, new_factors);
    factors_ += new_factors;
}

RGBDBackendModule::UpdateObservationResult
RGBDBackendModule::Updater::updateStaticObservations(FrameId from_frame, FrameId to_frame, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, const UpdateObservationParams& update_params) {
    CHECK_LT(from_frame, to_frame);
    CHECK_GT(from_frame, getMap()->firstFrameId());

    UpdateObservationResult result;
    for(auto frame_id = from_frame; frame_id <= to_frame; frame_id++) {
        gtsam::NonlinearFactorGraph factors;
        result += updateStaticObservations(frame_id, new_values, factors, update_params);

        new_factors += factors;
    }

    return result;
}

RGBDBackendModule::UpdateObservationResult
RGBDBackendModule::Updater::updateDynamicObservations(FrameId from_frame, FrameId to_frame, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, const UpdateObservationParams& update_params) {
    CHECK_LT(from_frame, to_frame);
    CHECK_GT(from_frame, getMap()->firstFrameId());

    UpdateObservationResult result;
    for(auto frame_id = from_frame; frame_id <= to_frame; frame_id++) {
        gtsam::NonlinearFactorGraph factors;
        result += updateDynamicObservations(frame_id, new_values, factors, update_params);
        new_factors += factors;
    }
    return result;
}



RGBDBackendModule::UpdateObservationResult
RGBDBackendModule::Updater::updateStaticObservations(FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, const UpdateObservationParams& update_params) {
    auto map = getMap();
    const auto& params = parent_->base_params_;
    auto static_point_noise = parent_->static_point_noise_;

    UpdateObservationResult result;
    if(update_params.enable_debug_info) {
        result.debug_info = DebugInfo();
    }

    Accessor::Ptr accessor = this->accessorFromTheta();

    const auto frame_node_k = map->getFrame(frame_id_k);
    CHECK_NOTNULL(frame_node_k);

    // pose estimate from frontend
    CHECK(parent_->initial_camera_poses_.exists(frame_id_k));
    const gtsam::Pose3& T_world_camera_frontend = parent_->initial_camera_poses_.at(frame_id_k);


    VLOG(20) << "Looping over " <<  frame_node_k->static_landmarks.size() << " static lmks for frame " << frame_id_k;
    for(const auto& lmk_node : frame_node_k->static_landmarks) {


        const gtsam::Key point_key = lmk_node->makeStaticKey();
        //check if lmk node is already in map (which should mean it is equivalently in isam)
        if(is_other_values_in_map.exists(point_key)) {
            const Landmark measured = lmk_node->getMeasurement(frame_id_k).landmark;
            new_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
                frame_node_k->makePoseKey(), //pose key for this frame
                point_key,
                measured,
                static_point_noise
            );

            if(result.debug_info) result.debug_info->num_static_factors++;
            result.updateAffectedObject(frame_id_k, 0);

        }
        else {
            //see if we have enough observations to add this lmk
            if(lmk_node->numObservations() < params.min_static_obs_) { continue;}

            //this condition should only run once per tracklet (ie.e the first time the tracklet has enough observations)
            //we gather the tracklet observations and then initalise it in the new values
            //these should then get added to the map
            //and map_->exists() should return true for all other times
            FrameNodePtrSet<LandmarkKeypoint> seen_frames = lmk_node->getSeenFrames();
            for(const auto& seen_frame : seen_frames) {

                auto seen_frame_id = seen_frame->getId();
                //only iterate up to the query frame
                if((FrameId)seen_frame_id > frame_id_k) { break; }

                //if we should not backtrack, only add the current frame!!!
                const auto do_backtrack = update_params.do_backtrack;
                if(!do_backtrack && (FrameId)seen_frame_id < frame_id_k) { continue; }

                const Landmark& measured = lmk_node->getMeasurement(seen_frame).landmark;
                new_factors.emplace_shared<PoseToPointFactor>(
                    seen_frame->makePoseKey(), //pose key at previous frames
                    point_key,
                    measured,
                    static_point_noise
                );
                result.updateAffectedObject(seen_frame_id, 0);
            }

            //pick the one in this frame
            const Landmark& measured = lmk_node->getMeasurement(frame_id_k).landmark;
            //add initial value, either from measurement or previous estimate
            gtsam::Point3 lmk_world;
            getSafeQuery(lmk_world, accessor->getStaticLandmark(lmk_node->tracklet_id), gtsam::Point3(T_world_camera_frontend * measured));
            new_values.insert(point_key, lmk_world);
            is_other_values_in_map.insert2(point_key, true);

            if(result.debug_info) result.debug_info->num_new_static_points++;
            result.updateAffectedObject(frame_id_k, 0);

        }
    }

    //update internal data structures
    theta_.insert_or_assign(new_values);
    factors_ += new_factors;

    if(result.debug_info) LOG(INFO) << "Num new static points: " << result.debug_info->num_new_static_points << " Num new static factors " << result.debug_info->num_static_factors;
    return result;
}


RGBDBackendModule::UpdateObservationResult
RGBDBackendModule::Updater::updateDynamicObservations(
        FrameId frame_id_k,
        gtsam::Values& new_values,
        gtsam::NonlinearFactorGraph& new_factors,
        const UpdateObservationParams& update_params) {
    auto map = getMap();
    const auto& params = parent_->base_params_;

    Accessor::Ptr accessor = this->accessorFromTheta();

    // collect noise models to be used
    auto landmark_motion_noise = parent_->landmark_motion_noise_;
    auto dynamic_point_noise = parent_->dynamic_point_noise_;


    UpdateObservationResult result;
    if(update_params.enable_debug_info) {
        result.debug_info = DebugInfo();
    }

    //! At least three points on the object are required to solve
    //! otherise the system is indeterminate
    constexpr static size_t kMinNumberPoints = 3u;

    const FrameId frame_id_k_1 = frame_id_k - 1u;
    LOG(INFO) << "Add dynamic observations between frames " << frame_id_k_1 << " and " << frame_id_k;
    const auto frame_node_k = map->getFrame(frame_id_k);
    const auto frame_node_k_1 = map->getFrame(frame_id_k_1);
    CHECK_NOTNULL(frame_node_k_1);

    // pose estimate from frontend
    CHECK(parent_->initial_camera_poses_.exists(frame_id_k));
    const gtsam::Pose3& T_world_camera_initial_k = new_values.at<gtsam::Pose3>(CameraPoseSymbol(frame_id_k));

    //for each object
    for(const auto& object_node : frame_node_k->objects_seen) {

        DebugInfo::ObjectInfo object_debug_info;
        const ObjectId object_id = object_node->getId();

        //first check that object exists in the previous frame
        if(!frame_node_k->objectMotionExpected(object_id)) {
            continue;
        }
        // possibly the longest call?
        // landmarks on this object seen at frame k
        auto seen_lmks_k = object_node->getLandmarksSeenAtFrame(frame_id_k);

        //if we dont have at least N observations of this object in this frame AND the previous frame
        if(seen_lmks_k.size() < kMinNumberPoints || object_node->getLandmarksSeenAtFrame(frame_id_k_1).size() < kMinNumberPoints) {
            continue;
        }

        //iterate over each lmk we have on this object
        for(const auto& obj_lmk_node : seen_lmks_k) {
            CHECK_EQ(obj_lmk_node->getObjectId(), object_id);

            //see if we have enough observations to add this lmk
            if(obj_lmk_node->numObservations() < params.min_dynamic_obs_) { continue;}

            TrackletId tracklet_id = obj_lmk_node->getId();

            //if does not exist, we need to go back and all the previous measurements & factors & motions
            if(!isDynamicTrackletInMap(obj_lmk_node)) {

                //add the points/motions from the past
                auto seen_frames = obj_lmk_node->getSeenFrames();
                //assert frame observations are continuous?

                //start at the first frame we want to start adding points in as we know have seen them enough times
                //start from +1, becuase the motion index is k-1 to k and there is no motion k-2 to k-1
                //but the first poitns we want to add are at k-1
                FrameId starting_motion_frame;
                if(update_params.do_backtrack) {
                    starting_motion_frame = seen_frames.getFirstIndex<FrameId>() + 1u; //as we index the motion from k
                }
                else {
                    //start from the requested index, this will still mean that we will add the previous frame as always add a motion
                    //between k-1 and k
                    starting_motion_frame = frame_id_k;

                    if(starting_motion_frame < seen_frames.getFirstIndex<FrameId>() + 1u) {
                        //if the requested starting frame is not the first frame + 1u of the actul track (ie. the second seen frame)
                        //we cannot use it yet as we have to index BACKWARDS from the starting motion frame
                        //if we used frame_id_k as the starting_motion_frame, we would end up with an iterator
                        //pointing to the end
                        //instead, we will have to get it next frame!!!
                        continue;
                    }
                }

                auto starting_motion_frame_itr = seen_frames.find(starting_motion_frame);
                CHECK(starting_motion_frame_itr != seen_frames.end()) << "Starting motion frame is " << starting_motion_frame << " but first frame is " << seen_frames.getFirstIndex<FrameId>();

                std::stringstream ss;
                ss << "Going back to add point on object " << object_id << " at frames\n";

                //iterate over k-N to k (inclusive) and all all
                for(auto seen_frames_itr = starting_motion_frame_itr; seen_frames_itr != seen_frames.end(); seen_frames_itr++) {
                    auto seen_frames_itr_prev = seen_frames_itr;
                    std::advance(seen_frames_itr_prev, -1);
                    CHECK(seen_frames_itr_prev != seen_frames.end()) << " For object  " << object_id;

                    auto query_frame_node_k = *seen_frames_itr;
                    auto query_frame_node_k_1 = *seen_frames_itr_prev;

                    CHECK_EQ(query_frame_node_k->frame_id, query_frame_node_k_1->frame_id + 1u);

                    //add points UP TO AND INCLUDING the current frame
                    if(query_frame_node_k->frame_id > frame_id_k) {
                        break;
                    }

                    //point needs to be be in k and k-1 -> we have validated the object exists in these two frames
                    //but not the points
                    CHECK(obj_lmk_node->seenAtFrame(query_frame_node_k->frame_id));
                    if(!obj_lmk_node->seenAtFrame(query_frame_node_k_1->frame_id)) {
                        LOG(WARNING) << "Tracklet " << tracklet_id << " on object " << object_id << " seen at " << query_frame_node_k->frame_id << " but not " << query_frame_node_k_1->frame_id;
                        break;
                    } //this miay mean this this point never gets added?

                    ss << query_frame_node_k_1->frame_id << " " << query_frame_node_k->frame_id << "\n";


                    //this should DEFINITELY be in the map, as long as we update the values in the map everyy time
                    StateQuery<gtsam::Pose3> T_world_camera_k_1_query = accessor->getSensorPose(query_frame_node_k_1->frame_id);
                    CHECK(T_world_camera_k_1_query) << "Failed cam pose query at frame " << query_frame_node_k_1->frame_id
                        << ". This may happen if the map_ is not updated every iteration OR something is wrong with the tracking...";
                    const gtsam::Pose3 T_world_camera_k_1 = T_world_camera_k_1_query.get();

                    gtsam::Pose3 T_world_camera_k = getInitialOrLinearizedSensorPose(query_frame_node_k_1->frame_id);

                    PointUpdateContext point_context;
                    point_context.lmk_node = obj_lmk_node;
                    point_context.frame_node_k_1 = query_frame_node_k_1;
                    point_context.frame_node_k = query_frame_node_k;
                    point_context.X_k_measured = T_world_camera_k;
                    point_context.X_k_1_measured = T_world_camera_k_1;


                    //this assumes we add all the points in order and have continuous frames (which we should have?)
                    if(seen_frames_itr == starting_motion_frame_itr) {
                        point_context.is_starting_motion_frame = true;

                    }

                    dynamicPointUpdateCallback(point_context, result, new_values, new_factors);
                    //update internal theta and factors
                    theta_.insert_or_assign(new_values);
                }
            }
            else {
                //these tracklets should already be in the graph so we should only need to add the new measurements from this frame
                //check that we have previous point for this frame

                PointUpdateContext point_context;
                point_context.lmk_node = obj_lmk_node;
                point_context.frame_node_k_1 = frame_node_k_1;
                point_context.frame_node_k = frame_node_k;
                point_context.X_k_1_measured = getInitialOrLinearizedSensorPose(frame_node_k_1->frame_id);
                point_context.X_k_measured = getInitialOrLinearizedSensorPose(frame_node_k->frame_id);
                point_context.is_starting_motion_frame = false;
                dynamicPointUpdateCallback(point_context, result, new_values, new_factors);

                //update internal theta and factors
                theta_.insert_or_assign(new_values);
            }
        }
    }

    //iterate over objects for which a motion was added
    //becuase we add lots of new points every frame, we may go over the same object many times
    //to account for backtracking over new points over this object
    //this is a bit inefficient as we do this iteration even if no new object values are added
    //becuuse we dont know if the affected frames are becuase of old points as well as new points
    for(const auto&[object_id, frames_affected] : result.objects_affected_per_frame) {
        LOG(INFO) << "Iterating over frames for which a motion was added " << container_to_string(frames_affected) << " for object " << object_id;

        auto object_node = map->getObject(object_id);
        const auto first_seen_frame = object_node->getFirstSeenFrame();

        std::vector<FrameId> frames_affected_vector(
            frames_affected.begin(), frames_affected.end()
        );

        //this is NO LONGER the case in the object centric case
        // //TODO: should always have at least two frames (prev and current) as we must add factors on this frame and the previous frame
        // CHECK_GE(frames_affected_vector.size(), 2u);
        for(size_t frame_idx = 0; frame_idx < frames_affected_vector.size(); frame_idx++) {
            const FrameId frame_id = frames_affected_vector.at(frame_idx);
            auto frame_node_k_impl = getMap()->getFrame(frame_id);

            ObjectUpdateContext object_update_context;
            //perform motion check on this frame -> if this is the first frame (of at least 2)
            //then there should be no motion at this frame (since it is k-1 of a motion pair and there is no k-2)
            //this nolonger holds for the object centric case as we dont deal with motion pairs!!
            if(frame_idx == 0) {
                object_update_context.has_motion_pair = false;
            }
            else {
                object_update_context.has_motion_pair = true;
            }
            object_update_context.frame_node_k = frame_node_k_impl;
            object_update_context.object_node = object_node;
            objectUpdateContext(object_update_context, result, new_values, new_factors);

            //update internal theta and factors
            theta_.insert_or_assign(new_values);
        }
    }

     //this doesnt really work any more as debug info is meant to be per frame?
    if(result.debug_info && VLOG_IS_ON(20)) {
        for(const auto&[object_id, object_info] : result.debug_info->getObjectInfos()) {
            std::stringstream ss;
            ss << "Object id debug info: " << object_id << "\n";
            ss << object_info;
            LOG(INFO) << ss.str();
        }
    }

    factors_ += new_factors;
    return result;
}

void RGBDBackendModule::Updater::logBackendFromMap() {
    LOG(INFO) << "Logging from backend!";
    BackendLogger logger(loggerPrefix());
    const auto& gt_packet_map = parent_->gt_packet_map_;
    auto map = getMap();
    auto accessor = this->accessorFromTheta();

    const ObjectPoseMap object_pose_map =  accessor->getObjectPoses();

   for(FrameId frame_k : map->getFrameIds()) {

        std::stringstream ss;
        ss << "Logging data from map at frame " << frame_k;

        //get MotionestimateMap
        // const MotionEstimateMap motions = map->getMotionEstimates(frame_k);
        {
        const MotionEstimateMap motions = accessor->getObjectMotions(frame_k);
        auto result = logger.logObjectMotion(gt_packet_map, frame_k, motions);
        if(result) ss << " Logged " << *result << " motions from " << motions.size() << " computed motions.";
        else ss << " Could not log object motions.";
        }

        StateQuery<gtsam::Pose3> X_k_query = accessor->getSensorPose(frame_k);

        if(X_k_query) {
            logger.logCameraPose(gt_packet_map, frame_k, X_k_query.get());
        }
        else {
            LOG(WARNING) << "Could not log camera pose estimate at frame " << frame_k;
        }


        logger.logObjectPose(gt_packet_map, frame_k, object_pose_map);


        if(map->frameExists(frame_k)) {
            //TODO:
            StatusLandmarkEstimates static_map = accessor->getStaticLandmarkEstimates(frame_k);
            // // LOG(INFO) << "static map size " << static_map.size();
            StatusLandmarkEstimates dynamic_map = accessor->getDynamicLandmarkEstimates(frame_k);
            // LOG(INFO) << "dynamic map size " << dynamic_map.size();

            CHECK(X_k_query); //actually not needed for points in world!!
            logger.logPoints(frame_k, *X_k_query, static_map);
            logger.logPoints(frame_k, *X_k_query, dynamic_map);
        }

        LOG(INFO) << ss.str();
    }


}


RGBDBackendModule::Accessor::Ptr
RGBDBackendModule::Updater::accessorFromTheta() const {
    if(!accessor_theta_) {
        accessor_theta_ = createAccessor(&theta_);
    }
    return accessor_theta_;
}

StateQuery<gtsam::Pose3> RGBDBackendModule::LLAccessor::getSensorPose(FrameId frame_id) const {
    const auto frame_node = getMap()->getFrame(frame_id);
    CHECK_NOTNULL(frame_node);
    return this->query<gtsam::Pose3>(
        frame_node->makePoseKey()
    );
}

StateQuery<gtsam::Pose3> RGBDBackendModule::LLAccessor::getObjectMotion(FrameId frame_id, ObjectId object_id) const {
    const auto frame_node_k = getMap()->getFrame(frame_id);
    CHECK_NOTNULL(frame_node_k);
    const auto object_motion_key = frame_node_k->makeObjectMotionKey(object_id);

    if(frame_id < 2) {
        //if the current frame is 1 then skip as we never get a frame that is 0
        //this is because the first frame (frame_id=0) is never sent to the backend
        //as at least two frames (0 and 1) are needed to track!
        return StateQuery<gtsam::Pose3>::NotInMap(object_motion_key);
    }

    const auto object_pose_k = this->getObjectPose(frame_id, object_id);
    const auto object_pose_k_1 = this->getObjectPose(frame_id - 1u, object_id);

    if(object_pose_k && object_pose_k_1) {
        // ^w_{k-1}H_k = ^wL_k \: ^wL_{k-1}^{-1}
        const gtsam::Pose3 motion = object_pose_k.get() * object_pose_k_1->inverse();
        return StateQuery<gtsam::Pose3>{object_motion_key, motion};
    }
    return StateQuery<gtsam::Pose3>::NotInMap(object_motion_key);
}
StateQuery<gtsam::Pose3> RGBDBackendModule::LLAccessor::getObjectPose(FrameId frame_id, ObjectId object_id) const {
    const auto frame_node = getMap()->getFrame(frame_id);
    if(!frame_node) {
        return StateQuery<gtsam::Pose3>::InvalidMap();
    }

    // CHECK(frame_node) << "Frame Id is null at frame " << frame_id;
    return this->query<gtsam::Pose3>(
        frame_node->makeObjectPoseKey(object_id)
    );
}

StateQuery<gtsam::Point3> RGBDBackendModule::LLAccessor::getDynamicLandmark(FrameId frame_id, TrackletId tracklet_id) const {
    const auto lmk = getMap()->getLandmark(tracklet_id);
    CHECK_NOTNULL(lmk);

    return this->query<gtsam::Point3>(
        lmk->makeDynamicKey(frame_id)
    );
}



void RGBDBackendModule::LLUpdater::dynamicPointUpdateCallback(
        const PointUpdateContext& context, UpdateObservationResult& result,
        gtsam::Values& new_values,
        gtsam::NonlinearFactorGraph& new_factors) {
    const auto lmk_node = context.lmk_node;
    const auto frame_node_k_1 = context.frame_node_k_1;
    const auto frame_node_k = context.frame_node_k;

    auto dynamic_point_noise = parent_->dynamic_point_noise_;

    Accessor::Ptr theta_accessor = this->accessorFromTheta();

    const gtsam::Key object_point_key_k_1 = lmk_node->makeDynamicKey(frame_node_k_1->frame_id);
    const gtsam::Key object_point_key_k = lmk_node->makeDynamicKey(frame_node_k->frame_id);

    // if first motion (i.e first time we have both k-1 and k), add both at k-1 and k
    if(context.is_starting_motion_frame) {
        new_factors.emplace_shared<PoseToPointFactor>(
            frame_node_k_1->makePoseKey(), //pose key at previous frames
            object_point_key_k_1,
            lmk_node->getMeasurement(frame_node_k_1).landmark,
            dynamic_point_noise
        );
        // object_debug_info.num_dynamic_factors++;
        result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());
        if(result.debug_info) result.debug_info->getObjectInfo(context.getObjectId()).num_dynamic_factors++;

        //add landmark at previous frame
        const Landmark measured_k_1 = lmk_node->getMeasurement(frame_node_k_1->frame_id).landmark;
        Landmark lmk_world_k_1;
        getSafeQuery(
            lmk_world_k_1,
            theta_accessor->query<Landmark>(object_point_key_k_1),
            gtsam::Point3(context.X_k_1_measured * measured_k_1)
        );
        new_values.insert(object_point_key_k_1, lmk_world_k_1);
        if(result.debug_info) result.debug_info->getObjectInfo(context.getObjectId()).num_new_dynamic_points++;
    }


    // previous point must be added by the previous iteration
    CHECK(new_values.exists(object_point_key_k_1) || theta_accessor->exists(object_point_key_k_1));

    const Landmark measured_k = lmk_node->getMeasurement(frame_node_k).landmark;

    new_factors.emplace_shared<PoseToPointFactor>(
        frame_node_k->makePoseKey(), //pose key at this (in the iteration) frames
        object_point_key_k,
        measured_k,
        dynamic_point_noise
    );
    if(result.debug_info) result.debug_info->getObjectInfo(context.getObjectId()).num_dynamic_factors++;
    result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());

    Landmark lmk_world_k;
    getSafeQuery(
        lmk_world_k,
        theta_accessor->query<Landmark>(object_point_key_k),
        gtsam::Point3(context.X_k_measured * measured_k)
    );
    new_values.insert(object_point_key_k, lmk_world_k);
    if(result.debug_info) result.debug_info->getObjectInfo( context.getObjectId()).num_new_dynamic_points++;


    const gtsam::Key object_pose_k_1_key = frame_node_k_1->makeObjectPoseKey(context.getObjectId());
    const gtsam::Key object_pose_k_key = frame_node_k->makeObjectPoseKey(context.getObjectId());

    auto landmark_motion_noise = parent_->landmark_motion_noise_;
    new_factors.emplace_shared<LandmarkMotionPoseFactor>(
        object_point_key_k_1,
        object_point_key_k,
        object_pose_k_1_key,
        object_pose_k_key,
        landmark_motion_noise
    );
    result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());
    result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
    if(result.debug_info) result.debug_info->getObjectInfo(context.getObjectId()).num_motion_factors++;

    //mark as now in map
    is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), true);

}


void RGBDBackendModule::LLUpdater::objectUpdateContext(
        const ObjectUpdateContext& context, UpdateObservationResult& result,
        gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors)
{
    auto frame_node_k = context.frame_node_k;
    const gtsam::Key object_pose_key_k = frame_node_k->makeObjectPoseKey(context.getObjectId());

    Accessor::Ptr theta_accessor = this->accessorFromTheta();

    if(!is_other_values_in_map.exists(object_pose_key_k)) {
        FrameId frame_id_k_1 = context.getFrameId() - 1u;
        //try and propogate from previous initalisation
        StateQuery<gtsam::Pose3> pose_k_1_query = theta_accessor->getObjectPose(
            frame_id_k_1,
            context.getObjectId()
        );

        //takes me from k-1 to k
        Motion3 motion;
        gtsam::Pose3 object_pose_k;
        //we have a motion from k-1 to k and a pose at k
        if(parent_->hasFrontendMotionEstimate(context.getFrameId(), context.getObjectId(), &motion) && pose_k_1_query) {
            object_pose_k = motion * pose_k_1_query.get();
        }
        else {
            //no motion or previous pose, so initalise with centroid translation and identity rotation
            const auto[centroid_initial, result] = theta_accessor->computeObjectCentroid(context.getFrameId(), context.getObjectId());
            CHECK(result);
            gtsam::Pose3 initial_object_pose(gtsam::Rot3::Identity(), centroid_initial);

            getSafeQuery(
                object_pose_k,
                theta_accessor->query<gtsam::Pose3>(object_pose_key_k),
                initial_object_pose
            );
        }

        new_values.insert(object_pose_key_k, object_pose_k);
        is_other_values_in_map.insert2(object_pose_key_k, true);
        LOG(INFO) << "Adding object pose key " << DynoLikeKeyFormatter(object_pose_key_k);
    }

    if(FLAGS_use_smoothing_factor) {
        const auto frame_id = context.getFrameId();
        const auto object_id = context.getObjectId();
        if(frame_id < 2) return;

        auto frame_node_k_2 = getMap()->getFrame(frame_id - 2u);
        auto frame_node_k_1 = getMap()->getFrame(frame_id - 1u);

        if (!frame_node_k_2 || !frame_node_k_1) { return; }

        //pose key at k-2
        const gtsam::Symbol object_pose_key_k_2 = frame_node_k_2->makeObjectPoseKey(object_id);
        //pose key at previous frame (k-1)
        const gtsam::Symbol object_pose_key_k_1 = frame_node_k_1->makeObjectPoseKey(object_id);

        auto object_smoothing_noise = parent_->object_smoothing_noise_;
        CHECK(object_smoothing_noise);
        CHECK_EQ(object_smoothing_noise->dim(), 6u);

        {
            ObjectId object_label_k_2, object_label_k_1, object_label_k;
            FrameId frame_id_k_2, frame_id_k_1, frame_id_k;
            CHECK(reconstructPoseInfo(object_pose_key_k_2, object_label_k_2, frame_id_k_2));
            CHECK(reconstructPoseInfo(object_pose_key_k_1, object_label_k_1, frame_id_k_1));
            CHECK(reconstructPoseInfo(object_pose_key_k, object_label_k, frame_id_k));
            CHECK_EQ(object_label_k_2, object_label_k);
            CHECK_EQ(object_label_k_1, object_label_k);
            CHECK_EQ(frame_id_k_1 + 1, frame_id_k); //assumes consequative frames
            CHECK_EQ(frame_id_k_2 + 1, frame_id_k_1); //assumes consequative frames

        }

        //if the motion key at k (motion from k-1 to k), and key at k-1 (motion from k-2 to k-1)
        //exists in the map or is about to exist via new values, add the smoothing factor
        if(is_other_values_in_map.exists(object_pose_key_k_1) &&
            is_other_values_in_map.exists(object_pose_key_k) &&
            is_other_values_in_map.exists(object_pose_key_k_2)) {

            new_factors.emplace_shared<LandmarkPoseSmoothingFactor>(
                object_pose_key_k_2,
                object_pose_key_k_1,
                object_pose_key_k,
                object_smoothing_noise
            );
            if(result.debug_info) result.debug_info->getObjectInfo(object_id).smoothing_factor_added=true;
            VLOG(50) << "Adding smoothing " << DynoLikeKeyFormatter(object_pose_key_k_2) << " -> " <<  DynoLikeKeyFormatter(object_pose_key_k);
        }


    }



}

StateQuery<gtsam::Pose3> RGBDBackendModule::MotionWorldAccessor::getObjectMotion(FrameId frame_id, ObjectId object_id) const {
    const auto frame_node_k = getMap()->getFrame(frame_id);
    CHECK(frame_node_k);

    //from k-1 to k
    return this->query<gtsam::Pose3>(
        frame_node_k->makeObjectMotionKey(object_id)
    );
}
StateQuery<gtsam::Pose3> RGBDBackendModule::MotionWorldAccessor::getObjectPose(FrameId frame_id, ObjectId object_id) const {
    const auto object_poses = getObjectPoses(frame_id);
    if(object_poses.exists(object_id)) {
        return StateQuery<gtsam::Pose3>(
            ObjectPoseSymbol(object_id, frame_id),
            object_poses.at(object_id)
        );
    }
    return StateQuery<gtsam::Pose3>::InvalidMap();
}

EstimateMap<ObjectId, gtsam::Pose3> RGBDBackendModule::MotionWorldAccessor::getObjectPoses(FrameId frame_id) const {
    EstimateMap<ObjectId, gtsam::Pose3> object_poses;
    //for each object, go over the cached object poses and check if that object has a pose at the query frame
    for(const auto&[object_id, pose] : object_pose_cache_.collectByFrame(frame_id)) {
        object_poses.insert2(object_id,
            ReferenceFrameValue<gtsam::Pose3>(
                pose,
                ReferenceFrame::GLOBAL
            )
        );
    }
    return object_poses;
}

void RGBDBackendModule::MotionWorldAccessor::postUpdateCallback() {
    //update object_pose_cache_ with new values
    //this means we have to start again at the first frame and update all the poses!!
    ObjectPoseMap object_poses;
    const auto frames = getMap()->getFrames();
    auto frame_itr = frames.begin();

    auto gt_packet_map = parent_->getGroundTruthPackets();

    //advance itr one so we're now at the second frame
    std::advance(frame_itr, 1);
    for(auto itr = frame_itr; itr != frames.end(); itr++) {
        auto prev_itr = itr;
        std::advance(prev_itr, -1);
        CHECK(prev_itr != frames.end());

        const auto[frame_id_k, frame_k_ptr] = *itr;
        const auto[frame_id_k_1, frame_k_1_ptr] = *prev_itr;
        CHECK_EQ(frame_id_k_1 + 1, frame_id_k);

        //collect all object centoids from the latest estimate
        gtsam::FastMap<ObjectId, gtsam::Point3> centroids_k = this->computeObjectCentroids(frame_id_k);
        gtsam::FastMap<ObjectId, gtsam::Point3> centroids_k_1 = this->computeObjectCentroids(frame_id_k_1);
        //collect motions from k-1 to k
        MotionEstimateMap motion_estimates = this->getObjectMotions(frame_id_k);

        //construct centroid vectors in object id order
        gtsam::Point3Vector object_centroids_k_1, object_centroids_k;
        //we may not have a pose for every motion, e.g. if there is a new object at frame k,
        //it wont have a motion yet!
        //if we have a motion we MUST have a pose at the previous frame!
        for(const auto& [object_id, _] : motion_estimates) {
            CHECK(centroids_k.exists(object_id));
            CHECK(centroids_k_1.exists(object_id));

            object_centroids_k.push_back(centroids_k.at(object_id));
            object_centroids_k_1.push_back(centroids_k_1.at(object_id));
        }

        PropogatePoseResult propogation_result;
        if(FLAGS_init_object_pose_from_gt) {
            if(!gt_packet_map) LOG(WARNING) << "FLAGS_init_object_pose_from_gt is true but gt_packet map not provided!";
            dyno::propogateObjectPoses(
                object_poses,
                motion_estimates,
                object_centroids_k_1,
                object_centroids_k,
                frame_id_k,
                gt_packet_map,
                &propogation_result);
        }
        else {
            dyno::propogateObjectPoses(
                object_poses,
                motion_estimates,
                object_centroids_k_1,
                object_centroids_k,
                frame_id_k,
                std::nullopt,
                &propogation_result);
        }

        if(VLOG_IS_ON(20)) {
            //report on how many were propogated with motions for this frame only
            const auto propogation_type = propogation_result.collectByFrame(frame_id_k);
            std::stringstream ss;
            ss << "Propogation result: " << propogation_type.size() << " objects for frame " << frame_id_k;

            size_t n_used_motion = 0;
            for(const auto&[_, type] : propogation_type) {
                if(type == PropogateType::Propogate) {
                    n_used_motion++;
                }
            }
            ss << " of which " << n_used_motion << " were motion propogated";
            VLOG(50) << ss.str();
        }

    }

    VLOG(50) << "Updated object pose cache";
    object_pose_cache_ = object_poses;
}


void RGBDBackendModule::MotionWorldUpdater::dynamicPointUpdateCallback(
        const PointUpdateContext& context, UpdateObservationResult& result,
        gtsam::Values& new_values,
        gtsam::NonlinearFactorGraph& new_factors) {
    const auto lmk_node = context.lmk_node;
    const auto frame_node_k_1 = context.frame_node_k_1;
    const auto frame_node_k = context.frame_node_k;

    auto dynamic_point_noise = parent_->dynamic_point_noise_;
    auto dynamic_projection_noise = parent_->dynamic_pixel_noise_;
    CHECK_NOTNULL(dynamic_projection_noise);

    auto gtsam_calibration = parent_->getGtsamCalibration();

    Accessor::Ptr theta_accessor = this->accessorFromTheta();

    const gtsam::Key object_point_key_k_1 = lmk_node->makeDynamicKey(frame_node_k_1->frame_id);
    const gtsam::Key object_point_key_k = lmk_node->makeDynamicKey(frame_node_k->frame_id);

    // if first motion (i.e first time we have both k-1 and k), add both at k-1 and k
    if(context.is_starting_motion_frame) {
        // new_factors.emplace_shared<GenericProjectionFactor>(
        //     lmk_node->getMeasurement(frame_node_k_1).keypoint,
        //     dynamic_projection_noise,
        //     frame_node_k_1->makePoseKey(),
        //     object_point_key_k_1,
        //     gtsam_calibration,
        //     false, false
        // );

        new_factors.emplace_shared<PoseToPointFactor>(
            frame_node_k_1->makePoseKey(), //pose key at previous frames
            object_point_key_k_1,
            lmk_node->getMeasurement(frame_node_k_1).landmark,
            dynamic_point_noise
        );
        if(result.debug_info) result.debug_info->getObjectInfo(context.getObjectId()).num_dynamic_factors++;
        result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());

        //add landmark at previous frame
        const Landmark measured_k_1 = lmk_node->getMeasurement(frame_node_k_1->frame_id).landmark;
        Landmark lmk_world_k_1;
        getSafeQuery(
            lmk_world_k_1,
            theta_accessor->query<Landmark>(object_point_key_k_1),
            gtsam::Point3(context.X_k_1_measured * measured_k_1)
        );
        new_values.insert(object_point_key_k_1, lmk_world_k_1);
        if(result.debug_info) result.debug_info->getObjectInfo(context.getObjectId()).num_new_dynamic_points++;
    }


    // previous point must be added by the previous iteration
    CHECK(new_values.exists(object_point_key_k_1) || theta_accessor->exists(object_point_key_k_1));

    const Landmark measured_k = lmk_node->getMeasurement(frame_node_k).landmark;

    new_factors.emplace_shared<PoseToPointFactor>(
        frame_node_k->makePoseKey(), //pose key at this (in the iteration) frames
        object_point_key_k,
        measured_k,
        dynamic_point_noise
    );
    if(result.debug_info) result.debug_info->getObjectInfo(context.getObjectId()).num_dynamic_factors++;

    // new_factors.emplace_shared<GenericProjectionFactor>(
    //         lmk_node->getMeasurement(frame_node_k).keypoint,
    //         dynamic_projection_noise,
    //         frame_node_k->makePoseKey(),
    //         object_point_key_k,
    //         gtsam_calibration,
    //         false, false
    //     );
    // object_debug_info.num_dynamic_factors++;
    result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());

    Landmark lmk_world_k;
    getSafeQuery(
        lmk_world_k,
        theta_accessor->query<Landmark>(object_point_key_k),
        gtsam::Point3(context.X_k_measured * measured_k)
    );
    new_values.insert(object_point_key_k, lmk_world_k);
    if(result.debug_info) result.debug_info->getObjectInfo(context.getObjectId()).num_new_dynamic_points++;


    const gtsam::Key object_motion_key_k = frame_node_k->makeObjectMotionKey(context.getObjectId());


    auto landmark_motion_noise = parent_->landmark_motion_noise_;
    new_factors.emplace_shared<LandmarkMotionTernaryFactor>(
        object_point_key_k_1,
        object_point_key_k,
        object_motion_key_k,
        landmark_motion_noise
    );
    result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());
    result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
    if(result.debug_info) result.debug_info->getObjectInfo(context.getObjectId()).num_motion_factors++;


    //mark as now in map
    is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), true);

}

void RGBDBackendModule::MotionWorldUpdater::objectUpdateContext(
        const ObjectUpdateContext& context, UpdateObservationResult& result,
        gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors)
{
    auto frame_node_k = context.frame_node_k;
    const gtsam::Key object_motion_key_k = frame_node_k->makeObjectMotionKey(context.getObjectId());

    Accessor::Ptr theta_accessor = this->accessorFromTheta();

    //skip if no motion pair availble
    if(!context.has_motion_pair) {
        return;
    }

    const auto frame_id = context.getFrameId();
    const auto object_id = context.getObjectId();

    if(!is_other_values_in_map.exists(object_motion_key_k)) {

        //when we have an initial motion - it seems ways better to use 2D reprojection error?
        Motion3 initial_motion = Motion3::Identity();
        // parent_->hasFrontendMotionEstimate(frame_id, object_id, &initial_motion);
        new_values.insert(object_motion_key_k, initial_motion);
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
            new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                object_motion_key_k_1,
                object_motion_key_k,
                gtsam::Pose3::Identity(),
                object_smoothing_noise
            );
            if(result.debug_info) result.debug_info->getObjectInfo(context.getObjectId()).smoothing_factor_added = true;

            // object_debug_info.smoothing_factor_added = true;
        }

        // if(smoothing_added) {
        //     //TODO: add back in
        //     // object_debug_info.smoothing_factor_added = true;
        // }

    }



}



bool RGBDBackendModule::buildSlidingWindowOptimisation(FrameId frame_k, gtsam::Values& optimised_values, double& error_before, double& error_after) {
    const auto overlap_size = FLAGS_opt_window_overlap;
    const auto window_size = FLAGS_opt_window_size;

    if( (frame_k-window_size+1)%(window_size-overlap_size)==0 && (frame_k - first_frame_id_)>=(window_size + 1)) {
        const auto start_frame = frame_k - window_size;
        const auto end_frame = frame_k;
        LOG(INFO) << "Running dynamic slam window on between frames " << start_frame << " - " << end_frame;

        gtsam::Values values;
        gtsam::NonlinearFactorGraph graph;
        std::tie(values, graph) = constructGraph(start_frame, end_frame, true);
        LOG(INFO) << " Finished graph construction";

        graph.error(values);
        gtsam::LevenbergMarquardtParams opt_params;
        if(VLOG_IS_ON(20))
            opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;

        optimised_values = gtsam::LevenbergMarquardtOptimizer(graph, values, opt_params).optimize();
        error_after = graph.error(optimised_values);

        return true;

    }
    return false;
}

// void RGBDBackendModule::addInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
//     new_values.insert(CameraPoseSymbol(frame_id_k), T_world_camera);
//     new_factors.addPrior(CameraPoseSymbol(frame_id_k), T_world_camera, initial_pose_prior_);
// }

// void RGBDBackendModule::addOdometry(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, FrameId frame_id_k_1, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
//     //add state
//     new_values.insert(CameraPoseSymbol(frame_id_k), T_world_camera);
//     timestamp_map_[CameraPoseSymbol(frame_id_k)] = frame_id_k;

//     StateQuery<gtsam::Pose3> pose_query = map_->getPoseEstimate(frame_id_k_1);
//     if(pose_query) {
//         LOG(INFO) << "Adding odom between " << frame_id_k_1 << " and " << frame_id_k;
//         const gtsam::Pose3 T_world_camera_k_1 = pose_query.get();
//         const gtsam::Pose3 odom = T_world_camera_k_1.inverse() * T_world_camera;

//         factor_graph_tools::addBetweenFactor(frame_id_k_1, frame_id_k, odom, odometry_noise_, new_factors);
//     }
// }


// void RGBDBackendModule::updateStaticObservations(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_point_factors) {
//     const FrameNode3d::Ptr frame_node_k = map_->getFrame(frame_id_k);
//     CHECK_NOTNULL(frame_node_k);

//     constexpr static size_t kMinObservations = 2u;

//     //TODO: debug stuff
//     int num_points = 0;

//     LOG(INFO) << "Looping over " <<  frame_node_k->static_landmarks.size() << " static lmks for frame " << frame_id_k;
//     for(const LandmarkNode3d::Ptr& lmk_node : frame_node_k->static_landmarks) {


//         const gtsam::Key point_key = lmk_node->makeStaticKey();
//         //check if lmk node is already in map (which should mean it is equivalently in isam)
//         if(map_->exists(point_key)) {
//             //3d point in camera frame
//             const Landmark& measured = lmk_node->getMeasurement(frame_id_k);
//             new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//                 frame_node_k->makePoseKey(), //pose key for this frame
//                 point_key,
//                 measured,
//                 static_point_noise_
//             );

//             debug_info_.num_static_factors++;

//         }
//         else {
//             //see if we have enough observations to add this lmk
//             if(lmk_node->numObservations() < kMinObservations) { continue;}

//             num_points++;

//             // if(num_points > 5) {break;}

//             //this condition should only run once per tracklet (ie.e the first time the tracklet has enough observations)
//             //we gather the tracklet observations and then initalise it in the new values
//             //these should then get added to the map
//             //and map_->exists() should return true for all other times
//             FrameNodePtrSet<Landmark> seen_frames = lmk_node->getSeenFrames();
//             for(const FrameNode3d::Ptr& seen_frame : seen_frames) {
//                 //check for sanity that the seen frames are <= than the current frame
//                 CHECK_LE(seen_frame->getId(), frame_id_k);
//                 const Landmark& measured = lmk_node->getMeasurement(seen_frame);
//                 new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//                     seen_frame->makePoseKey(), //pose key at previous frames
//                     point_key,
//                     measured,
//                     static_point_noise_
//                 );
//             }

//             //add initial value
//             //pick the one in this frame
//             const Landmark& measured = lmk_node->getMeasurement(frame_id_k);
//             const Landmark lmk_world = T_world_camera * measured;
//             new_values.insert(point_key, lmk_world);
//             timestamp_map_[point_key] = frame_id_k;

//             debug_info_.num_new_static_points++;

//         }
//     }

//     LOG(INFO) << "Num new static points: " << debug_info_.num_new_static_points << "\n" << "Num new static factors " << debug_info_.num_static_factors;
// }

// void RGBDBackendModule::updateDynamicObservations(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_point_factors) {
//     const FrameId frame_id_k_1 = frame_id_k - 1u;

//     const FrameNode3d::Ptr frame_node_k = map_->getFrame(frame_id_k);
//     const FrameNode3d::Ptr frame_node_k_1 = map_->getFrame(frame_id_k_1);
//     CHECK_NOTNULL(frame_node_k_1);

//     constexpr static size_t kMinObservations = 3u;

//     //for each object
//     for(const ObjectNode3d::Ptr& object_node : frame_node_k->objects_seen) {


//         DebugInfo::ObjectInfo object_debug_info;

//         const ObjectId object_id = object_node->getId();

//         //first check that object exists in the previous frame
//         if(!frame_node_k->objectMotionExpected(object_id)) {
//            continue;
//         }

//         // possibly the longest call?
//         // landmarks on this object seen at frame k
//         LandmarkNodePtrSet<Landmark> seen_lmks_k = object_node->getLandmarksSeenAtFrame(frame_id_k);

//         //if we dont have at least 3 observations of this object in this frame AND the previous frame
//         if(seen_lmks_k.size() < 3u || object_node->getLandmarksSeenAtFrame(frame_id_k_1).size() < 3) {
//             continue;
//         }

//         //TODO: debug
//         int num_lmks = 0;


//         //iterate over each lmk we have on this object
//         for(const LandmarkNode3d::Ptr& obj_lmk_node : seen_lmks_k) {
//             CHECK_EQ(obj_lmk_node->getObjectId(), object_id);

//             //see if we have enough observations to add this lmk
//             if(obj_lmk_node->numObservations() < kMinObservations) { continue;}

//             num_lmks++;

//             // if(num_lmks > 5) {
//             //     break;
//             // }

//             TrackletId tracklet_id = obj_lmk_node->getId();

//             // LOG(INFO) << "Tracklet exists " << is_dynamic_tracklet_in_map_.exists(tracklet_id);

//             //if does not exist, we need to go back and all the previous measurements & factors & motions
//             if(!is_dynamic_tracklet_in_map_.exists(tracklet_id)) {
//                 //could sanity check that no object motion exists for the previous frame?

//                 //add the points/motions from the past
//                 FrameNodePtrSet<Landmark> seen_frames = obj_lmk_node->getSeenFrames();
//                 // CHECK_EQ(frame_id_k - kMinObservations, seen_frames.getFirstIndex<FrameId>()) << container_to_string(seen_frames.collectIds<FrameId>());
//                 // CHECK_EQ(frame_id_k, seen_frames.getLastIndex<FrameId>()) <<  container_to_string(seen_frames.collectIds<FrameId>());
//                 //assert frame observations are continuous?

//                 //start at the first frame we want to start adding points in as we know have seen them enough times
//                 //start from +1, becuase the motion index is k-1 to k and there is no motion k-2 to k-1
//                 //but the first poitns we want to add are at k-1
//                 const FrameId starting_motion_frame = seen_frames.getFirstIndex<FrameId>() + 1u; //as we index the motion from k

//                 auto starting_motion_frame_itr = seen_frames.find(starting_motion_frame);
//                 CHECK(starting_motion_frame_itr != seen_frames.end());

//                 std::stringstream ss;
//                 ss << "Going back to add point on object " << object_id << " at frames\n";

//                 //iterate over k-N to k (inclusive) and all all
//                 for(auto seen_frames_itr = starting_motion_frame_itr; seen_frames_itr != seen_frames.end(); seen_frames_itr++) {
//                     auto seen_frames_itr_prev = seen_frames_itr;
//                     std::advance(seen_frames_itr_prev, -1);
//                     CHECK(seen_frames_itr_prev != seen_frames.end());

//                     FrameNode3d::Ptr query_frame_node_k = *seen_frames_itr;
//                     FrameNode3d::Ptr query_frame_node_k_1 = *seen_frames_itr_prev;


//                     CHECK_EQ(query_frame_node_k->frame_id, query_frame_node_k_1->frame_id + 1u);

//                     //add points UP TO AND INCLUDING the current frame
//                     if(query_frame_node_k->frame_id > frame_id_k) {
//                         break;
//                     }

//                     //point needs to be be in k and k-1 -> we have validated the object exists in these two frames
//                     //but not the points
//                     CHECK(obj_lmk_node->seenAtFrame(query_frame_node_k->frame_id));
//                     if(!obj_lmk_node->seenAtFrame(query_frame_node_k_1->frame_id)) {
//                         LOG(WARNING) << "Tracklet " << tracklet_id << " on object " << object_id << " seen at " << query_frame_node_k->frame_id << " but not " << query_frame_node_k_1->frame_id;
//                         break;
//                     } //this miay mean this this point never gets added?

//                     ss << query_frame_node_k_1->frame_id << " " << query_frame_node_k->frame_id << "\n";

//                     const Landmark& measured_k = obj_lmk_node->getMeasurement(query_frame_node_k);
//                     const Landmark& measured_k_1 = obj_lmk_node->getMeasurement(query_frame_node_k_1);

//                     //this should DEFINITELY be in the map, as long as we update the values in the map everyy time
//                     StateQuery<gtsam::Pose3> T_world_camera_k_1_query = query_frame_node_k_1->getPoseEstimate();
//                     CHECK(T_world_camera_k_1_query) << "Failed cam pose query at frame " << query_frame_node_k_1->frame_id
//                         << ". This may happen if the map_ is not updated every iteration OR something is wrong with the tracking...";
//                     const gtsam::Pose3 T_world_camera_k_1 = T_world_camera_k_1_query.get();

//                     StateQuery<gtsam::Pose3> T_world_camera_k_query = query_frame_node_k->getPoseEstimate();
//                     gtsam::Pose3 T_world_camera_k;
//                     //querying the map at the current frame ie. the last iteration in this loop as we havent added
//                     //the pose to the map yet!!
//                     if(!T_world_camera_k_query) {
//                         //this SHOULD happen when query_frame_node_k->frame_id == frame_id_k and at no other time
//                         CHECK_EQ(query_frame_node_k->frame_id, frame_id_k);
//                         //if this is okay, just set the pose to be the argument, since this should be the pose at frame_k
//                         T_world_camera_k = T_world_camera;
//                     }
//                     else {
//                         //we are not at the last frame of this loop yet so query_frame_node_k->frame_id < frame_id_k
//                         CHECK_LT(query_frame_node_k->frame_id, frame_id_k);
//                         T_world_camera_k = T_world_camera_k_query.get();
//                     }


//                     // const gtsam::Pose3 T_world_camera_k = T_world_camera_k_query.get();

//                     gtsam::Key object_point_key_k = obj_lmk_node->makeDynamicKey(query_frame_node_k->frame_id);
//                     gtsam::Key object_point_key_k_1 = obj_lmk_node->makeDynamicKey(query_frame_node_k_1->frame_id);

//                     //this assumes we add all the points in order and have continuous frames (which we should have?)
//                     if(seen_frames_itr == starting_motion_frame_itr) {
//                         //on first iteration we should add both values at query_k and query_k_1
//                         //otherwise we just need to add k, as k_1 will be added from the previous iteration
//                         new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//                             query_frame_node_k_1->makePoseKey(), //pose key at previous frames
//                             object_point_key_k_1,
//                             measured_k_1,
//                             dynamic_point_noise_
//                         );
//                         object_debug_info.num_dynamic_factors++;

//                         const Landmark lmk_world_k_1 = T_world_camera_k_1 * measured_k_1;
//                         new_values.insert(object_point_key_k_1, lmk_world_k_1);
//                         timestamp_map_[object_point_key_k_1] = query_frame_node_k_1->frame_id;
//                         object_debug_info.num_new_dynamic_points++;
//                     }

//                     //previous point must be added by the previous iteration
//                     CHECK(new_values.exists(object_point_key_k_1));

//                     new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//                         query_frame_node_k->makePoseKey(), //pose key at this (in the iteration) frames
//                         object_point_key_k,
//                         measured_k,
//                         dynamic_point_noise_
//                     );
//                     object_debug_info.num_dynamic_factors++;

//                     const Landmark lmk_world_k = T_world_camera_k * measured_k;
//                     new_values.insert(object_point_key_k, lmk_world_k);
//                     timestamp_map_[object_point_key_k] = query_frame_node_k->frame_id;
//                     object_debug_info.num_new_dynamic_points++;

//                     const gtsam::Key object_motion_key_k = query_frame_node_k->makeObjectMotionKey(object_id);
//                     //check in map as well, because this could be a new point on an old object, meaning the motion already exists
//                     if(!map_->exists(object_motion_key_k, new_values)) {
//                         //make new object motion
//                         new_values.insert(object_motion_key_k, gtsam::Pose3::Identity());
//                         //at time k or k-1 since a motion is between frames?
//                         timestamp_map_[object_motion_key_k] = query_frame_node_k->frame_id;
//                         LOG(INFO) << "Adding value " << DynoLikeKeyFormatterVerbose(object_motion_key_k);
//                     }

//                     new_point_factors.emplace_shared<LandmarkMotionTernaryFactor>(
//                         object_point_key_k_1,
//                         object_point_key_k,
//                         object_motion_key_k,
//                         landmark_motion_noise_
//                     );
//                     object_debug_info.num_motion_factors++;

//                     if(FLAGS_use_smoothing_factor) {
//                         //motion key at previous frame
//                         const gtsam::Symbol object_motion_key_k_1 = query_frame_node_k_1->makeObjectMotionKey(object_id);
//                         bool smoothing_added =
//                             safeAddObjectSmoothingFactor(object_motion_key_k_1, object_motion_key_k, new_values, new_point_factors);

//                         if(smoothing_added) {
//                             object_debug_info.smoothing_factor_added = true;
//                         }

//                     }


//                 }
//                 is_dynamic_tracklet_in_map_.insert2(tracklet_id, true);
//             }
//             else {
//                 //these tracklets should already be in the graph so we should only need to add the new measurements from this frame
//                 //check that we have previous point for this frame
//                 gtsam::Key object_point_key_k = obj_lmk_node->makeDynamicKey(frame_id_k);
//                 gtsam::Key object_point_key_k_1 = obj_lmk_node->makeDynamicKey(frame_id_k_1);

//                 const gtsam::Key object_motion_key_k = frame_node_k->makeObjectMotionKey(object_id);

//                 CHECK(map_->exists(object_point_key_k_1));
//                 CHECK(!map_->exists(object_point_key_k, new_values));

//                 const Landmark measured_k = obj_lmk_node->getMeasurement(frame_id_k);
//                 new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//                     frame_node_k->makePoseKey(), //pose key at previous frames
//                     object_point_key_k,
//                     measured_k,
//                     dynamic_point_noise_
//                 );
//                 object_debug_info.num_dynamic_factors++;

//                 new_point_factors.emplace_shared<LandmarkMotionTernaryFactor>(
//                     object_point_key_k_1,
//                     object_point_key_k,
//                     object_motion_key_k,
//                     landmark_motion_noise_
//                 );
//                 object_debug_info.num_motion_factors++;

//                 const Landmark lmk_world_k = T_world_camera * measured_k;
//                 new_values.insert(object_point_key_k, lmk_world_k);
//                 timestamp_map_[object_point_key_k] = frame_id_k;
//                 object_debug_info.num_new_dynamic_points++;

//                 //should not have to check the map because this is an old point
//                 if(!new_values.exists(object_motion_key_k)) {
//                     //make new object motion -> this can happen if all the points on the object are well tracked from the previous motion
//                     new_values.insert(object_motion_key_k, gtsam::Pose3::Identity());
//                     timestamp_map_[object_motion_key_k] = frame_id_k;
//                     LOG(INFO) << "Adding value " << DynoLikeKeyFormatterVerbose(object_motion_key_k);
//                 }

//                 //
//                 if(FLAGS_use_smoothing_factor && frame_node_k_1->objectObserved(object_id)) {
//                     //motion key at previous frame
//                     const gtsam::Symbol object_motion_key_k_1 = frame_node_k_1->makeObjectMotionKey(object_id);
//                     bool smoothing_added =
//                         safeAddObjectSmoothingFactor(object_motion_key_k_1, object_motion_key_k, new_values, new_point_factors);

//                     if(smoothing_added) {
//                         object_debug_info.smoothing_factor_added = true;
//                     }

//                 }
//             }
//         }

//         debug_info_.object_info.insert2(object_id, object_debug_info);

//     }

//     for(const auto&[object_id, object_info] : debug_info_.object_info) {
//         std::stringstream ss;
//         ss << "Object id debug info: " << object_id << "\n";
//         ss << "Num point factors: " << object_info.num_dynamic_factors << "\n";
//         ss << "Num point variables: " << object_info.num_new_dynamic_points << "\n";
//         ss << "Num motion factors: " << object_info.num_motion_factors << "\n";
//         ss << "Smoothing factor added: " << std::boolalpha <<  object_info.smoothing_factor_added << "\n";

//         LOG(INFO) << ss.str();
//     }
// }


// void RGBDBackendModule::optimize(FrameId frame_id_k, const gtsam::Values& new_values,  const gtsam::NonlinearFactorGraph& new_factors) {
//     DynoISAM2UpdateObservationParams isam_update_params;
//     // isam_update_params.

//     isam_update_params.setOrderingFunctions([=](const DynoISAM2UpdateObservationParams& updateParams,
//                                         const DynoISAM2Result& result,
//                                         const gtsam::KeySet& affectedKeysSet,
//                                         const gtsam::VariableIndex& affectedFactorsVarIndex) -> gtsam::Ordering {
//         // affectedFactorsVarIndex.print("VariableIndex: ", DynoLikeKeyFormatterVerbose);
//         gtsam::FastMap<gtsam::Key, int> constrainedKeys;
//         std::vector<std::pair<FrameId, gtsam::Key>> ordered_dynamic_lmks;
//         std::set<FrameId> dynamic_lmk_frames_set;

//         //points, motion, pose (in index order)

//         for(const gtsam::Key key : affectedKeysSet) {
//             auto chr = DynoChrExtractor(key);

//             if(chr == kDynamicLandmarkSymbolChar) {
//                 auto frame_id = DynamicPointSymbol(key).frameId();
//                 dynamic_lmk_frames_set.insert(frame_id);
//                 ordered_dynamic_lmks.push_back(std::make_pair(frame_id, key));
//             }
//         }

//         int max_group = dynamic_lmk_frames_set.size();

//         gtsam::FastMap<FrameId, int> frame_ordering;

//         for(const auto& frame_key : ordered_dynamic_lmks) {
//             int index = std::distance(dynamic_lmk_frames_set.begin(), dynamic_lmk_frames_set.find(frame_key.first));
//             // CHECK(index >= 0);
//             // //index tells use the order, starting at 0, with the first frame being what we want to have last in the ordering
//             // constrainedKeys.insert2(frame_key.second, index);
//             frame_ordering.insert2(frame_key.first, index);
//             // LOG(INFO) << "Adding group: " << index << " for key " << DynoLikeKeyFormatterVerbose(frame_key.second);
//         }

//         //last frame
//         const auto last_frame = frame_id_k;

//         //frame ordering
//         //frame_index * 3 (number of variabels we want to order; points, motion, pose) + offset
//         //ordering but by frame
//         // for(const gtsam::Key key : affectedKeysSet) {
//         //     auto chr = DynoChrExtractor(key);

//         //     int group = 0;

//         //     if(chr == kPoseSymbolChar) {
//         //         FrameId frame_id = gtsam::Symbol(key).index();
//         //         int frame_index = std::distance(dynamic_lmk_frames_set.begin(), dynamic_lmk_frames_set.find(frame_id));
//         //         //2 offset for pose
//         //         group = frame_index * 3 + 2;
//         //     }
//         //     else if(chr == kObjectMotionSymbolChar) {
//         //         ObjectId object_label;
//         //         FrameId frame_id;
//         //         CHECK(reconstructMotionInfo(key, object_label, frame_id));
//         //         int frame_index = std::distance(dynamic_lmk_frames_set.begin(), dynamic_lmk_frames_set.find(frame_id));
//         //         //1 offset for motion
//         //         group = frame_index * 3 + 0;
//         //     }
//         //     else if(chr == kDynamicLandmarkSymbolChar) {
//         //         auto frame_id = DynamicPointSymbol(key).frameId();
//         //         int frame_index = std::distance(dynamic_lmk_frames_set.begin(), dynamic_lmk_frames_set.find(frame_id));
//         //         //0 offset for points
//         //         group = frame_index * 3 + 1;
//         //     }

//         //     LOG(INFO) << "Adding group: " << group << " for key " << DynoLikeKeyFormatterVerbose(key);
//         //     constrainedKeys.insert2(key, group);
//         // }

//         //dont assign static, so start groups at 1
//         //eliminate static, then dynamic points BUT NOT THE LAST POINT, then motions, then the last dynamic points, then poses
//         //IDEA: eliminate all the dynamic points before the motions (except the alst ones)
//         //so that we can write the motions in terms of the poses?
//         // for(const gtsam::Key key : affectedKeysSet) {
//         //     auto chr = DynoChrExtractor(key);

//         //     int group = 0;

//         //     if(chr == kPoseSymbolChar) {
//         //         FrameId frame_id = gtsam::Symbol(key).index();
//         //         if(frame_id == last_frame) {
//         //             group = 4; //put alst
//         //         }
//         //         else {
//         //             group = 0; //put with the other non-last dynamic points?
//         //         }
//         //     }
//         //     else if(chr == kObjectMotionSymbolChar) {
//         //         ObjectId object_label;
//         //         FrameId frame_id;
//         //         CHECK(reconstructMotionInfo(key, object_label, frame_id));
//         //         if(frame_id == last_frame) {
//         //             //I think this should be after the last dynamic point key...
//         //             group = 3; //put last
//         //         }
//         //         else {
//         //             //put as early as possible so we can try and marginalize it out!!
//         //             group = 0;
//         //         }

//         //     }
//         //     else if(chr == kDynamicLandmarkSymbolChar) {
//         //         auto frame_id = DynamicPointSymbol(key).frameId();

//         //         if(frame_id == last_frame) {
//         //             group = 4;
//         //         }
//         //         else {
//         //             group = 0;
//         //         }

//         //     }
//         //     // else if(chr == kStaticLandmarkSymbolChar) {


//         //     // }

//         //     LOG(INFO) << "Adding group: " << group << " for key " << DynoLikeKeyFormatterVerbose(key);
//         //     constrainedKeys.insert2(key, group);
//         // }



//         return Ordering::ColamdConstrained(affectedFactorsVarIndex, constrainedKeys);
//     });

//     //create an ordering putting motions last?
//     gtsam::FastMap<gtsam::Key, int> constrainedKeys;
//     //for each object

//     // for(const auto& [keys, value] : new_values) {
//     //     constrainedKeys.insert2(keys, 1);
//     // }

//     //very slow - find motion factors index's and mark them as newly affected keys
//     // gtsam::NonlinearFactorGraph old_graph = smoother_->getFactorsUnsafe();

//     // gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet> affected_motion_factors;
//     // for(size_t i = 0; i < old_graph.size(); i++) {
//     //     auto factor = old_graph.at(i);
//     //     auto motion_factor = boost::dynamic_pointer_cast<LandmarkMotionTernaryFactor>(
//     //         factor;
//     //     );

//     //     if(motion_factor) {
//     //         //check if it contains a new key
//     //         if(motion_factor.find())
//     //     }

//     // }

//     //allows keys to be re-ordered
//     gtsam::FastList<gtsam::Key> extraReelimKeys;
//     // gtsam::KeyVector marginalize_keys;

//     // std::vector<std::pair<FrameId, gtsam::Key>> ordered_dynamic_lmks;
//     // std::set<FrameId> dynamic_lmk_frames_set;

//     // for(const gtsam::Key key : new_values.keys()) {
//     //     auto chr = DynoChrExtractor(key);

//     //     if(chr == kDynamicLandmarkSymbolChar) {
//     //         auto frame_id = DynamicPointSymbol(key).frameId();
//     //         dynamic_lmk_frames_set.insert(frame_id);
//     //         ordered_dynamic_lmks.push_back(std::make_pair(frame_id, key));
//     //     }
//     // }

//     // int max_group = dynamic_lmk_frames_set.size();

//     // for(const auto& frame_key : ordered_dynamic_lmks) {
//     //     int index = std::distance(dynamic_lmk_frames_set.begin(), dynamic_lmk_frames_set.find(frame_key.first));
//     //     CHECK(index >= 0);
//     //     //index tells use the order, starting at 0, with the first frame being what we want to have last in the ordering
//     //     constrainedKeys.insert2(frame_key.second, index);
//     //     LOG(INFO) << "Adding group: " << index << " for key " << DynoLikeKeyFormatterVerbose(frame_key.second);
//     // }

//     // for(const gtsam::Key key : new_values.keys()) {
//     //     auto chr = DynoChrExtractor(key);

//     //     if(chr == kPoseSymbolChar || chr == kObjectMotionSymbolChar) {
//     //         constrainedKeys.insert2(key, max_group);
//     //         LOG(INFO) << "Adding group: " << max_group << " for key " << DynoLikeKeyFormatterVerbose(key);
//     //     }
//     // }

//     // // //eliminate motions last? (group < dynamic key)
//     // const FrameNode3d::Ptr frame_node_k = map_->getFrame(frame_id_k);
//     // const FrameNode3d::Ptr frame_node_k_1 = map_->getFrame(frame_id_k - 1u);
//     // for(const ObjectNode3d::Ptr& object_node : frame_node_k->objects_seen) {
//     //     const gtsam::Key object_motion_key_k = frame_node_k->makeObjectMotionKey(object_node->getId());

//     //     if(new_values.exists(object_motion_key_k)) {
//     //         constrainedKeys[object_motion_key_k] = 1;
//     //     }

//     //     // if(frame_node_k_1) {
//     //     //     const gtsam::Key object_motion_key_k_1 = frame_node_k_1->makeObjectMotionKey(object_node->getId());
//     //     //     if(map_->exists(object_motion_key_k_1)) {
//     //     //         constrainedKeys.insert2(object_motion_key_k_1, 0);
//     //     //         marginalize_keys.push_back(object_motion_key_k_1);
//     //     //     }

//     //     // }
//     // }

//     // for(const auto& dyn_lmk_node : frame_node_k->dynamic_landmarks) {
//     //     const gtsam::Key key = dyn_lmk_node->makeDynamicKey(frame_node_k->getId());


//     //     //eliminate previous keys first
//     //     if(frame_id_k > 1) {
//     //         const gtsam::Key previous_key = dyn_lmk_node->makeDynamicKey(frame_node_k->getId() - 1);
//     //         // if(map_->exists(previous_key)) {
//     //             //should be in map but right now we dont have a way of checking if the keys have been deleted in the map
//     //             //once we get an estimate update from the smoother
//     //         // if(smoother_->valueExists(previous_key)) {
//     //         //     //have to put these keys are the start
//     //         //     constrainedKeys.insert2(previous_key, 0);
//     //         //     old_dynamic_points.push_back(previous_key);
//     //         //     // extraReelimKeys.push_back(previous_key);
//     //         // }
//     //     }


//     //     // if(new_values.exists(key)) {
//     //     //     constrainedKeys.insert2(key, 1);
//     //     // }
//     // }

//     // //further up the tree than the other two?
//     // constrainedKeys.insert2(CameraPoseSymbol(frame_id_k), 1);

//     // if(frame_id_k > 1) {
//     //     constrainedKeys.insert2(CameraPoseSymbol(frame_id_k -1), 0);
//     // }
//     // // gtsam::FastList<gtsam::Key> no_relin_keys;

//     // //make sure only the poses between the last motions are reliminated
//     // if((int)frame_id_k - 1 > 1) {
//     //     for(FrameId f = 0; f < frame_id_k - 1; f++) {
//     //         no_relin_keys.push_back(CameraPoseSymbol(f));
//     //     }
//     // }
//     // std::cout << "No relin keys: ";
//     // for (const auto key : no_relin_keys) {
//     //     std::cout << DynoLikeKeyFormatter(key) << " ";
//     // }
//     // std::cout << std::endl;


//     // Mark additional keys between the constrainted keys and the leaves
//     //from fixed lag smoother
//     std::function<void(const gtsam::Key&,const gtsam::ISAM2Clique::shared_ptr&, std::set<gtsam::Key>&)> recursiveMarkAffectedKeys =
//     [&recursiveMarkAffectedKeys](const gtsam::Key& key, const gtsam::ISAM2Clique::shared_ptr& clique, std::set<gtsam::Key>& additionalKeys) -> void{
//             // Check if the separator keys of the current clique contain the specified key
//         if (std::find(clique->conditional()->beginParents(),
//             clique->conditional()->endParents(), key)
//             != clique->conditional()->endParents()) {

//             // Mark the frontal keys of the current clique
//             for(gtsam::Key i: clique->conditional()->frontals()) {
//                 additionalKeys.insert(i);
//             }

//             // Recursively mark all of the children
//             for(const gtsam::ISAM2Clique::shared_ptr& child: clique->children) {
//                 recursiveMarkAffectedKeys(key, child, additionalKeys);
//             }
//         }
//         // If the key was not found in the separator/parents, then none of its children can have it either
//     };

//     //these keys should already be in the smoother and should now be at the leaves of the tree
//     // std::set<gtsam::Key> additionalKeys;
//     // for(gtsam::Key key : marginalize_keys) {
//     //     gtsam::ISAM2Clique::shared_ptr clique = smoother_->operator[](key);
//     //     for(const gtsam::ISAM2Clique::shared_ptr& child: clique->children) {
//     //         recursiveMarkAffectedKeys(key, child, additionalKeys);
//     //     }
//     // }
//     // gtsam::KeyList additionalMarkedKeys(additionalKeys.begin(), additionalKeys.end());

//     // isam_update_params.constrainedKeys = constrainedKeys;
//     // isam_update_params.extraReelimKeys = additionalMarkedKeys;
//     // isam_update_params.
//     // isam_update_params.noRelinKeys = no_relin_keys;

//     LOG(INFO) << "Starting optimization for " << frame_id_k;

//     gtsam::Values old_state = map_->getValues();
//     gtsam::Values all_values = old_state;
//     all_values.insert_or_assign(new_values);

//     try {
//         // gtsam::IncrementalFixedLagSmoother::Result fl_result = smoother_->update(new_factors, new_values, timestamp_map_);
//         // gtsam::ISAM2Result result = smoother_->getISAM2Result();
//         utils::TimingStatsCollector("backend.update");
//         smoother_result_ = smoother_->update(new_factors, new_values, isam_update_params);
//         // result = smoother_->update();
//         // result = smoother_->update();

//         // smoother_->print("isam2 ", DynoLikeKeyFormatter);
//         smoother_result_.print();
//         LOG(INFO) << "Num total vars: " << all_values.size() << " num new vars " << new_values.size();
//         //keys that were part of the bayes tree that is removed before conversion to a factor graph
//         LOG(INFO) << "Num marked keys: " << smoother_result_.markedKeys.size();
//         LOG(INFO) << "Num total factors: " << smoother_->getFactorsUnsafe().size() << " num new factors " << new_factors.size();
//         // LOG(INFO) << "Num total factors: " << smoother_->getFactors().size() << " num new factors " << new_factors.size();

//         // auto detailed_results = result.details();
//         // if(detailed_results) {
//         //     std::stringstream ss;
//         //     ss << "Keys re-eliminated: ";
//         //     for(const auto& [key, variable_status] : detailed_results->variableStatus) {
//         //         if(variable_status.isReeliminated) {
//         //             ss << DynoLikeKeyFormatter(key) << " ";
//         //         }
//         //     }

//         //     LOG(INFO) << ss.str();
//         // }

//         // bool did_batch = result.
//     }
//     catch(const gtsam::ValuesKeyAlreadyExists& e) {
//         new_factors.saveGraph(getOutputFilePath("isam2_graph.dot"), DynoLikeKeyFormatter);
//         LOG(FATAL) << "gtsam::ValuesKeyAlreadyExists thrown near key " << DynoLikeKeyFormatterVerbose(e.key());
//     }
//     catch(const gtsam::ValuesKeyDoesNotExist& e) {
//         new_factors.saveGraph(getOutputFilePath("isam2_graph.dot"), DynoLikeKeyFormatter);
//         LOG(FATAL) << "gtsam::ValuesKeyDoesNotExist thrown near key " << DynoLikeKeyFormatterVerbose(e.key());
//     }

//     timestamp_map_.clear();

//     gtsam::KeySet marginalize_keys = factor_graph_tools::travsersal::getLeafKeys(*smoother_);
//     gtsam::PrintKeySet(marginalize_keys, "Marginalize keys", dyno::DynoLikeKeyFormatter);


//     gtsam::Values estimate = smoother_->calculateBestEstimate();
//     gtsam::NonlinearFactorGraph graph = smoother_->getFactorsUnsafe();

//     LOG(INFO) << "Starting marginalization...";
//     // Marginalize out any needed variables
//     //only works in certain ordering situations.....
//     //does marginalization reuslt in null ptrs in the graph?
//     if (marginalize_keys.size() > 1) {
//         gtsam::FastList<gtsam::Key> leafKeys(marginalize_keys.begin(),
//             marginalize_keys.end());
//         //remove poses?
//         auto is_pose = [](auto key) {
//             auto chr = DynoChrExtractor(key);
//             return chr == kPoseSymbolChar || chr == kStaticLandmarkSymbolChar;
//             // return false;
//         };
//         // leafKeys.erase(std::remove_if(leafKeys.begin(), leafKeys.end(), is_pose), leafKeys.end());

//         // smoother_->marginalizeLeaves(leafKeys);
//     }

//     LOG(INFO) << "finished marginalization...";

//     // gtsam::Values estimate = smoother_->calculateBestEstimate();
//     // gtsam::NonlinearFactorGraph graph = smoother_->getFactorsUnsafe();
//     // gtsam::Values all_values = smoother_->

//     double error_before = graph.error(all_values);
//     double error_after = graph.error(estimate);

//     LOG(INFO) << "Optimization Errors:\n"
//              << " - Error before :" << error_before
//              << '\n'
//              << " - Error after  :" << error_after;

//     map_->updateEstimates(estimate, graph, frame_id_k);
//     // map_->updateEstimates(all_values, graph, frame_id_k);
// }


// void RGBDBackendModule::UpdateImplInWorld::logBackendFromMap(FrameId frame_k, RGBDMap::Ptr map, BackendLogger& logger) {
//     const auto& gt_packet_map = parent_->gt_packet_map_;

//     const ObjectIds object_ids = map->getObjectIds();
//     ObjectPoseMap composed_object_pose_map;

//     for(auto object_id : object_ids) {
//         const auto& object_node = map->getObject(object_id);
//         composed_object_pose_map.insert2(
//             //composed poses
//             object_id, object_node->computeComposedPoseMap(gt_packet_map, FLAGS_init_object_pose_from_gt));
//     }

//     //get MotionestimateMap
//     const MotionEstimateMap motions = map->getMotionEstimates(frame_k);
//     logger.logObjectMotion(gt_packet_map, frame_k, motions);

//     StateQuery<gtsam::Pose3> X_k_query = map->getPoseEstimate(frame_k);

//     if(X_k_query) {
//         logger.logCameraPose(gt_packet_map, frame_k, X_k_query.get());
//     }
//     else {
//         LOG(WARNING) << "Could not log camera pose estimate at frame " << frame_k;
//     }


//     logger.logObjectPose(gt_packet_map, frame_k, composed_object_pose_map);


//     if(map->frameExists(frame_k)) {
//         StatusLandmarkEstimates static_map = map->getStaticMap(frame_k);
//         // LOG(INFO) << "static map size " << static_map.size();
//         StatusLandmarkEstimates dynamic_map = map->getDynamicMap(frame_k);
//         // LOG(INFO) << "dynamic map size " << dynamic_map.size();

//         CHECK(X_k_query); //actually not needed for points in world!!
//         logger.logPoints(frame_k, *X_k_query, static_map);
//         logger.logPoints(frame_k, *X_k_query, dynamic_map);
//     }





// }





//TODO: these functions can go in base
void RGBDBackendModule::saveGraph(const std::string& file) {

    //TODO: must be careful as there could be inconsistencies between the graph in the optimzier,
    //and the graph in the map
    //TODO:
    // gtsam::NonlinearFactorGraph graph = map_->getGraph();
    // // gtsam::NonlinearFactorGraph graph = smoother_->getFactorsUnsafe();
    // graph.saveGraph(getOutputFilePath(file), DynoLikeKeyFormatter);
}
void RGBDBackendModule::saveTree(const std::string& file) {
    // auto incremental_optimizer = safeCast<Optimizer<Landmark>, IncrementalOptimizer<Landmark>>(optimizer_);
    // if(incremental_optimizer) {
    //     auto smoother = incremental_optimizer->getSmoother();
    //     smoother.saveGraph(getOutputFilePath(file), DynoLikeKeyFormatter);
    //     // smoother.getISAM2().saveGraph(getOutputFilePath(file), DynoLikeKeyFormatter);
    // }
}



} //dyno
