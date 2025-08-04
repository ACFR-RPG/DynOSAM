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

#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/utils/TimingStats.hpp"

namespace dyno {

template <typename MAP>
Formulation<MAP>::Formulation(const FormulationParams& params,
                              typename Map::Ptr map,
                              const NoiseModels& noise_models,
                              const FormulationHooks& hooks)
    : params_(params), map_(map), noise_models_(noise_models), hooks_(hooks) {}

template <typename MAP>
void Formulation<MAP>::setTheta(const gtsam::Values& linearization) {
  theta_ = linearization;
  // TODO: comment backed in!!
  //  accessorFromTheta()->postUpdateCallback();
}

template <typename MAP>
void Formulation<MAP>::updateTheta(const gtsam::Values& linearization) {
  // theta_.update(linearization);
  // why would we need to assign new values?
  theta_.insert_or_assign(linearization);
  // TODO: comment backed in!!
  //  accessorFromTheta()->postUpdateCallback();
}

template <typename MAP>
gtsam::Pose3 Formulation<MAP>::getInitialOrLinearizedSensorPose(
    FrameId frame_id) const {
  const auto accessor = this->accessorFromTheta();
  // sensor pose from a previous/current linearisation point
  StateQuery<gtsam::Pose3> X_k_theta = accessor->getSensorPose(frame_id);

  gtsam::Pose3 X_k_initial;
  CHECK(this->map()->hasInitialSensorPose(frame_id, &X_k_initial));
  // take either the query value from the map (if we have a previous
  // initalisation), or the estimate from the camera
  gtsam::Pose3 X_k;
  getSafeQuery(X_k, X_k_theta, X_k_initial);
  return X_k;
}

template <typename MAP>
BackendLogger::UniquePtr Formulation<MAP>::makeFullyQualifiedLogger() const {
  return std::make_unique<BackendLogger>(getFullyQualifiedName());
}

template <typename MAP>
void Formulation<MAP>::addSensorPoseValue(const gtsam::Pose3& X_W_k,
                                          FrameId frame_id_k,
                                          gtsam::Values& new_values) {
  gtsam::Values values;
  values.insert(CameraPoseSymbol(frame_id_k), X_W_k);

  new_values.insert_or_assign(values);
  theta_.insert_or_assign(values);
}

template <typename MAP>
void Formulation<MAP>::addValuesFunctional(
    std::function<void(gtsam::Values&)> callback, gtsam::Values& new_values) {
  gtsam::Values internal_new_values;
  callback(internal_new_values);

  new_values.insert_or_assign(internal_new_values);
  theta_.insert_or_assign(internal_new_values);
}

template <typename MAP>
void Formulation<MAP>::addFactorsFunctional(
    std::function<void(gtsam::NonlinearFactorGraph&)> callback,
    gtsam::NonlinearFactorGraph& new_factors) {
  gtsam::NonlinearFactorGraph internal_new_factors;
  callback(internal_new_factors);

  new_factors += internal_new_factors;
  factors_ += internal_new_factors;
}

template <typename MAP>
void Formulation<MAP>::addSensorPosePriorFactor(
    const gtsam::Pose3& X_W_k, gtsam::SharedNoiseModel noise_model,
    FrameId frame_id_k, gtsam::NonlinearFactorGraph& new_factors) {
  // keep track of the new factors added in this function
  // these are then appended to the internal factors_ and new_factors
  gtsam::NonlinearFactorGraph internal_new_factors;
  internal_new_factors.addPrior(CameraPoseSymbol(frame_id_k), X_W_k,
                                noise_model);
  new_factors += internal_new_factors;
  factors_ += internal_new_factors;
}

template <typename MAP>
void Formulation<MAP>::setInitialPose(const gtsam::Pose3& T_world_camera,
                                      FrameId frame_id_k,
                                      gtsam::Values& new_values) {
  this->addSensorPoseValue(T_world_camera, frame_id_k, new_values);
}

template <typename MAP>
void Formulation<MAP>::setInitialPosePrior(
    const gtsam::Pose3& T_world_camera, FrameId frame_id_k,
    gtsam::NonlinearFactorGraph& new_factors) {
  auto initial_pose_prior = noise_models_.initial_pose_prior;
  this->addSensorPosePriorFactor(T_world_camera, initial_pose_prior, frame_id_k,
                                 new_factors);
}

template <typename MAP>
UpdateObservationResult Formulation<MAP>::updateStaticObservations(
    FrameId frame_id_k, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors,
    const UpdateObservationParams& update_params) {
  typename Map::Ptr map = this->map();
  auto accessor = this->accessorFromTheta();

  auto static_point_noise = CHECK_NOTNULL(noise_models_.static_point_noise);

  gtsam::SharedNoiseModel static_keypoint_noise =
      gtsam::noiseModel::Isotropic::Sigma(2u, 3);
  static_keypoint_noise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(0.01),
      static_keypoint_noise);
  // keep track of the new factors added in this function
  // these are then appended to the internal factors_ and new_factors
  gtsam::NonlinearFactorGraph internal_new_factors;

  UpdateObservationResult result(update_params);

  // // VIODE!!
  // double fx = 376.0;
  // double fy = 376.0;
  // double cx = 376.0;
  // double cy = 240.0;
  // double skew = 0.0;  // assuming zero skew

  // const auto K = boost::make_shared<gtsam::Cal3_S2>(fx, fy, skew, cx, cy);

  const auto frame_node_k = map->getFrame(frame_id_k);
  CHECK_NOTNULL(frame_node_k);

  // pose estimate from frontend
  gtsam::Pose3 T_world_camera_frontend;
  CHECK(
      this->map()->hasInitialSensorPose(frame_id_k, &T_world_camera_frontend));

  VLOG(20) << "Looping over " << frame_node_k->static_landmarks.size()
           << " static lmks for frame " << frame_id_k;
  for (const auto& lmk_node : frame_node_k->static_landmarks) {
    const gtsam::Key point_key = lmk_node->makeStaticKey();
    // check if lmk node is already in map (which should mean it is equivalently
    // in isam)
    if (is_other_values_in_map.exists(point_key)) {
      Landmark measured_point_local;
      gtsam::SharedNoiseModel measurement_covariance;
      std::tie(measured_point_local, measurement_covariance) =
          MeasurementTraits::pointWithCovariance(
              lmk_node->getMeasurement(frame_id_k));
      CHECK(measurement_covariance);

      if (params_.makeStaticMeasurementsRobust()) {
        measurement_covariance = factor_graph_tools::robustifyHuber(
            params_.k_huber_3d_points_, measurement_covariance);
      }

      internal_new_factors
          .emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
              frame_node_k->makePoseKey(),  // pose key for this frame
              point_key, measured_point_local, measurement_covariance);

      if (result.debug_info) result.debug_info->num_static_factors++;
      result.updateAffectedObject(frame_id_k, 0);

    } else {
      // see if we have enough observations to add this lmk
      if (lmk_node->numObservations() < params_.min_static_observations) {
        continue;
      }

      // this condition should only run once per tracklet (ie.e the first time
      // the tracklet has enough observations) we gather the tracklet
      // observations and then initalise it in the new values these should then
      // get added to the map and map_->exists() should return true for all
      // other times
      const auto& seen_frames = lmk_node->getSeenFrames();
      for (const auto& seen_frame : seen_frames) {
        auto seen_frame_id = seen_frame->getId();
        // only iterate up to the query frame
        if ((FrameId)seen_frame_id > frame_id_k) {
          break;
        }

        // if we should not backtrack, only add the current frame!!!
        const auto do_backtrack = update_params.do_backtrack;
        if (!do_backtrack && (FrameId)seen_frame_id < frame_id_k) {
          continue;
        }

        Landmark measured_point_local;
        gtsam::SharedNoiseModel measurement_covariance;
        std::tie(measured_point_local, measurement_covariance) =
            MeasurementTraits::pointWithCovariance(
                lmk_node->getMeasurement(frame_id_k));
        CHECK(measurement_covariance);

        if (params_.makeStaticMeasurementsRobust()) {
          measurement_covariance = factor_graph_tools::robustifyHuber(
              params_.k_huber_3d_points_, measurement_covariance);
        }

        internal_new_factors.emplace_shared<PoseToPointFactor>(
            seen_frame->makePoseKey(),  // pose key at previous frames
            point_key, measured_point_local, measurement_covariance);

        if (result.debug_info) result.debug_info->num_static_factors++;
        result.updateAffectedObject(seen_frame_id, 0);
      }

      // pick the one in this frame
      const Landmark& measured =
          MeasurementTraits::point(lmk_node->getMeasurement(frame_id_k));
      // add initial value, either from measurement or previous estimate
      gtsam::Point3 lmk_world;
      getSafeQuery(lmk_world,
                   accessor->getStaticLandmark(lmk_node->tracklet_id),
                   gtsam::Point3(T_world_camera_frontend * measured));
      new_values.insert(point_key, lmk_world);
      is_other_values_in_map.insert2(point_key, true);

      if (result.debug_info) result.debug_info->num_new_static_points++;
      result.updateAffectedObject(frame_id_k, 0);
    }
  }

  // update internal data structures
  theta_.insert_or_assign(new_values);
  factors_ += internal_new_factors;
  new_factors += internal_new_factors;

  if (result.debug_info)
    LOG(INFO) << "Num new static points: "
              << result.debug_info->num_new_static_points
              << " Num new static factors "
              << result.debug_info->num_static_factors;
  return result;
}

template <typename MAP>
UpdateObservationResult Formulation<MAP>::updateDynamicObservations(
    FrameId frame_id_k, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors,
    const UpdateObservationParams& update_params) {
  typename Map::Ptr map = this->map();
  auto accessor = this->accessorFromTheta();
  // collect noise models to be used
  auto landmark_motion_noise = noise_models_.landmark_motion_noise;
  auto dynamic_point_noise = noise_models_.dynamic_point_noise;
  // keep track of the new factors added in this function
  // these are then appended to the internal factors_ and new_factors
  gtsam::NonlinearFactorGraph internal_new_factors;
  // keep track of the new values added in this function
  // these are then appended to the internal values_ and new_values
  gtsam::Values internal_new_values;

  UpdateObservationResult result(update_params);

  // starting slot number is size of new factors
  // as long as the new factor slot is calculated before adding a new factor
  const Slot starting_factor_slot = new_factors.size();

  const FrameId frame_id_k_1 = frame_id_k - 1u;
  VLOG(20) << "Add dynamic observations between frames " << frame_id_k_1
           << " and " << frame_id_k;
  const auto frame_node_k = map->getFrame(frame_id_k);
  const auto frame_node_k_1 = map->getFrame(frame_id_k_1);
  CHECK_NOTNULL(frame_node_k_1);

  // pose estimate from frontend
  // CHECK(parent_->initial_camera_poses_.exists(frame_id_k));
  // const gtsam::Pose3& T_world_camera_initial_k =
  // new_values.at<gtsam::Pose3>(CameraPoseSymbol(frame_id_k));

  utils::TimingStatsCollector dyn_obj_itr_timer(this->loggerPrefix() +
                                                ".dynamic_object_itr");
  for (const auto& object_node : frame_node_k->objects_seen) {
    DebugInfo::ObjectInfo object_debug_info;
    const ObjectId object_id = object_node->getId();

    // first check that object exists in the previous frame
    if (!frame_node_k->objectMotionExpected(object_id)) {
      continue;
    }
    // possibly the longest call?
    // landmarks on this object seen at frame k
    auto seen_lmks_k = object_node->getLandmarksSeenAtFrame(frame_id_k);

    // if we dont have at least N observations of this object in this frame AND
    // the previous frame
    if (seen_lmks_k.size() < params_.min_dynamic_observations ||
        object_node->getLandmarksSeenAtFrame(frame_id_k_1).size() <
            params_.min_dynamic_observations) {
      continue;
    }

    utils::TimingStatsCollector dyn_point_itr_timer(this->loggerPrefix() +
                                                    ".dynamic_point_itr");
    VLOG(10) << "Seen lmks at frame " << frame_id_k << " obj " << object_id
             << ": " << seen_lmks_k.size();
    // iterate over each lmk we have on this object
    for (const auto& obj_lmk_node : seen_lmks_k) {
      CHECK_EQ(obj_lmk_node->getObjectId(), object_id);

      // see if we have enough observations to add this lmk
      if (obj_lmk_node->numObservations() < params_.min_dynamic_observations) {
        continue;
      }

      TrackletId tracklet_id = obj_lmk_node->getId();

      // if does not exist, we need to go back and all the previous measurements
      // & factors & motions
      if (!isDynamicTrackletInMap(obj_lmk_node)) {
        // add the points/motions from the past
        auto seen_frames = obj_lmk_node->getSeenFrames();
        // assert frame observations are continuous?

        // start at the first frame we want to start adding points in as we know
        // have seen them enough times start from +1, becuase the motion index
        // is k-1 to k and there is no motion k-2 to k-1 but the first poitns we
        // want to add are at k-1
        // TODO: shouldnt this actually just be frame_id_k -
        // params_.min_dynamic_observations
        // TODO: how to handle the if starting_motion_frame < a frame actually
        // seen in!
        FrameId starting_motion_frame;
        if (update_params.do_backtrack) {
          starting_motion_frame =
              seen_frames.template getFirstIndex<FrameId>() +
              1u;  // as we index the motion from k
        } else {
          // start from the requested index, this will still mean that we will
          // add the previous frame as always add a motion between k-1 and k
          starting_motion_frame = frame_id_k;

          if (starting_motion_frame <
              seen_frames.template getFirstIndex<FrameId>() + 1u) {
            // if the requested starting frame is not the first frame + 1u of
            // the actul track (ie. the second seen frame) we cannot use it yet
            // as we have to index BACKWARDS from the starting motion frame if
            // we used frame_id_k as the starting_motion_frame, we would end up
            // with an iterator pointing to the end instead, we will have to get
            // it next frame!!!
            continue;
          }
        }

        auto starting_motion_frame_itr =
            seen_frames.find(starting_motion_frame);
        CHECK(starting_motion_frame_itr != seen_frames.end())
            << "Starting motion frame is " << starting_motion_frame
            << " but first frame is "
            << seen_frames.template getFirstIndex<FrameId>();

        std::stringstream ss;
        ss << "Going back to add point on object " << object_id
           << " at frames\n";

        // iterate over k-N to k (inclusive) and all all
        utils::TimingStatsCollector dyn_point_backtrack_timer(
            this->loggerPrefix() + ".dynamic_point_backtrack");
        for (auto seen_frames_itr = starting_motion_frame_itr;
             seen_frames_itr != seen_frames.end(); seen_frames_itr++) {
          auto seen_frames_itr_prev = seen_frames_itr;
          std::advance(seen_frames_itr_prev, -1);
          CHECK(seen_frames_itr_prev != seen_frames.end())
              << " For object  " << object_id;

          auto query_frame_node_k = *seen_frames_itr;
          auto query_frame_node_k_1 = *seen_frames_itr_prev;

          CHECK_EQ(query_frame_node_k->frame_id,
                   query_frame_node_k_1->frame_id + 1u);

          // add points UP TO AND INCLUDING the current frame
          if (query_frame_node_k->frame_id > frame_id_k) {
            break;
          }

          // point needs to be be in k and k-1 -> we have validated the object
          // exists in these two frames but not the points
          CHECK(obj_lmk_node->seenAtFrame(query_frame_node_k->frame_id));
          if (!obj_lmk_node->seenAtFrame(query_frame_node_k_1->frame_id)) {
            LOG(WARNING) << "Tracklet " << tracklet_id << " on object "
                         << object_id << " seen at "
                         << query_frame_node_k->frame_id << " but not "
                         << query_frame_node_k_1->frame_id;
            break;
          }  // this miay mean this this point never gets added?

          ss << query_frame_node_k_1->frame_id << " "
             << query_frame_node_k->frame_id << "\n";

          const gtsam::Pose3 T_world_camera_k_1 =
              getInitialOrLinearizedSensorPose(query_frame_node_k_1->frame_id);

          gtsam::Pose3 T_world_camera_k =
              getInitialOrLinearizedSensorPose(query_frame_node_k->frame_id);

          PointUpdateContextType point_context;
          point_context.lmk_node = obj_lmk_node;
          point_context.frame_node_k_1 = query_frame_node_k_1;
          point_context.frame_node_k = query_frame_node_k;
          point_context.X_k_measured = T_world_camera_k;
          point_context.X_k_1_measured = T_world_camera_k_1;
          point_context.starting_factor_slot = starting_factor_slot;

          // this assumes we add all the points in order and have continuous
          // frames (which we should have?)
          if (seen_frames_itr == starting_motion_frame_itr) {
            point_context.is_starting_motion_frame = true;
          }
          utils::TimingStatsCollector dyn_point_update_timer(
              this->loggerPrefix() + ".dyn_point_update_1");
          // the true set of values that are added from the update
          dynamicPointUpdateCallback(point_context, result, internal_new_values,
                                     internal_new_factors);
          // // update internal theta and factors
          // theta_.insert(local_new_values);
          // // add to the external new_values
          // new_values.insert(local_new_values);
        }
      } else {
        // these tracklets should already be in the graph so we should only need
        // to add the new measurements from this frame check that we have
        // previous point for this frame
        PointUpdateContextType point_context;
        point_context.lmk_node = obj_lmk_node;
        point_context.frame_node_k_1 = frame_node_k_1;
        point_context.frame_node_k = frame_node_k;
        point_context.X_k_1_measured =
            getInitialOrLinearizedSensorPose(frame_node_k_1->frame_id);
        point_context.X_k_measured =
            getInitialOrLinearizedSensorPose(frame_node_k->frame_id);
        point_context.starting_factor_slot = starting_factor_slot;
        point_context.is_starting_motion_frame = false;
        utils::TimingStatsCollector dyn_point_update_timer(
            this->loggerPrefix() + ".dyn_point_update_2");
        // the true set of values that are added from the update
        // gtsam::Values local_new_values;
        dynamicPointUpdateCallback(point_context, result, internal_new_values,
                                   internal_new_factors);
        // // update internal theta and factors
        // theta_.insert(local_new_values);
        // // add to the external new_values
        // new_values.insert(local_new_values);
      }
    }
  }
  // iterate over objects for which a motion was added
  // becuase we add lots of new points every frame, we may go over the same
  // object many times to account for backtracking over new points over this
  // object this is a bit inefficient as we do this iteration even if no new
  // object values are added becuuse we dont know if the affected frames are
  // becuase of old points as well as new points
  //  utils::TimingStatsCollector dyn_obj_affected_timer(this->loggerPrefix() +
  //  ".dyn_object_affected");
  for (const auto& [object_id, frames_affected] :
       result.objects_affected_per_frame) {
    VLOG(20) << "Iterating over frames for which a motion was added "
             << container_to_string(frames_affected) << " for object "
             << object_id;

    auto object_node = map->getObject(object_id);

    std::vector<FrameId> frames_affected_vector(frames_affected.begin(),
                                                frames_affected.end());
    // TODO: this is NO LONGER the case in the object centric case
    // TODO: should always have at least two frames (prev and current) as we
    // must add factors on this frame and the previous frame
    //  CHECK_GE(frames_affected_vector.size(), 2u);
    for (size_t frame_idx = 0; frame_idx < frames_affected_vector.size();
         frame_idx++) {
      const FrameId frame_id = frames_affected_vector.at(frame_idx);
      auto frame_node_k_impl = map->getFrame(frame_id);

      ObjectUpdateContextType object_update_context;
      // perform motion check on this frame -> if this is the first frame (of at
      // least 2) then there should be no motion at this frame (since it is k-1
      // of a motion pair and there is no k-2)
      if (frame_idx == 0) {
        object_update_context.has_motion_pair = false;
      } else {
        object_update_context.has_motion_pair = true;
      }
      object_update_context.frame_node_k = frame_node_k_impl;
      object_update_context.object_node = object_node;

      // gtsam::Values local_new_values;
      objectUpdateContext(object_update_context, result, internal_new_values,
                          internal_new_factors);
      // // update internal theta and factors
      // theta_.insert(local_new_values);
      // // add to the external new_values
      // new_values.insert(local_new_values);
    }
  }

  // this doesnt really work any more as debug info is meant to be per frame?
  if (result.debug_info && VLOG_IS_ON(20)) {
    for (const auto& [object_id, object_info] :
         result.debug_info->getObjectInfos()) {
      std::stringstream ss;
      ss << "Object id debug info: " << object_id << "\n";
      ss << object_info;
      LOG(INFO) << ss.str();
    }
  }

  factors_ += internal_new_factors;
  new_factors += internal_new_factors;
  // update internal theta and factors
  theta_.insert(internal_new_values);
  // add to the external new_values
  new_values.insert(internal_new_values);

  return result;
}

template <typename MAP>
void Formulation<MAP>::logBackendFromMap(const BackendMetaData& backend_info) {
  // TODO:
  std::string logger_prefix = this->getFullyQualifiedName();
  const std::string suffix = backend_info.logging_suffix;

  // add suffix to name if required
  if (!suffix.empty()) {
    logger_prefix += ("_" + suffix);
  }
  BackendLogger::UniquePtr logger =
      std::make_unique<BackendLogger>(logger_prefix);

  typename Map::Ptr map = this->map();
  auto accessor = this->accessorFromTheta();

  CHECK(hooks().ground_truth_packets_request);
  const auto ground_truth_packets = hooks().ground_truth_packets_request();

  // TODO: formulation params are now backend params so no longer need to
  //  pass backend params into Formulation with BackendMetaData
  //  CHECK_NOTNULL(backend_info.backend_params);
  //  const auto& backend_params = *backend_info.backend_params;

  const ObjectPoseMap object_pose_map = accessor->getObjectPoses();

  for (FrameId frame_k : map->getFrameIds()) {
    // TODO: hack - only go up to frames < full batch so we actually only
    // include the optimised alues
    // TODO: actually should be based on the optimization mode!!
    if (params_.use_full_batch_opt &&
        params_.full_batch_frame - 1 == (int)frame_k) {
      break;
    }

    std::stringstream ss;
    ss << "Logging data from map at frame " << frame_k;

    // get MotionestimateMap
    //  const MotionEstimateMap motions = map->getMotionEstimates(frame_k);
    {
      const MotionEstimateMap motions = accessor->getObjectMotions(frame_k);
      auto result =
          logger->logObjectMotion(frame_k, motions, ground_truth_packets);
      if (result)
        ss << " Logged " << *result << " motions from " << motions.size()
           << " computed motions.";
      else
        ss << " Could not log object motions.";
    }

    StateQuery<gtsam::Pose3> X_k_query = accessor->getSensorPose(frame_k);

    if (X_k_query) {
      logger->logCameraPose(frame_k, X_k_query.get(), ground_truth_packets);
    } else {
      LOG(WARNING) << "Could not log camera pose estimate at frame " << frame_k;
    }

    logger->logObjectPose(frame_k, object_pose_map, ground_truth_packets);

    if (map->frameExists(frame_k)) {
      auto static_map = accessor->getStaticLandmarkEstimates(frame_k);
      auto dynamic_map = accessor->getDynamicLandmarkEstimates(frame_k);

      CHECK(X_k_query);  // actually not needed for points in world!!
      logger->logPoints(frame_k, *X_k_query, static_map);
      logger->logPoints(frame_k, *X_k_query, dynamic_map);
    }

    LOG(INFO) << ss.str();
  }

  logger.reset();
}

template <typename MAP>
typename Formulation<MAP>::AccessorType::Ptr
Formulation<MAP>::accessorFromTheta() const {
  if (!accessor_theta_) {
    SharedFormulationData shared_data(&theta_, &hooks_);
    accessor_theta_ = createAccessor(shared_data);
  }
  return accessor_theta_;
}

template <typename MAP>
std::string Formulation<MAP>::setFullyQualifiedName() const {
  // get the derived name of the formulation
  std::string logger_prefix = this->loggerPrefix();
  const std::string suffix = params_.updater_suffix;

  // add suffix to name if required
  if (!suffix.empty()) {
    logger_prefix += ("_" + suffix);
  }
  fully_qualified_name_ = logger_prefix;
  return *fully_qualified_name_;
}

}  // namespace dyno
