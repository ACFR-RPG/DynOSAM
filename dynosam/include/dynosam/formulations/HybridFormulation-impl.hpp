#pragma once

#include "dynosam/formulations/HybridAccessor-impl.hpp"
#include "dynosam/formulations/HybridEstimator.hpp"

namespace dyno {

template <typename MAP>
HybridFormulation<MAP>::HybridFormulation(const FormulationParams& params,
                                          typename Map::Ptr map,
                                          const NoiseModels& noise_models,
                                          const Sensors& sensors,
                                          const FormulationHooks& hooks)
    : Base(params, map, noise_models, sensors, hooks) {
  auto camera = this->sensors_.camera;
  CHECK_NOTNULL(camera);
  rgbd_camera_ = camera->safeGetRGBDCamera();
  CHECK_NOTNULL(rgbd_camera_);
}

template <typename MAP>
void HybridFormulation<MAP>::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;
  const auto object_id = context.getObjectId();
  const auto frame_id_k_1 = frame_node_k_1->getId();

  Accessor::Ptr theta_accessor = this->accessorFromTheta();

  gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(object_id);
  const gtsam::Key object_motion_key_k_1 =
      frame_node_k_1->makeObjectMotionKey(object_id);

  // gtsam::Pose3 L_e;
  const IntermediateMotionInfo keyframe_info =
      getIntermediateMotionInfo(object_id, frame_id_k_1);

  const FrameId& s0 = keyframe_info.kf_id;
  const gtsam::Pose3& L_e = keyframe_info.keyframe_pose;
  const gtsam::Pose3& H_W_e_k_initial = keyframe_info.H_W_e_k_initial;

  auto landmark_motion_noise = this->noiseModels().landmark_motion_noise;

  if (!isDynamicTrackletInMap(lmk_node)) {
    // mark as now in map and include associated frame!!s
    is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), s0);
    all_dynamic_landmarks_.insert2(context.getTrackletId(), s0);
    CHECK(isDynamicTrackletInMap(lmk_node));

    Landmark lmk_L0_init = HybridObjectMotion::projectToObject3(
        context.X_k_1_measured, H_W_e_k_initial, L_e,
        MeasurementTraits::point(lmk_node->getMeasurement(frame_node_k_1)));

    // TODO: this should not every be true as this is a new value!!!
    Landmark lmk_L0;
    getSafeQuery(lmk_L0, theta_accessor->query<Landmark>(point_key),
                 lmk_L0_init);
    // TODO: cache what s0 the landmark is made at so we can propogate them
    // later using the right motions within the correct Keyframe range!!!!
    new_values.insert(point_key, lmk_L0);
    result.updateAffectedObject(frame_node_k_1->frameId(),
                                context.getObjectId());
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_new_dynamic_points++;
  }

  if (context.is_starting_motion_frame) {
    // add factor at k-1
    Landmark measured_point_local;
    gtsam::SharedNoiseModel measurement_covariance;
    std::tie(measured_point_local, measurement_covariance) =
        MeasurementTraits::pointWithCovariance(
            lmk_node->getMeasurement(frame_node_k_1));

    if (this->params_.makeDynamicMeasurementsRobust()) {
      measurement_covariance = factor_graph_tools::robustifyHuber(
          this->params_.k_huber_3d_points_, measurement_covariance);
    }

    new_factors.emplace_shared<HybridMotionFactor>(
        frame_node_k_1->makePoseKey(),  // pose key at previous frames,
        object_motion_key_k_1, point_key, measured_point_local, L_e,
        measurement_covariance);
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_dynamic_factors++;
  }

  // add factor at k

  Landmark measured_point_local;
  gtsam::SharedNoiseModel measurement_covariance;
  std::tie(measured_point_local, measurement_covariance) =
      MeasurementTraits::pointWithCovariance(
          lmk_node->getMeasurement(frame_node_k));

  if (this->params_.makeDynamicMeasurementsRobust()) {
    measurement_covariance = factor_graph_tools::robustifyHuber(
        this->params_.k_huber_3d_points_, measurement_covariance);
  }

  new_factors.emplace_shared<HybridMotionFactor>(
      frame_node_k->makePoseKey(),  // pose key at previous frames,
      object_motion_key_k, point_key, measured_point_local, L_e,
      measurement_covariance);

  result.updateAffectedObject(frame_node_k->frameId(), context.getObjectId());
  if (result.debug_info)
    result.debug_info->getObjectInfo(context.getObjectId())
        .num_dynamic_factors++;
}

template <typename MAP>
void HybridFormulation<MAP>::objectUpdateContext(
    const ObjectUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  auto frame_node_k = context.frame_node_k;
  auto object_node = context.object_node;
  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());

  auto theta_accessor = this->accessorFromTheta();
  const auto frame_id = context.getFrameId();
  const auto object_id = context.getObjectId();

  const IntermediateMotionInfo keyframe_info =
      getIntermediateMotionInfo(object_id, frame_id);

  if (!this->is_other_values_in_map.exists(object_motion_key_k)) {
    // gtsam::Pose3 motion;
    const gtsam::Pose3 X_world =
        this->getInitialOrLinearizedSensorPose(frame_id);
    // gtsam::Pose3 motion = computeInitialH(object_id, frame_id);
    VLOG(5) << "Added motion at  " << DynosamKeyFormatter(object_motion_key_k);
    // gtsam::Pose3 motion;
    new_values.insert(object_motion_key_k, keyframe_info.H_W_e_k_initial);
    this->is_other_values_in_map.insert2(object_motion_key_k, true);

    // for now lets treat num_motion_factors as motion (values) added!!
    if (result.debug_info)
      result.debug_info->getObjectInfo(context.getObjectId())
          .num_motion_factors++;

    // we are at object keyframe
    // NOTE: this should never happen for hybrid KF!!
    if (keyframe_info.kf_id == frame_id) {
      // add prior
      new_factors.addPrior<gtsam::Pose3>(
          object_motion_key_k, gtsam::Pose3::Identity(),
          this->noiseModels().initial_pose_prior);
    }

    // test stuff
    FrameId first_seen_object_frame = object_node->getFirstSeenFrame();
    if (first_seen_object_frame == frame_id) {
      CHECK_EQ(keyframe_info.kf_id, frame_id);
    }
  }

  if (frame_id < 2) return;

  auto frame_node_k_1 = this->map_->getFrame(frame_id - 1u);
  auto frame_node_k_2 = this->map_->getFrame(frame_id - 2u);
  if (!frame_node_k_1 || !frame_node_k_2) {
    return;
  }

  if (this->params_.use_smoothing_factor &&
      frame_node_k_1->objectObserved(object_id) &&
      frame_node_k_2->objectObserved(object_id)) {
    // motion key at previous frame
    const gtsam::Symbol object_motion_key_k_1 =
        frame_node_k_1->makeObjectMotionKey(object_id);

    const gtsam::Symbol object_motion_key_k_2 =
        frame_node_k_2->makeObjectMotionKey(object_id);

    auto object_smoothing_noise = this->noiseModels().object_smoothing_noise;
    CHECK(object_smoothing_noise);
    CHECK_EQ(object_smoothing_noise->dim(), 6u);

    {
      ObjectId object_label_k_1, object_label_k;
      FrameId frame_id_k_1, frame_id_k;
      CHECK(reconstructMotionInfo(object_motion_key_k_1, object_label_k_1,
                                  frame_id_k_1));
      CHECK(reconstructMotionInfo(object_motion_key_k, object_label_k,
                                  frame_id_k));
      CHECK_EQ(object_label_k_1, object_label_k);
      CHECK_EQ(frame_id_k_1 + 1, frame_id_k);  // assumes
      // consequative frames
    }

    // if the motion key at k (motion from k-1 to k), and key at k-1 (motion
    //  from k-2 to k-1)
    // exists in the map or is about to exist via new values, add the
    //  smoothing factor
    bool smoothing_factor_added =
        smoothing_factors_added_.exists(object_motion_key_k);
    if (!smoothing_factor_added &&
        this->is_other_values_in_map.exists(object_motion_key_k_2) &&
        this->is_other_values_in_map.exists(object_motion_key_k_1) &&
        this->is_other_values_in_map.exists(object_motion_key_k)) {
      new_factors.emplace_shared<HybridSmoothingFactor>(
          object_motion_key_k_2, object_motion_key_k_1, object_motion_key_k,
          keyframe_info.keyframe_pose, object_smoothing_noise);
      if (result.debug_info)
        result.debug_info->getObjectInfo(context.getObjectId())
            .smoothing_factor_added = true;

      // update internal containers
      smoothing_factors_added_.insert(object_motion_key_k);
    }
  }
}

template <typename MAP>
bool HybridFormulation<MAP>::isDynamicTrackletInMap(
    const typename MapTraitsType::SharedLandmarkNode& lmk_node) const {
  const TrackletId tracklet_id = lmk_node->trackletId();
  return is_dynamic_tracklet_in_map_.exists(tracklet_id);
}

}  // namespace dyno
