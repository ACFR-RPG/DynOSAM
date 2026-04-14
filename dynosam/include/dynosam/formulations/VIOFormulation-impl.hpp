#pragma once

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>

#include "dynosam/backend/Formulation.hpp"
#include "dynosam/formulations/VIOFormulation.hpp"
#include "dynosam/formulations/VIOUpdater-impl.hpp"

namespace dyno {

template <typename MAP>
VIOFormulation<MAP>::VIOFormulation(const FormulationParams& params,
                                    typename Map::Ptr map,
                                    const NoiseModels& noise_models,
                                    const Sensors& sensors,
                                    const FormulationHooks& hooks)
    : Base(params, map, noise_models, sensors, hooks) {
  init_vel_prior_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, 1e-5);

  gtsam::Vector6 prior_imu_bias_sigmas;
  prior_imu_bias_sigmas.head<3>().setConstant(0.1);
  prior_imu_bias_sigmas.tail<3>().setConstant(0.01);
  init_imu_bias_prior_noise_ =
      gtsam::noiseModel::Diagonal::Sigmas(prior_imu_bias_sigmas);

  using PTPUpdater = PTPUpdater<MAP>;
  using GenericProjectionUpdater = GenericProjectionUpdater<MAP>;
  using StereoProjectionUpdater = StereoProjectionUpdater<MAP>;

  const StaticFormulationType& static_method_type = params.static_formulation;
  switch (static_method_type) {
    case StaticFormulationType::PTP:
      VLOG(20) << "Using Point-to-Pose formulation for Visual SLAM";
      static_updater_ = std::make_unique<PTPUpdater>(this);
      break;
    case StaticFormulationType::GENERIC_PROJECTION:
      VLOG(20) << "Using Generic Projection formulation for Visual SLAM";
      static_updater_ = std::make_unique<GenericProjectionUpdater>(this);
      break;
    case StaticFormulationType::STEREO_PROJECTION:
      VLOG(20) << "Using Stereo Projection formulation for Visual SLAM";
      static_updater_ = std::make_unique<StereoProjectionUpdater>(this);
      break;
    default:
      LOG(FATAL) << "Unknown method type for Static Formulation!";
  }
}

template <typename MAP>
UpdateObservationResult VIOFormulation<MAP>::updateStaticObservations(
    FrameId frame_id_k, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors,
    const UpdateObservationParams& update_params) {
  typename Map::Ptr map = this->map();
  auto accessor = this->accessorFromTheta();

  // keep track of the new factors added in this function
  // these are then appended to the internal factors_ and new_factors
  gtsam::NonlinearFactorGraph internal_new_factors;

  UpdateObservationResult result(update_params);

  const size_t initial_factors_size = new_factors.size();
  const size_t initial_values_size = new_values.size();

  auto frame_node_k = map->getFrame(frame_id_k);
  CHECK_NOTNULL(frame_node_k);

  const auto& static_method_type = this->params().static_formulation;

  using PTPUpdater = PTPUpdater<MAP>;
  using GenericProjectionUpdater = GenericProjectionUpdater<MAP>;
  using StereoProjectionUpdater = StereoProjectionUpdater<MAP>;

  const auto& static_landmarks = frame_node_k->staticLandmarks();
  VLOG(20) << "Looping over " << static_landmarks.size()
           << " static lmks for frame " << frame_id_k;
  for (auto lmk_node : static_landmarks) {
    gtsam::Key point_key;
    std::optional<Landmark> initial_value;

    switch (static_method_type) {
      case StaticFormulationType::PTP:
        updaterAs<PTPUpdater>()->addLandmark(
            lmk_node, frame_node_k, update_params, new_values,
            internal_new_factors, point_key, result, initial_value);
        break;
      case StaticFormulationType::GENERIC_PROJECTION:
        updaterAs<GenericProjectionUpdater>()->addLandmark(
            lmk_node, frame_node_k, update_params, new_values,
            internal_new_factors, point_key, result, initial_value);
        break;
      case StaticFormulationType::STEREO_PROJECTION:
        updaterAs<StereoProjectionUpdater>()->addLandmark(
            lmk_node, frame_node_k, update_params, new_values,
            internal_new_factors, point_key, result, initial_value);
        break;
      default:
        LOG(FATAL) << "Unknown method type for Static Formulation!";
    }

    CHECK_EQ(point_key, lmk_node->makeStaticKey());
  }

  if (result.debug_info) {
    result.debug_info->num_static_factors =
        internal_new_factors.size() - initial_factors_size;
    result.debug_info->num_new_static_points =
        new_values.size() - initial_values_size;
  }

  // update internal data structures
  this->shared_data_->threadSafeInsertOrAssignTheta(new_values);
  this->factors_ += internal_new_factors;
  new_factors += internal_new_factors;

  if (result.debug_info)
    LOG(INFO) << "Num new static points: "
              << result.debug_info->num_new_static_points
              << " Num new static factors "
              << result.debug_info->num_static_factors;
  return result;
}

template <typename MAP>
gtsam::NavState VIOFormulation<MAP>::addStatesInitalise(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& X_W_k,
    const gtsam::Vector3& V_W_k) {
  this->addSensorPose(new_values, frame_id_k, X_W_k);
  this->addSensorPosePrior(new_factors, frame_id_k, X_W_k,
                           this->noise_models_.initial_pose_prior);

  gtsam::imuBias::ConstantBias initial_bias;  // TODO: make param
  initial_imu_bias_ = initial_bias;

  initial_nav_state_ = gtsam::NavState(X_W_k, V_W_k);

  // // add body velocity state
  // this->addValue(new_values, V_W_k, velocity_key);
  // // add bias state
  // this->addValue(new_values, initial_bias, imu_bias_key);

  // this->addFactor(new_factors,
  //                 boost::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
  //                     velocity_key, V_W_k, init_vel_prior_noise_));
  // this->addFactor(
  //     new_factors,
  //     boost::make_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
  //         imu_bias_key, initial_bias, init_imu_bias_prior_noise_));

  first_frame_ = frame_id_k;
  last_propogate_frame_ = frame_id_k;
  last_propogate_time_ = timestamp_k;
  // auto nav_state_query = accessor->getNavState();
  // CHECK(nav_state_query);
  // return nav_state_query.value();
  // return DYNO_GET_QUERY_DEBUG(accessor->getNavState(frame_id_k));
  return initial_nav_state_;
}

template <typename MAP>
gtsam::NavState VIOFormulation<MAP>::addStatesPropogate(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& T_k_1_k,
    const ImuFrontend::PimPtr& pim) {
  const FrameId from_frame = last_propogate_frame_;
  const FrameId to_frame = frame_id_k;

  CHECK_GT(to_frame, 0);
  CHECK_GT(to_frame, from_frame)
      << "State frame id's are not incrementally ascending. Are we dropping "
         "inertial/VO data per frame?";

  gtsam::NavState nav_state_k;
  bool predict_imu = false;
  if (pim) {
    // first propogate frame
    // check if we have IMU and initalise state as such
    if (first_frame_ == last_propogate_frame_) {
      imu_states_initalise_ = true;
      // only now do we add velocity and IMU bias at k=first_frame_
      // this ensures we dont add imu specific states unless we have
      // measurements from the IMU
      addImuStatesFromInitialNavState(new_values, new_factors);
      LOG(INFO) << "Initised VIO Formulation to use IMU";
    }

    CHECK(isImuInitalized())
        << "Inconsistent VisionImu state - Preintegration recieved "
           "at frame "
        << frame_id_k << " but formulation is not IMU initalized!";

    nav_state_k = predictAndAddFactorsIMU(new_values, new_factors, frame_id_k,
                                          timestamp_k, pim);
  } else {
    CHECK(!isImuInitalized());
    nav_state_k = predictAndAddFactorsVO(new_values, new_factors, frame_id_k,
                                         timestamp_k, T_k_1_k);
  }

  // update frame/timestamp data
  last_propogate_frame_ = frame_id_k;
  last_propogate_time_ = timestamp_k;

  return nav_state_k;
}

template <typename MAP>
void VIOFormulation<MAP>::addSensorPosePrior(
    gtsam::NonlinearFactorGraph& new_factors, FrameId frame_id_k,
    const gtsam::Pose3& X_W_k, gtsam::SharedNoiseModel noise_model) {
  CHECK(noise_model);
  CHECK_EQ(noise_model->dim(), gtsam::traits<gtsam::Pose3>::dimension);

  this->addFactor(new_factors,
                  boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                      CameraPoseSymbol(frame_id_k), X_W_k, noise_model));
}

template <typename MAP>
void VIOFormulation<MAP>::addSensorPose(gtsam::Values& new_values,
                                        FrameId frame_id_k,
                                        const gtsam::Pose3& X_W_k) {
  this->addValue(new_values, X_W_k, CameraPoseSymbol(frame_id_k));
}

template <typename MAP>
VIOAccessor::Ptr VIOFormulation<MAP>::getAsVIOAccessor() const {
  VIOAccessor::Ptr vio_accessor = this->template derivedAccessor<VIOAccessor>();

  if (!vio_accessor) {
    throw DynosamException(
        "VIOAccessor is null in formulation " + this->getFullyQualifiedName() +
        "."
        " If you have a formulation that derives from VIOFormulation you must "
        "construct an Accessor that inherits"
        " from VIOAccessor. When using AccessorT<MAP, DerivedAccessor> ensure "
        "Derived DerivedAccessor=VIOAccessor"
        " or if extending (DerivedAccessor) with another custom accessor, "
        "ensure the custom accessor derived from VIOAccessor");
  }
  return vio_accessor;
}

template <typename MAP>
StateQuery<gtsam::NavState> VIOFormulation<MAP>::getNavState(FrameId frame_id) {
  VIOAccessor::Ptr accessor = this->getAsVIOAccessor();
  return accessor->getNavState(frame_id);
};

template <typename MAP>
gtsam::NavState VIOFormulation<MAP>::predictAndAddFactorsVO(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& T_k_1_k) {
  CHECK(!isImuInitalized());

  const FrameId from_frame = last_propogate_frame_;
  const FrameId to_frame = frame_id_k;

  VIOAccessor::Ptr accessor = this->getAsVIOAccessor();
  const gtsam::NavState nav_state_prev =
      DYNO_GET_QUERY_DEBUG(accessor->getNavState(from_frame));

  VLOG(10) << "Forward predicting " << from_frame << " -> " << frame_id_k
           << " t=" << timestamp_k << "VO";
  const gtsam::Pose3 X_W_km1 = nav_state_prev.pose();
  // apply relative pose
  const gtsam::Pose3 X_W_k = X_W_km1 * T_k_1_k;

  const double dt = timestamp_k - last_propogate_time_;
  CHECK_GT(dt, 0);
  // discrete derivative
  const gtsam::Vector3 V_C_k = T_k_1_k.translation() / dt;
  const gtsam::Vector3 V_W_k = X_W_k.rotation().rotate(V_C_k);
  const gtsam::NavState nav_state_k(X_W_k, V_W_k);

  if (this->params().use_vo) {
    auto odometry_noise = this->noiseModels().odometry_noise;
    CHECK(odometry_noise);
    CHECK_EQ(odometry_noise->dim(), 6u);

    this->addFactor(new_factors,
                    boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                        CameraPoseSymbol(from_frame),
                        CameraPoseSymbol(to_frame), T_k_1_k, odometry_noise));
    VLOG(30) << "Added Between factor frames " << from_frame << " -> "
             << to_frame << " using VO";
  }

  addSensorPose(new_values, to_frame, nav_state_k.pose());

  return nav_state_k;
}

template <typename MAP>
gtsam::NavState VIOFormulation<MAP>::predictAndAddFactorsIMU(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    FrameId frame_id_k, Timestamp timestamp_k, const ImuFrontend::PimPtr& pim) {
  CHECK(isImuInitalized());
  CHECK(pim);
  const FrameId from_frame = last_propogate_frame_;
  const FrameId to_frame = frame_id_k;

  VLOG(10) << "Forward predicting " << from_frame << " -> " << frame_id_k
           << " t=" << timestamp_k << " using PIM";

  VIOAccessor::Ptr accessor = this->getAsVIOAccessor();
  const gtsam::NavState nav_state_prev =
      DYNO_GET_QUERY_DEBUG(accessor->getNavState(from_frame));
  const gtsam::imuBias::ConstantBias imu_bias_prev =
      DYNO_GET_QUERY_DEBUG(accessor->getImuBias(from_frame));

  LOG(INFO) << "Nav state prev " << nav_state_prev;

  const gtsam::NavState nav_state_k =
      pim->predict(nav_state_prev, imu_bias_prev);
  // add predicted camera value
  addSensorPose(new_values, frame_id_k, nav_state_k.pose());
  // add predicted velocity value
  this->addValue(new_values, nav_state_k.velocity(),
                 CameraVelocitySymbol(frame_id_k));
  // initalise imu bias
  this->addValue(new_values, imu_bias_prev, ImuBiasSymbol(frame_id_k));
  // add IMU factor
  const gtsam::PreintegratedCombinedMeasurements& pim_combined =
      dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(*pim);

  this->addFactor(
      new_factors,
      boost::make_shared<gtsam::CombinedImuFactor>(
          CameraPoseSymbol(from_frame), CameraVelocitySymbol(from_frame),
          CameraPoseSymbol(to_frame), CameraVelocitySymbol(to_frame),
          ImuBiasSymbol(from_frame), ImuBiasSymbol(to_frame), pim_combined));

  return nav_state_k;
}

template <typename MAP>
void VIOFormulation<MAP>::addImuStatesFromInitialNavState(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const gtsam::Symbol velocity_key(CameraVelocitySymbol(first_frame_));
  const gtsam::Symbol imu_bias_key(ImuBiasSymbol(first_frame_));

  const gtsam::Point3& V_W_first = initial_nav_state_.velocity();
  const auto& initial_bias = initial_imu_bias_;

  this->addValue(new_values, V_W_first, velocity_key);
  // add bias state
  this->addValue(new_values, initial_bias, imu_bias_key);

  this->addFactor(new_factors,
                  boost::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
                      velocity_key, V_W_first, init_vel_prior_noise_));
  this->addFactor(
      new_factors,
      boost::make_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
          imu_bias_key, initial_bias, init_imu_bias_prior_noise_));
}

}  // namespace dyno
