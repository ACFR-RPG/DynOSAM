#include "dynosam/backend/rgbd/VIOFormulation.hpp"

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>

namespace dyno {

StateQuery<gtsam::NavState> VIOAccessor::getNavState(FrameId frame_id) const {
  StateQuery<gtsam::Pose3> X_W_k_query = this->getSensorPose(frame_id);

  if (!X_W_k_query) {
    return StateQuery<gtsam::NavState>::NotInMap(X_W_k_query.key());
  }
  StateQuery<gtsam::Vector3> V_W_k_query =
      this->query<gtsam::Vector3>(CameraVelocitySymbol(frame_id));
  // implict check for "is in imu state"
  // if Camera velocity query is false then we assume we dont have imu values
  // becuase we dont have imu measurements
  // instead calculate nav state via finite difference
  // NOTE: ideally we should check the VIOFormulation::isImuInitalized
  // or equivalent but the Accessor structure is such we need VIOAccessor
  // to have a default constructor!
  if (V_W_k_query) {
    gtsam::NavState nav_state(X_W_k_query.get(), V_W_k_query.get());
    return StateQuery<gtsam::NavState>(X_W_k_query.key(), nav_state);
  }

  const FrameId first_frame = this->getFrameIds().front();
  // if first frame then we dont know the veloicity so just use zero
  // this is hacky and also may be inconcsistent with the initial velocity
  // used in the VIOFormulation but right now we only ever use the default
  if (frame_id == first_frame) {
    gtsam::NavState nav_state(X_W_k_query.get(), gtsam::Vector3(0.0, 0.0, 0.0));
    return StateQuery<gtsam::NavState>(X_W_k_query.key(), nav_state);
  }

  StateQuery<gtsam::Pose3> X_W_km1_query = this->getSensorPose(frame_id - 1u);
  if (!X_W_km1_query) {
    const gtsam::NavState nav_state(X_W_k_query.get(),
                                    gtsam::Vector3(0.0, 0.0, 0.0));
    return StateQuery<gtsam::NavState>(X_W_k_query.key(), nav_state);
  }

  // only works if we have pose every frame... not the case when keyframing ;)
  const Timestamp timestamp_k = this->getTimestamp(frame_id);
  const Timestamp timestamp_km1 = this->getTimestamp(frame_id - 1u);

  // calculate relative pose
  const gtsam::Pose3 T_km1_k = X_W_km1_query->inverse() * X_W_k_query.get();
  const double dt = timestamp_k - timestamp_km1;
  CHECK_GT(dt, 0);
  // discrete derivative
  const gtsam::Vector3 V_C_k = T_km1_k.translation() / dt;
  const gtsam::Vector3 V_W_k = X_W_k_query->rotation().rotate(V_C_k);
  const gtsam::NavState nav_state(X_W_k_query.get(), V_W_k);

  return StateQuery<gtsam::NavState>(X_W_k_query.key(), nav_state);
}

StateQuery<gtsam::imuBias::ConstantBias> VIOAccessor::getImuBias(
    FrameId frame_id) const {
  return this->query<gtsam::imuBias::ConstantBias>(ImuBiasSymbol(frame_id));
}

VIOFormulation::VIOFormulation(const FormulationParams& params,
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
}

gtsam::NavState VIOFormulation::addStatesInitalise(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& X_W_k,
    const gtsam::Vector3& V_W_k) {
  this->addSensorPose(new_values, frame_id_k, X_W_k);
  this->addSensorPosePrior(new_factors, frame_id_k, X_W_k,
                           noise_models_.initial_pose_prior);

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

gtsam::NavState VIOFormulation::addStatesPropogate(
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

void VIOFormulation::addSensorPosePrior(
    gtsam::NonlinearFactorGraph& new_factors, FrameId frame_id_k,
    const gtsam::Pose3& X_W_k, gtsam::SharedNoiseModel noise_model) {
  CHECK(noise_model);
  CHECK_EQ(noise_model->dim(), gtsam::traits<gtsam::Pose3>::dimension);

  this->addFactor(new_factors,
                  boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                      CameraPoseSymbol(frame_id_k), X_W_k, noise_model));
}

void VIOFormulation::addSensorPose(gtsam::Values& new_values,
                                   FrameId frame_id_k,
                                   const gtsam::Pose3& X_W_k) {
  this->addValue(new_values, X_W_k, CameraPoseSymbol(frame_id_k));
}

VIOAccessor::Ptr VIOFormulation::getAsVIOAccessor() const {
  VIOAccessor::Ptr vio_accessor = this->derivedAccessor<VIOAccessor>();

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

StateQuery<gtsam::NavState> VIOFormulation::getNavState(FrameId frame_id) {
  VIOAccessor::Ptr accessor = this->getAsVIOAccessor();
  return accessor->getNavState(frame_id);
};

gtsam::NavState VIOFormulation::predictAndAddFactorsVO(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& T_k_1_k) {
  CHECK(!isImuInitalized());

  LOG(INFO) << "here";

  const FrameId from_frame = last_propogate_frame_;
  const FrameId to_frame = frame_id_k;

  VIOAccessor::Ptr accessor = this->getAsVIOAccessor();
  const gtsam::NavState nav_state_prev =
      DYNO_GET_QUERY_DEBUG(accessor->getNavState(from_frame));

  VLOG(10) << "Forward predicting k=" << frame_id_k << " t=" << timestamp_k
           << " using VO";
  const gtsam::Pose3 X_W_km1 = nav_state_prev.pose();
  // apply relative pose
  const gtsam::Pose3 X_W_k = X_W_km1 * T_k_1_k;

  const double dt = timestamp_k - last_propogate_time_;
  CHECK_GT(dt, 0);
  // discrete derivative
  const gtsam::Vector3 V_C_k = T_k_1_k.translation() / dt;
  const gtsam::Vector3 V_W_k = X_W_k.rotation().rotate(V_C_k);
  const gtsam::NavState nav_state_k(X_W_k, V_W_k);

  if (params().use_vo) {
    auto odometry_noise = noiseModels().odometry_noise;
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

  // const gtsam::Key velocity_key(CameraVelocitySymbol(to_frame));
  // // add predicted velocity value
  // this->addValue(new_values, nav_state_k.velocity(), velocity_key);
  // // initalise imu bias
  // const gtsam::Key imu_bias_key(ImuBiasSymbol(to_frame));
  // this->addValue(new_values, imu_bias_prev, imu_bias_key);

  return nav_state_k;
}

gtsam::NavState VIOFormulation::predictAndAddFactorsIMU(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    FrameId frame_id_k, Timestamp timestamp_k, const ImuFrontend::PimPtr& pim) {
  CHECK(isImuInitalized());
  CHECK(pim);
  const FrameId from_frame = last_propogate_frame_;
  const FrameId to_frame = frame_id_k;

  VLOG(10) << "Forward predicting frame=" << frame_id_k << " using PIM";

  VIOAccessor::Ptr accessor = this->getAsVIOAccessor();
  const gtsam::NavState nav_state_prev =
      DYNO_GET_QUERY_DEBUG(accessor->getNavState(from_frame));
  const gtsam::imuBias::ConstantBias imu_bias_prev =
      DYNO_GET_QUERY_DEBUG(accessor->getImuBias(from_frame));

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

void VIOFormulation::addImuStatesFromInitialNavState(
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
