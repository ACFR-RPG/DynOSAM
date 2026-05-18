#include "dynosam/formulations/VIOFormulation.hpp"

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

}  // namespace dyno
