#pragma once

#include "dynosam/backend/Accessor.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam_opt/Map.hpp"

namespace dyno {

class VIOAccessor : public Accessor {
 public:
  DYNO_POINTER_TYPEDEFS(VIOAccessor)

  VIOAccessor() {}
  virtual ~VIOAccessor() {}

  // in these cases we rely mostly on the derived accessor functions for look up
  // these cases should have access to the map rather than just doing a raw
  // query with the key
  StateQuery<gtsam::NavState> getNavState(FrameId frame_id) const;
  StateQuery<gtsam::imuBias::ConstantBias> getImuBias(FrameId frame_id) const;

 private:
};

template <typename MAP>
class VIOUpdater;

template <typename MAP>
class VIOFormulation : public FormulationT<MAP> {
 public:
  using Base = FormulationT<MAP>;
  using This = VIOFormulation<MAP>;
  using Map = typename Base::Map;

  // Map alias's available to deriving classes
  using Base::AccessorTypePointer;
  using Base::MapTraitsType;
  using Base::ObjectUpdateContextType;
  using Base::PointUpdateContextType;

  DYNO_POINTER_TYPEDEFS(This)

  VIOFormulation(const FormulationParams& params, typename Map::Ptr map,
                 const NoiseModels& noise_models, const Sensors& sensors,
                 const FormulationHooks& hooks);
  virtual ~VIOFormulation() = default;

  UpdateObservationResult updateStaticObservations(
      FrameId frame_id_k, gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors,
      const UpdateObservationParams& update_params) override;

  gtsam::NavState addStatesInitalise(
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
      FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& X_W_k,
      const gtsam::Vector3& V_W_k = gtsam::Vector3(0, 0, 0));

  // Eventually should encpsualte propogate in EgoMotion struct which contains
  // both T_km1_k and imu-preintegration
  gtsam::NavState addStatesPropogate(gtsam::Values& new_values,
                                     gtsam::NonlinearFactorGraph& new_factors,
                                     FrameId frame_id_k, Timestamp timestamp_k,
                                     const gtsam::Pose3& T_k_1_k,
                                     const ImuFrontend::PimPtr& pim = nullptr);

  void addSensorPosePrior(gtsam::NonlinearFactorGraph& new_factors,
                          FrameId frame_id_k, const gtsam::Pose3& X_W_k,
                          gtsam::SharedNoiseModel noise_model);

  void addSensorPose(gtsam::Values& new_values, FrameId frame_id_k,
                     const gtsam::Pose3& X_W_k);

  VIOAccessor::Ptr getAsVIOAccessor() const;

  bool isImuInitalized() const { return imu_states_initalise_; };

  StateQuery<gtsam::NavState> getNavState(FrameId frame_id);

  FrameId getLastPropogatedFrame() const { return last_propogate_frame_; }

  /* If a landmark exists in the estimation problem */
  bool staticLandmarkExists(TrackletId tracklet_id) const;

  gtsam::Point3 staticLandmarkEstimate(TrackletId tracklet_id) const;

 private:
  gtsam::NavState predictAndAddFactorsVO(
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
      FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& T_k_1_k);

  gtsam::NavState predictAndAddFactorsIMU(
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
      FrameId frame_id_k, Timestamp timestamp_k,
      const ImuFrontend::PimPtr& pim);

  void addImuStatesFromInitialNavState(
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors);

  template <typename DerivedUpdater>
  std::shared_ptr<DerivedUpdater> updaterAs() const {
    return std::static_pointer_cast<DerivedUpdater>(static_updater_);
  }

 private:
  bool imu_states_initalise_{false};

  FrameId first_frame_{0};
  //! First nav state
  gtsam::NavState initial_nav_state_;
  //! First imu bias
  gtsam::imuBias::ConstantBias initial_imu_bias_;

  FrameId last_propogate_frame_{0};
  Timestamp last_propogate_time_{0.0};

  //! Initial ego-motion velocity prior noise model
  gtsam::SharedNoiseModel init_vel_prior_noise_;
  //! Initial imu bias prior noise model
  gtsam::SharedNoiseModel init_imu_bias_prior_noise_;

  friend class VIOUpdater<MAP>;
  // Only shared to keep implementation hidden
  using VIOUpdaterM = VIOUpdater<MAP>;
  std::shared_ptr<VIOUpdaterM> static_updater_;

  Camera::CalibrationType::shared_ptr K_;
};

}  // namespace dyno

#include "dynosam/formulations/VIOFormulation-impl.hpp"
#include "dynosam/formulations/VIOUpdater-impl.hpp"
