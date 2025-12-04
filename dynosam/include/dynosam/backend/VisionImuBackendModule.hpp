/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>

#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/frontend/imu/ImuFrontend.hpp"

namespace dyno {

// TODO(jesse) not sure this should be here but where to put you...?

/**
 * @brief Class that handles the management of ego-motion (NavState) states from
 * VO and (optionally) IMU.
 *
 * This class encapsualtes the current NavState and handles the forward
 * prediction of the current state given new VO/IMU data and the addition of
 * odometry/pre-integration factors into the factor graph. In this way it acts a
 * bit like a Formulation where the internal state is updated and returns (by
 * out-args) the resultant new values and factors.
 *
 * NOTE: needs access to frame id/timestamp data which it gets from the
 * shared_module_info from the base BackendModule. This is a bit confusing as
 * its not obvious how this data is updated (usually it gets updated in the base
 * class on spin) but this data is really important as it is used by the forward
 * prediction
 *
 *
 * @tparam MODULE_TRAITS
 */
template <class MODULE_TRAITS>
class VisionImuBackendModule : public BackendModuleType<MODULE_TRAITS> {
 public:
  using This = VisionImuBackendModule<MODULE_TRAITS>;
  using Base = BackendModuleType<MODULE_TRAITS>;
  using MapType = typename Base::MapType;
  using FormulationType = typename Base::FormulationType;

  DYNO_POINTER_TYPEDEFS(This)

  VisionImuBackendModule(const BackendParams& params,
                         ImageDisplayQueue* display_queue)
      : Base(params, display_queue) {}

  bool isImuInitalized() const { return imu_states_initalise_; };
  const gtsam::NavState& currentNavState() const { return nav_state_; }
  FrameId frameId() const { return last_nav_state_frame_id_; }
  Timestamp timestamp() const { return last_nav_state_time_; }
  gtsam::imuBias::ConstantBias currentImuBias() const { return imu_bias_; }

  virtual gtsam::KeyVector getInvolvedStates(FrameId frame_id_k) const {
    return {CameraPoseSymbol(frame_id_k),
            gtsam::Symbol(kVelocitySymbolChar, frame_id_k),
            gtsam::Symbol(kImuBiasSymbolChar, frame_id_k)};
  }

  gtsam::NavState addInitialVisualState(
      FrameId frame_id_k, Timestamp timestamp_k, FormulationType* formulation,
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
      const NoiseModels& noise_models, const gtsam::Pose3& X_k_w,
      const gtsam::Vector3& V_k_body = gtsam::Vector3(0, 0, 0)) {
    CHECK_NOTNULL(formulation);
    formulation->addSensorPoseValue(X_k_w, frame_id_k, new_values);

    const auto& initial_pose_prior = noise_models.initial_pose_prior;
    formulation->addSensorPosePriorFactor(X_k_w, initial_pose_prior, frame_id_k,
                                          new_factors);

    // update nav state
    return updateNavState(frame_id_k, timestamp_k, X_k_w, V_k_body);
  }

  gtsam::NavState addInitialVisualInertialState(
      FrameId frame_id_k, Timestamp timestamp_k, FormulationType* formulation,
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
      const NoiseModels& noise_models, const gtsam::NavState& nav_state_k,
      const gtsam::imuBias::ConstantBias& imu_bias) {
    return addInitialVisualInertialState(
        frame_id_k, timestamp_k, formulation, new_values, new_factors,
        noise_models, nav_state_k.pose(), nav_state_k.bodyVelocity(), imu_bias);
  }

  gtsam::NavState addInitialVisualInertialState(
      FrameId frame_id_k, Timestamp timestamp_k, FormulationType* formulation,
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
      const NoiseModels& noise_models, const gtsam::Pose3& X_k_w,
      const gtsam::Vector3& V_k_body,
      const gtsam::imuBias::ConstantBias& imu_bias)

  {
    CHECK_NOTNULL(formulation);

    // update pose and pose covariance
    addInitialVisualState(frame_id_k, timestamp_k, formulation, new_values,
                          new_factors, noise_models, X_k_w);

    gtsam::SharedNoiseModel noise_init_vel_prior =
        gtsam::noiseModel::Isotropic::Sigma(3, 1e-5);

    gtsam::Vector6 prior_imu_bias_sigmas;
    prior_imu_bias_sigmas.head<3>().setConstant(0.1);
    prior_imu_bias_sigmas.tail<3>().setConstant(0.01);
    gtsam::SharedNoiseModel imu_bias_prior_noise =
        gtsam::noiseModel::Diagonal::Sigmas(prior_imu_bias_sigmas);

    // add initial imu values
    formulation->addValuesFunctional(
        [&frame_id_k, &V_k_body, &imu_bias](gtsam::Values& values) {
          values.insert(gtsam::Symbol(kVelocitySymbolChar, frame_id_k),
                        V_k_body);
          values.insert(gtsam::Symbol(kImuBiasSymbolChar, frame_id_k),
                        imu_bias);
        },
        new_values);

    // add priors on initial IMU values
    formulation->addFactorsFunctional(
        [&](gtsam::NonlinearFactorGraph& factors) {
          factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
              gtsam::Symbol(kVelocitySymbolChar, frame_id_k), V_k_body,
              noise_init_vel_prior);
          new_factors
              .emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                  gtsam::Symbol(kImuBiasSymbolChar, frame_id_k), imu_bias,
                  imu_bias_prior_noise);
        },
        new_factors);

    // tell system that IMU states have been initalised
    imu_states_initalise_ = true;

    // update internal state information
    imu_bias_ = imu_bias;
    return updateNavState(frame_id_k, timestamp_k, X_k_w, V_k_body);
  }

  gtsam::NavState addVisualInertialStates(
      FrameId frame_id_k, Timestamp timestamp_k, FormulationType* formulation,
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
      const NoiseModels& noise_models, const gtsam::Pose3& T_k_1_k,
      const ImuFrontend::PimPtr& pim = nullptr) {
    CHECK_GT(frame_id_k, 0);
    // FrameId frame_id_k_1 = frame_id_k - 1u;
    // NOTE: this should becomes last_kf_id
    FrameId frame_id_k_1 = last_nav_state_frame_id_;
    LOG(INFO) << "Last kf id " << frame_id_k_1;

    auto from_id = frame_id_k_1;
    auto to_id = frame_id_k;

    auto accessor = CHECK_NOTNULL(formulation)->accessorFromTheta();

    // could be from VO or IMU. Do we need to know?
    const gtsam::NavState predicted_navstate_k =
        predictNewState(frame_id_k, timestamp_k, T_k_1_k, pim);

    // always add sensor pose
    formulation->addSensorPoseValue(predicted_navstate_k.pose(), frame_id_k,
                                    new_values);

    // TODO: using RuntimeSensorOptions...
    if (pim) {
      VLOG(10) << "Adding states/factors between frames " << from_id << " -> "
               << to_id << " using IMU";
      CHECK(accessor->exists(gtsam::Symbol(kVelocitySymbolChar, frame_id_k_1)));
      CHECK(accessor->exists(gtsam::Symbol(kImuBiasSymbolChar, frame_id_k_1)));
      CHECK(accessor->exists(gtsam::Symbol(kPoseSymbolChar, frame_id_k_1)));

      const gtsam::PreintegratedCombinedMeasurements& pim_combined =
          dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(*pim);

      // add new initial values for velocity using predicted velocity and
      // current imu_bias (which will be updated after opt if necessary)
      formulation->addValuesFunctional(
          [&](gtsam::Values& values) {
            values.insert(gtsam::Symbol(kVelocitySymbolChar, frame_id_k),
                          predicted_navstate_k.velocity());
            values.insert(gtsam::Symbol(kImuBiasSymbolChar, frame_id_k),
                          imu_bias_);
          },
          new_values);

      formulation->addFactorsFunctional(
          [&](gtsam::NonlinearFactorGraph& factors) {
            factors.emplace_shared<gtsam::CombinedImuFactor>(
                gtsam::Symbol(kPoseSymbolChar, from_id),
                gtsam::Symbol(kVelocitySymbolChar, from_id),
                gtsam::Symbol(kPoseSymbolChar, to_id),
                gtsam::Symbol(kVelocitySymbolChar, to_id),
                gtsam::Symbol(kImuBiasSymbolChar, from_id),
                gtsam::Symbol(kImuBiasSymbolChar, to_id), pim_combined);
          },
          new_factors);

    } else {
      VLOG(10) << "Adding states/factors between frames " << from_id << " -> "
               << to_id << " using VO";
      if (this->base_params_.use_vo) {
        formulation->addFactorsFunctional(
            [&](gtsam::NonlinearFactorGraph& factors) {
              auto odometry_noise = noise_models.odometry_noise;
              factor_graph_tools::addBetweenFactor(from_id, to_id, T_k_1_k,
                                                   odometry_noise, factors);
            },
            new_factors);
        VLOG(30) << "Added Between factor frames " << from_id << " -> " << to_id
                 << " using VO";
      }
    }

    return predicted_navstate_k;
  }

  gtsam::NavState predictNewState(FrameId frame_id_k, Timestamp timestamp_k,
                                  const gtsam::Pose3& T_k_1_k,
                                  const ImuFrontend::PimPtr& pim) const {
    CHECK_GT(frame_id_k, 0);
    CHECK_EQ(frame_id_k, (last_nav_state_frame_id_ + 1))
        << "State frame id's are not incrementally ascending. Are we dropping "
           "inertial/VO data per frame?";

    gtsam::NavState navstate_k;

    bool predict_imu = false;

    if (pim) {
      if (!isImuInitalized()) {
        LOG(FATAL) << "Inconsistent VisionImu state - Preintegration recieved "
                      "at frame "
                   << frame_id_k << " but module is not IMU initalized!";

      } else {
        predict_imu = true;
      }
    }

    // TODO: using RuntimeSensorOptions...
    if (predict_imu) {
      // TODO: checks that pim is from last state?
      navstate_k = pim->predict(nav_state_, imu_bias_);
      VLOG(10) << "Forward predicting frame=" << frame_id_k << " using PIM";
    } else {
      VLOG(10) << "Forward predicting frame=" << frame_id_k << " using VO";
      gtsam::Pose3 X_w_k_1 = nav_state_.pose();

      // apply relative pose
      gtsam::Pose3 X_w_k = X_w_k_1 * T_k_1_k;

      // assume previous timestep is frame frameid - 1
      CHECK_GT(timestamp_k, last_nav_state_time_);

      double dt = timestamp_k - last_nav_state_time_;
      gtsam::Vector3 V_body_k = T_k_1_k.translation() / dt;

      navstate_k = NavState(X_w_k, V_body_k);
    }

    return navstate_k;
  }

  const gtsam::NavState& updateNavStateFromFormulation(
      FrameId frame_id_k, Timestamp timestamp_k,
      const FormulationType* formulation) {
    const auto nav_state = navState(frame_id_k, timestamp_k, formulation);
    return updateNavState(frame_id_k, timestamp_k, nav_state);
  }

  // Ideally this should be const but it updated the imu_bias_... why?
  gtsam::NavState navState(FrameId frame_id_k, Timestamp timestamp_k,
                           const FormulationType* formulation) {
    auto accessor = CHECK_NOTNULL(formulation)->accessorFromTheta();
    const gtsam::Values& values = formulation->getTheta();
    StateQuery<gtsam::Pose3> X_w_k_query = accessor->getSensorPose(frame_id_k);
    CHECK(X_w_k_query);

    gtsam::NavState nav_state;
    if (imu_states_initalise_) {
      // update imu internal state
      imu_bias_ = values.at<gtsam::imuBias::ConstantBias>(
          gtsam::Symbol(kImuBiasSymbolChar, frame_id_k));

      gtsam::Vector3 V_body_k = values.at<gtsam::Vector3>(
          gtsam::Symbol(kVelocitySymbolChar, frame_id_k));

      nav_state = gtsam::NavState(X_w_k_query.value(), V_body_k);
    } else {
      // assume previous timestep is frame frameid - 1
      CHECK_GT(timestamp_k, last_nav_state_time_);

      // nav state pose should be the same as querrying the accessor for the
      // camera pose at k-1 however we are already using the last nav_state_time
      // to get the previous timestamp and the rest of the code (should!)
      // enforce this consistency!!
      gtsam::Pose3 T_k_1_k = nav_state_.pose().inverse() * X_w_k_query.get();

      double dt = timestamp_k - last_nav_state_time_;
      gtsam::Vector3 V_body_k = T_k_1_k.translation() / dt;

      nav_state = gtsam::NavState(X_w_k_query.value(), V_body_k);
    }
    return nav_state;
  }

  const gtsam::NavState& updateNavState(FrameId frame_id_k,
                                        Timestamp timestamp_k,
                                        const gtsam::Pose3& X_k_w,
                                        const gtsam::Vector3& V_k_body) {
    return updateNavState(frame_id_k, timestamp_k,
                          gtsam::NavState(X_k_w, V_k_body));
  }

  const gtsam::NavState& updateNavState(FrameId frame_id_k,
                                        Timestamp timestamp_k,
                                        const gtsam::NavState& nav_state_k) {
    nav_state_ = nav_state_k;
    last_nav_state_time_ = timestamp_k;
    last_nav_state_frame_id_ = frame_id_k;

    return nav_state_;
  }

 protected:
  // currently assumuing nav/state updates every frame so last last state time
  // is the previous timestamp (ie. no KF)
  gtsam::NavState nav_state_;
  Timestamp last_nav_state_time_;
  FrameId last_nav_state_frame_id_{0};
  gtsam::imuBias::ConstantBias imu_bias_;

 private:
  bool imu_states_initalise_{false};
};

}  // namespace dyno
