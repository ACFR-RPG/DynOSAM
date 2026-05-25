/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <functional>
#include <unordered_map>

#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_opt/Symbols.hpp"
#include "dynosam_sensors/Camera.hpp"  //for calibration type

namespace dyno {

/// @brief Alias to a gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>
using PoseToPointFactor = gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>;

enum BackendType : int {

  WCME = 0,             // world-centric motion estimator
  WCPE = 1,             // world-centric pose estimator
  HYBRID = 2,           // full-hybrid
  PARALLEL_HYBRID = 3,  // associated to its own special class
  // the following are test formulations that were not specifcially part of a
  // paper but were used for (internal) development/research. they may not work
  // as intended and are included for posterity
  TESTING_HYBRID_SD = 4,   // (SD) structureless-decoupled
  TESTING_HYBRID_D = 5,    // (D) decoupled
  TESTING_HYBRID_S = 6,    // (S) structureless
  TESTING_HYBRID_SMF = 7,  // (SFM) smart motion factor
  KF_HYBRID = 8
};

// TODO: landmark related noise models should not be used anymore!!
struct NoiseModels {
  gtsam::SharedNoiseModel initial_pose_prior;
  //! Between factor noise for between two consequative poses
  gtsam::SharedNoiseModel odometry_noise;
  //! Noise on the landmark tenrary factor
  gtsam::SharedNoiseModel landmark_motion_noise;
  //! Contant velocity noise model between motions
  gtsam::SharedNoiseModel object_smoothing_noise;
  //! Isometric [3x3] noise model on dynamic points;
  gtsam::SharedNoiseModel dynamic_point_noise;
  //! Isometric [3x3] noise model on static points;
  gtsam::SharedNoiseModel static_point_noise;

  static NoiseModels fromBackendParams(const BackendParams&);

  void print(const std::string& name) const;
};

/**
 * @brief Defines a set of input hooks to the formulation that allow
 * communication to outside the formulation
 *
 */
struct FormulationHooks {
  GroundTruthPacketsRequest ground_truth_packets_request;

  void setGroundTruthPacketRequest(const SharedGroundTruth shared_ground_truth);
};

/**
 * @brief Data shared between a Formulation and its accessor
 *
 */
struct SharedFormulationData {
  DYNO_POINTER_TYPEDEFS(SharedFormulationData)

  //! Current linearization shared between the accessor and formulation and
  //! represents the current state
  gtsam::Values theta;
  mutable std::mutex theta_mutex;

  //! Shared formulation hooks
  FormulationHooks hooks;

  void threadSafeSetTheta(const gtsam::Values& linearization) {
    std::lock_guard<std::mutex> lock(this->theta_mutex);
    this->theta = linearization;
  }

  void threadSafeInsertOrAssignTheta(const gtsam::Values& linearization) {
    std::lock_guard<std::mutex> lock(this->theta_mutex);
    this->theta.insert_or_assign(linearization);
  }

  void threadSafeInsertTheta(const gtsam::Values& linearization) {
    std::lock_guard<std::mutex> lock(this->theta_mutex);
    this->theta.insert(linearization);
  }
};

using CalibrationType =
    Camera::CalibrationType;  // TODO: really need to check that this one
                              // matches the calibration in the camera!!

using Slot = long int;

constexpr static Slot UninitialisedSlot =
    -1;  //! Inidicates that a factor is not in the graph or uninitialised

using SmartProjectionFactor = gtsam::SmartProjectionPoseFactor<CalibrationType>;
using GenericProjectionFactor =
    gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3,
                                   CalibrationType>;
using GenericStereoFactor = gtsam::GenericStereoFactor<gtsam::Pose3, Landmark>;

using SmartProjectionFactorParams = gtsam::SmartProjectionParams;

template <typename Key, typename Factor>
using GenericFactorMap = gtsam::FastMap<Key, std::pair<Factor, Slot>>;

template <typename Factor>
using FactorMap = GenericFactorMap<TrackletId, Factor>;

class DebugInfo {
 public:
  DYNO_POINTER_TYPEDEFS(DebugInfo)

  int num_static_factors = 0;  // num new static factors added
  int num_new_static_points = 0;

  struct ObjectInfo {
    int num_dynamic_factors = 0;
    int num_new_dynamic_points = 0;
    int num_motion_factors = 0;
    bool smoothing_factor_added{false};

    operator std::string() const;
    friend std::ostream& operator<<(std::ostream& os,
                                    const ObjectInfo& object_info);
  };

  ObjectInfo& getObjectInfo(ObjectId object_id);
  const ObjectInfo& getObjectInfo(ObjectId object_id) const;

  const gtsam::FastMap<ObjectId, ObjectInfo>& getObjectInfos() const {
    return object_info_;
  }

  bool odometry_factor_added{false};

  double update_static_time = 0;
  double update_dynamic_time = 0;
  double optimize_time = 0;

  double error_before = 0;
  double error_after = 0;

  size_t num_factors = 0;
  size_t num_values = 0;

  int num_elements_in_matrix = 0;
  int num_zeros_in_matrix = 0;

 private:
  mutable gtsam::FastMap<ObjectId, ObjectInfo> object_info_{};

  inline auto& getObjectInfoImpl(ObjectId object_id) const {
    if (!object_info_.exists(object_id)) {
      object_info_.insert2(object_id, ObjectInfo{});
    }
    return object_info_.at(object_id);
  }
};

class BackendLogger : public EstimationModuleLogger {
 public:
  DYNO_POINTER_TYPEDEFS(BackendLogger)
  BackendLogger(const std::string& name_prefix);
  virtual ~BackendLogger() = default;

 private:
};

}  // namespace dyno
