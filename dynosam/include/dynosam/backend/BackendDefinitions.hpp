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

#include <gtsam/inference/LabeledSymbol.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <functional>
#include <unordered_map>

#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/backend/DynamicPointSymbol.hpp"
#include "dynosam/common/Camera.hpp"  //for calibration type
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/common/GroundTruthPacket.hpp"
#include "dynosam/common/SensorModels.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/logger/Logger.hpp"

namespace dyno {

/// @brief Alias to a gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>
using PoseToPointFactor = gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>;

using SymbolChar = unsigned char;
static constexpr SymbolChar kPoseSymbolChar = 'X';
static constexpr SymbolChar kVelocitySymbolChar = 'V';
static constexpr SymbolChar kObjectMotionSymbolChar = 'H';
static constexpr SymbolChar kObjectPoseSymbolChar = 'L';
static constexpr SymbolChar kStaticLandmarkSymbolChar = 'l';
static constexpr SymbolChar kDynamicLandmarkSymbolChar = 'm';
static constexpr SymbolChar kImuBiasSymbolChar = 'b';

inline gtsam::Key H(unsigned char label, std::uint64_t j) {
  return gtsam::LabeledSymbol(kObjectMotionSymbolChar, label, j);
}
inline gtsam::Key L(unsigned char label, std::uint64_t j) {
  return gtsam::LabeledSymbol(kObjectPoseSymbolChar, label, j);
}

inline gtsam::Symbol CameraPoseSymbol(FrameId frame_id) {
  return gtsam::Symbol(kPoseSymbolChar, frame_id);
}
inline gtsam::Symbol StaticLandmarkSymbol(TrackletId tracklet_id) {
  return gtsam::Symbol(kStaticLandmarkSymbolChar, tracklet_id);
}
inline DynamicPointSymbol DynamicLandmarkSymbol(FrameId frame_id,
                                                TrackletId tracklet_id) {
  return DynamicPointSymbol(kDynamicLandmarkSymbolChar, tracklet_id, frame_id);
}
inline gtsam::Key ObjectMotionSymbol(ObjectId object_label, FrameId frame_id) {
  unsigned char label = object_label + '0';
  return H(label, static_cast<std::uint64_t>(frame_id));
}

inline gtsam::Key ObjectPoseSymbol(ObjectId object_label, FrameId frame_id) {
  unsigned char label = object_label + '0';
  return L(label, static_cast<std::uint64_t>(frame_id));
}

enum BackendType : int {

  WCME = 0,             // world-centric motion estimator
  WCPE = 1,             // world-centric pose estimator
  HYBRID = 2,           // full-hybrid
  PARALLEL_HYBRID = 3,  // associated to its own special class
  // the following are test formulations that were not specifcially part of a
  // paper but were used for (internal) development/research. they may not work
  // as intended and are included for posterity
  TESTING_HYBRID_SD = 4,  // (SD) structureless-decoupled
  TESTING_HYBRID_D = 5,   // (D) decoupled
  TESTING_HYBRID_S = 6,   // (S) structureless
  TESTING_HYBRID_SMF = 7  // (SFM) smart motion factor
};

bool checkIfLabeledSymbol(gtsam::Key key);
bool reconstructMotionInfo(gtsam::Key key, ObjectId& object_label,
                           FrameId& frame_id);
bool reconstructPoseInfo(gtsam::Key key, ObjectId& object_label,
                         FrameId& frame_id);

template <typename T = gtsam::Pose3>
class UnaryPoseMeasurement : public UnaryMeasurement<T> {
 public:
  using Base = UnaryMeasurement<T>;
  using typename Base::Measurement;

  UnaryPoseMeasurement(Timestamp timestamp, const Measurement& measurement)
      : Base(timestamp, measurement) {}

  gtsam::Key makeSymbol(FrameId frame_id) override {
    return CameraPoseSymbol(frame_id);
  }
};

// TODO: this information is sort of duplicated when the ROS odometry messages
// are constructed.
//  streamline!!
struct TemporalObjectMetaData {
  //! ID of the object
  ObjectId object_id;

  FrameId first_seen;
  FrameId last_seen;
};

/**
 * @brief Helper class that allows functional callbacks to be triggered based on
 * the type of gtsam::Key provided, where the key should refer to a valid type
 * within the DynoSAM ecosystem.
 *
 * These include the symbols defined by CameraPoseSymbol, StaticLandmarkSymbol,
 * DynamicLandmarkSymbol, ObjectMotionSymbol and ObjectPoseSymbol.
 *
 * Callbacks can be registered to the class and then the operator will trigger
 * the callback based on the type, extracting and providing associated
 * meta-data.
 *
 * Not all callbacks for all symbols need to be registered.
 *
 */
class ApplyFunctionalSymbol {
 public:
  using CameraPoseFunc = std::function<void(FrameId, const gtsam::Symbol&)>;
  using ObjectMotionFunc =
      std::function<void(FrameId, ObjectId, const gtsam::LabeledSymbol&)>;
  using ObjectPoseFunc =
      std::function<void(FrameId, ObjectId, const gtsam::LabeledSymbol&)>;
  using StaticLmkFunc = std::function<void(TrackletId, const gtsam::Symbol&)>;
  using DynamicLmkFunc =
      std::function<void(TrackletId, const DynamicPointSymbol&)>;

  ApplyFunctionalSymbol() = default;
  virtual ~ApplyFunctionalSymbol() = default;

  bool operator()(gtsam::Key key) const;

  void reset();

  ApplyFunctionalSymbol& cameraPose(const CameraPoseFunc&);
  ApplyFunctionalSymbol& objectMotion(const ObjectMotionFunc&);
  ApplyFunctionalSymbol& objectPose(const ObjectPoseFunc&);
  ApplyFunctionalSymbol& staticLandmark(const StaticLmkFunc&);
  ApplyFunctionalSymbol& dynamicLandmark(const DynamicLmkFunc&);

 protected:
  CameraPoseFunc pose_func_;
  ObjectMotionFunc object_motion_func_;
  ObjectPoseFunc object_pose_func_;
  StaticLmkFunc static_lmk_func_;
  DynamicLmkFunc dynamic_lmk_func_;
};

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
  using GroundTruthPacketsRequest =
      std::function<std::optional<GroundTruthPacketMap>()>;

  GroundTruthPacketsRequest ground_truth_packets_request;
};

/**
 * @brief Data shared between a Formulation and its accessor
 *
 */
struct SharedFormulationData {
  const gtsam::Values* values;
  const FormulationHooks* hooks;

  SharedFormulationData(const gtsam::Values* v, const FormulationHooks* h)
      : values(v), hooks(h) {}
};

struct BackendMetaData {
  // TODO: should streamline this to only include what we actually need from the
  // params
  const BackendParams* backend_params = nullptr;
  //! Suffix that is used when logging data from a formulation
  //! This is additional to the suffix specified in formulation params in case
  //! further nameing specificity is needed; this is mostly helpful during
  //! testing
  std::string logging_suffix;
};

struct BackendSpinState {
  FrameId frame_id{0u};
  Timestamp timestamp{0.0};
  size_t iteration{0u};  //! Indexed from 1, such that when iteration==1, this
                         //! is the first iteration

  BackendSpinState() {}
  BackendSpinState(FrameId frame, Timestamp t, size_t itr)
      : frame_id(frame), timestamp(t), iteration(itr) {}
};

std::string DynoLikeKeyFormatter(gtsam::Key);
std::string DynoLikeKeyFormatterVerbose(gtsam::Key);

constexpr static SymbolChar InvalidDynoSymbol = '\0';

// TODO: not actually sure if this is necessary
// in this sytem we mix Symbol and LabelledSymbol so I just check which one the
// correct cast is and use that label, This will return InvalidDynoSymbol if a
// key cannot be constructed
SymbolChar DynoChrExtractor(gtsam::Key);

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

using SmartProjectionFactorParams = gtsam::SmartProjectionParams;

template <typename T>
using FactorMap = gtsam::FastMap<TrackletId, std::pair<T, Slot>>;

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
  ~BackendLogger();

  void logTrackletIdToObjectId(
      const gtsam::FastMap<TrackletId, ObjectId>& mapping);
  void logEllipsoids(const gtsam::FastMap<ObjectId, gtsam::Vector3>& mapping);

 private:
  std::string tracklet_to_object_id_file_name_;
  std::string ellipsoid_radii_file_name_;

  CsvWriter::UniquePtr tracklet_to_object_id_csv_;
  CsvWriter::UniquePtr ellipsoid_radii_csv_;
};

}  // namespace dyno
