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
#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/logger/Logger.hpp"

namespace dyno {

/// @brief Alias to a gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>
using PoseToPointFactor = gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>;

using SymbolChar = unsigned char;
static constexpr SymbolChar kPoseSymbolChar = 'X';
static constexpr SymbolChar kObjectMotionSymbolChar = 'H';
static constexpr SymbolChar kObjectPoseSymbolChar = 'L';
static constexpr SymbolChar kStaticLandmarkSymbolChar = 'l';
static constexpr SymbolChar kDynamicLandmarkSymbolChar = 'm';

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

bool checkIfLabeledSymbol(gtsam::Key key);
bool reconstructMotionInfo(gtsam::Key key, ObjectId& object_label,
                           FrameId& frame_id);
bool reconstructPoseInfo(gtsam::Key key, ObjectId& object_label,
                         FrameId& frame_id);

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
};

// /**
//  * @brief Internal back-end structure allowing Formulation/Accessors to get
//  meta-data
//  * about the backend via hooks
//  *
//  */
// struct InternalBackendMetaData {

//     using BackendParamsRequest = std::function<const BackendParams&()>;
//     using GroundTruthPacketsRequest = std::function<const
//     std::optional<GroundTruthPacketMap>&()>;

//     struct Hooks {
//         BackendParamsRequest backend_params_request;
//         GroundTruthPacketsRequest ground_truth_packets_request;
//     };

//     Hooks hooks;

// };

// better to change to hook (or pointer?) and parse as config
struct BackendMetaData {
  BackendParams params;
  std::optional<GroundTruthPacketMap> ground_truth_packets;
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
