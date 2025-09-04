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

#include "dynosam/backend/BackendDefinitions.hpp"

#include <gtsam/inference/LabeledSymbol.h>
#include <gtsam/inference/Symbol.h>

#include "dynosam/backend/DynamicPointSymbol.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/Metrics.hpp"

namespace dyno {

static bool internalReconstructInfo(gtsam::Key key, SymbolChar expected_chr,
                                    ObjectId& object_label, FrameId& frame_id) {
  // assume motion/pose key is a labelled symbol
  if (!checkIfLabeledSymbol(key)) {
    return false;
  }

  const gtsam::LabeledSymbol as_labeled_symbol(key);
  if (as_labeled_symbol.chr() != expected_chr) {
    return false;
  }

  frame_id = static_cast<FrameId>(as_labeled_symbol.index());

  SymbolChar label = as_labeled_symbol.label();
  object_label = label - '0';
  return true;
}

bool checkIfLabeledSymbol(gtsam::Key key) {
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  return (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0);
}

bool reconstructMotionInfo(gtsam::Key key, ObjectId& object_label,
                           FrameId& frame_id) {
  return internalReconstructInfo(key, kObjectMotionSymbolChar, object_label,
                                 frame_id);
}

bool reconstructPoseInfo(gtsam::Key key, ObjectId& object_label,
                         FrameId& frame_id) {
  return internalReconstructInfo(key, kObjectPoseSymbolChar, object_label,
                                 frame_id);
}

bool ApplyFunctionalSymbol::operator()(gtsam::Key key) const {
  const gtsam::Symbol sym(key);
  switch (sym.chr()) {
    case kPoseSymbolChar:
      if (pose_func_) {
        pose_func_(static_cast<FrameId>(sym.index()), sym);
      }
      return true;
    case kObjectMotionSymbolChar: {
      FrameId frame_id;
      ObjectId object_id;
      // attempt to get info about this key
      bool valid = reconstructMotionInfo(key, object_id, frame_id);
      // if valid and motion func registered, do call back
      if (valid && object_motion_func_)
        object_motion_func_(frame_id, object_id, gtsam::LabeledSymbol(key));
    }
      return true;
    case kObjectPoseSymbolChar: {
      FrameId frame_id;
      ObjectId object_id;
      // attempt to get info about this key
      bool valid = reconstructPoseInfo(key, object_id, frame_id);
      // if valid and motion func registered, do call back
      if (valid && object_pose_func_)
        object_pose_func_(frame_id, object_id, gtsam::LabeledSymbol(key));
    }
      return true;
    case kStaticLandmarkSymbolChar: {
      if (static_lmk_func_) {
        static_lmk_func_(static_cast<TrackletId>(sym.index()), sym);
      }
    }
      return true;
    case kDynamicLandmarkSymbolChar: {
      if (dynamic_lmk_func_) {
        DynamicPointSymbol dps(key);
        dynamic_lmk_func_(dps.trackletId(), dps);
      }
    }
      return true;

    default:
      return false;
  }
}

void ApplyFunctionalSymbol::reset() {
  pose_func_ = nullptr;
  object_motion_func_ = nullptr;
  object_pose_func_ = nullptr;
  static_lmk_func_ = nullptr;
  dynamic_lmk_func_ = nullptr;
}

ApplyFunctionalSymbol& ApplyFunctionalSymbol::cameraPose(
    const CameraPoseFunc& func) {
  pose_func_ = func;
  return *this;
}

ApplyFunctionalSymbol& ApplyFunctionalSymbol::objectMotion(
    const ObjectMotionFunc& func) {
  object_motion_func_ = func;
  return *this;
}

ApplyFunctionalSymbol& ApplyFunctionalSymbol::objectPose(
    const ObjectPoseFunc& func) {
  object_pose_func_ = func;
  return *this;
}
ApplyFunctionalSymbol& ApplyFunctionalSymbol::staticLandmark(
    const StaticLmkFunc& func) {
  static_lmk_func_ = func;
  return *this;
}
ApplyFunctionalSymbol& ApplyFunctionalSymbol::dynamicLandmark(
    const DynamicLmkFunc& func) {
  dynamic_lmk_func_ = func;
  return *this;
}

static std::optional<FrameId> getPoseMotionFrameId(gtsam::Key key) {
  static ApplyFunctionalSymbol afs;
  std::optional<FrameId> frame_id_of_symbol;
  afs.cameraPose([&frame_id_of_symbol](FrameId frame_id, const gtsam::Symbol&) {
    frame_id_of_symbol = frame_id;
  });
  afs.objectMotion([&frame_id_of_symbol](FrameId frame_id, ObjectId,
                                         const gtsam::LabeledSymbol&) {
    frame_id_of_symbol = frame_id;
  });

  afs(key);
  return frame_id_of_symbol;
}

NoiseModels NoiseModels::fromBackendParams(
    const BackendParams& backend_params) {
  NoiseModels noise_models;

  // odometry
  gtsam::Vector6 odom_sigmas;
  odom_sigmas.head<3>().setConstant(backend_params.odometry_rotation_sigma_);
  odom_sigmas.tail<3>().setConstant(backend_params.odometry_translation_sigma_);
  noise_models.odometry_noise =
      gtsam::noiseModel::Diagonal::Sigmas(odom_sigmas);
  CHECK(noise_models.odometry_noise);

  // first pose prior (world frame)
  noise_models.initial_pose_prior =
      gtsam::noiseModel::Isotropic::Sigma(6u, 0.000001);
  CHECK(noise_models.initial_pose_prior);

  // landmark motion noise (needed for some formulations ie world-centric)
  noise_models.landmark_motion_noise = gtsam::noiseModel::Isotropic::Sigma(
      3u, backend_params.motion_ternary_factor_noise_sigma_);
  CHECK(noise_models.landmark_motion_noise);

  // smoothing factor noise model (can be any variant of the smoothing factor as
  // long as the dimensions are 6, ie. pose)
  gtsam::Vector6 object_constant_vel_sigmas;
  object_constant_vel_sigmas.head<3>().setConstant(
      backend_params.constant_object_motion_rotation_sigma_);
  object_constant_vel_sigmas.tail<3>().setConstant(
      backend_params.constant_object_motion_translation_sigma_);
  noise_models.object_smoothing_noise =
      gtsam::noiseModel::Diagonal::Sigmas(object_constant_vel_sigmas);
  CHECK(noise_models.object_smoothing_noise);

  // TODO: CHECKS that values are not zero!!!

  // TODO: should now depricate if we're using covariance from frontend??
  noise_models.static_point_noise = gtsam::noiseModel::Isotropic::Sigma(
      3u, backend_params.static_point_noise_sigma);
  noise_models.dynamic_point_noise = gtsam::noiseModel::Isotropic::Sigma(
      3u, backend_params.dynamic_point_noise_sigma);

  if (backend_params.use_robust_kernals_) {
    LOG(INFO) << "Using robust huber loss function: "
              << backend_params.k_huber_3d_points_;

    if (backend_params.static_point_noise_as_robust) {
      LOG(INFO) << "Making static point noise model robust!";
      noise_models.static_point_noise = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(
              backend_params.k_huber_3d_points_),
          noise_models.static_point_noise);
    }

    // TODO: JUST FOR TESTING!!!
    if (backend_params.dynamic_point_noise_as_robust) {
      LOG(INFO) << "Making dynamic point noise model robust!";
      noise_models.dynamic_point_noise = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(
              backend_params.k_huber_3d_points_),
          noise_models.dynamic_point_noise);
    }

    // TODO: not k_huber_3d_points_ not just used for 3d points
    noise_models.landmark_motion_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(
            backend_params.k_huber_3d_points_),
        noise_models.landmark_motion_noise);
  }

  return noise_models;
}

void NoiseModels::print(const std::string& name) const {
  auto print_impl = [](gtsam::SharedNoiseModel model,
                       const std::string& name) -> void {
    if (model) {
      model->print(name);
    } else {
      std::cout << "Noise model " << name << " is null!";
    }
  };

  print_impl(initial_pose_prior, "Pose Prior ");
  print_impl(odometry_noise, "VO ");
  print_impl(landmark_motion_noise, "Landmark Motion ");
  print_impl(object_smoothing_noise, "Object Smoothing ");
  print_impl(dynamic_point_noise, "Dynamic Point ");
  print_impl(static_point_noise, "Static Point ");
}

std::string DynoLikeKeyFormatter(gtsam::Key key) {
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  if (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0) {
    // if used as motion
    if (asLabeledSymbol.chr() == kObjectMotionSymbolChar ||
        asLabeledSymbol.chr() == kObjectPoseSymbolChar) {
      return (std::string)asLabeledSymbol;
    }
    return (std::string)asLabeledSymbol;
  }

  const gtsam::Symbol asSymbol(key);
  if (asLabeledSymbol.chr() > 0) {
    if (asLabeledSymbol.chr() == kDynamicLandmarkSymbolChar) {
      const DynamicPointSymbol asDynamicPointSymbol(key);
      return (std::string)asDynamicPointSymbol;
    } else {
      return (std::string)asSymbol;
    }

  } else {
    return std::to_string(key);
  }
}

SymbolChar DynoChrExtractor(gtsam::Key key) {
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  if (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0) {
    return asLabeledSymbol.chr();
  }
  const gtsam::Symbol asSymbol(key);
  if (asLabeledSymbol.chr() > 0) {
    return asSymbol.chr();
  } else {
    return InvalidDynoSymbol;
  }
}

std::string DynoLikeKeyFormatterVerbose(gtsam::Key key) {
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  if (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0) {
    // if used as motion
    if (asLabeledSymbol.chr() == kObjectMotionSymbolChar) {
      ObjectId object_label;
      FrameId frame_id;
      CHECK(reconstructMotionInfo(asLabeledSymbol, object_label, frame_id));

      std::stringstream ss;
      ss << "H: label" << object_label << ", frames: " << frame_id - 1 << " -> "
         << frame_id;
      return ss.str();
    } else if (asLabeledSymbol.chr() == kObjectPoseSymbolChar) {
      ObjectId object_label;
      FrameId frame_id;
      CHECK(reconstructPoseInfo(asLabeledSymbol, object_label, frame_id));

      std::stringstream ss;
      ss << "K: label" << object_label << ", frame: " << frame_id;
      return ss.str();
    }
    return (std::string)asLabeledSymbol;
  }

  const gtsam::Symbol asSymbol(key);
  if (asLabeledSymbol.chr() > 0) {
    if (asLabeledSymbol.chr() == kDynamicLandmarkSymbolChar) {
      const DynamicPointSymbol asDynamicPointSymbol(key);

      FrameId frame_id = asDynamicPointSymbol.frameId();
      TrackletId tracklet_id = asDynamicPointSymbol.trackletId();
      std::stringstream ss;
      ss << kDynamicLandmarkSymbolChar << ": frame " << frame_id
         << ", tracklet " << tracklet_id;
      return ss.str();

    } else {
      return (std::string)asSymbol;
    }

  } else {
    return std::to_string(key);
  }
}

DebugInfo::ObjectInfo::operator std::string() const {
  std::stringstream ss;
  ss << "Num point factors: " << num_dynamic_factors << "\n";
  ss << "Num point variables: " << num_new_dynamic_points << "\n";
  ss << "Num motion factors: " << num_motion_factors << "\n";
  ss << "Smoothing factor added: " << std::boolalpha << smoothing_factor_added;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os,
                         const DebugInfo::ObjectInfo& object_info) {
  os << (std::string)object_info;
  return os;
}

DebugInfo::ObjectInfo& DebugInfo::getObjectInfo(ObjectId object_id) {
  return getObjectInfoImpl(object_id);
}

const DebugInfo::ObjectInfo& DebugInfo::getObjectInfo(
    ObjectId object_id) const {
  return getObjectInfoImpl(object_id);
}

BackendLogger::BackendLogger(const std::string& name_prefix)
    : EstimationModuleLogger(name_prefix + "_backend"),
      tracklet_to_object_id_file_name_("tracklet_to_object_id.csv") {
  ellipsoid_radii_file_name_ = module_name_ + "_ellipsoid_radii.csv";

  tracklet_to_object_id_csv_ =
      std::make_unique<CsvWriter>(CsvHeader("tracklet_id", "object_id"));

  ellipsoid_radii_csv_ =
      std::make_unique<CsvWriter>(CsvHeader("object_id", "a", "b", "c"));
}

void BackendLogger::logTrackletIdToObjectId(
    const gtsam::FastMap<TrackletId, ObjectId>& mapping) {
  for (const auto& [tracklet_id, object_id] : mapping) {
    *tracklet_to_object_id_csv_ << tracklet_id << object_id;
  }
}

void BackendLogger::logEllipsoids(
    const gtsam::FastMap<ObjectId, gtsam::Vector3>& mapping) {
  for (const auto& [object_id, radii] : mapping) {
    *ellipsoid_radii_csv_ << object_id << radii(0) << radii(1) << radii(2);
  }
}

BackendLogger::~BackendLogger() {
  OfstreamWrapper::WriteOutCsvWriter(*ellipsoid_radii_csv_,
                                     ellipsoid_radii_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*tracklet_to_object_id_csv_,
                                     tracklet_to_object_id_file_name_);
}

}  // namespace dyno
