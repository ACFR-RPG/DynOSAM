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

#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_common/utils/Metrics.hpp"

namespace dyno {

BackendType::BackendType(int int_enum)
    : BackendType(static_cast<Internal>(int_enum)) {}
BackendType::BackendType(const BackendType::Internal& interal)
    : type_(interal) {
  CHECK(!isExternalType());
}
BackendType::BackendType(const std::string& external) : type_(external) {
  CHECK(isExternalType());
}

bool BackendType::isExternalType() const {
  return std::holds_alternative<std::string>(type_);
}

bool BackendType::isInternalType() const {
  return std::holds_alternative<BackendType::Internal>(type_);
}

const std::string& BackendType::asExternalType() const {
  if (!isExternalType()) {
    throw IncorrectBackendTypeRequest(
        type_name<std::string>(), type_name<BackendType::Internal>(),
        to_string(std::get<BackendType::Internal>(type_)));
  }

  return std::get<std::string>(type_);
}

BackendType::Internal BackendType::asInternalType() const {
  if (isExternalType()) {
    throw IncorrectBackendTypeRequest(type_name<BackendType::Internal>(),
                                      type_name<std::string>(),
                                      std::get<std::string>(type_));
  }
  return std::get<BackendType::Internal>(type_);
}

bool BackendType::operator==(const BackendType::Internal& internal_type) const {
  if (!isExternalType()) {
    return asInternalType() == internal_type;
  }
  return false;
}

bool BackendType::operator!=(const BackendType::Internal& internal_type) const {
  return !(*this == internal_type);
}

bool BackendType::operator==(const std::string& external_type) const {
  if (isExternalType()) {
    return asExternalType() == external_type;
  }
  return false;
}

bool BackendType::operator!=(const std::string& external_type) const {
  return !(*this == external_type);
}

BackendType::operator std::string() const {
  std::stringstream ss;
  if (isExternalType()) {
    ss << asExternalType() << " (external type)";
  } else {
    ss << asInternalType() << " (internal type)";
  }
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const BackendType& backend_type) {
  os << (std::string)backend_type;
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const BackendType::Internal& internal_backend_type) {
  os << to_string(internal_backend_type);
  return os;
}

template <>
std::string to_string(const BackendType::Internal& internal_backend_type) {
  switch (internal_backend_type) {
    case BackendType::Internal::WCME:
      return "WCME";
    case BackendType::Internal::WCPE:
      return "WCPE";
    case BackendType::Internal::HYBRID:
      return "HYBRID";
    case BackendType::Internal::PARALLEL_HYBRID:
      return "PARALLEL-HYBRID";
    default:
      return "UNKNOWN BackendType::Internal";
      break;
  }
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
