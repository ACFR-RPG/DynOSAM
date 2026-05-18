#include "dynosam/backend/Formulation.hpp"

namespace dyno {

void logFromAccessor(Accessor::Ptr accessor, BackendLogger& logger,
                     const std::optional<GroundTruthPacketMap>& ground_truth) {
  CHECK(accessor);
  const PoseTrajectory& camera_trajectory = accessor->getCameraTrajectory();
  logger.logCameraPose(camera_trajectory, ground_truth);

  const MultiObjectTrajectories& object_trajectories =
      accessor->getMultiObjectTrajectories();
  logger.logObjectTrajectory(object_trajectories, ground_truth);

  auto static_map = accessor->getFullStaticMap();
  auto dynamic_map = accessor->getFullTemporalDynamicMap();

  logger.logMapPoints(static_map);
  logger.logMapPoints(dynamic_map);
}

Formulation::Formulation(const FormulationParams& params,
                         const NoiseModels& noise_models,
                         const Sensors& sensors, const FormulationHooks& hooks)
    : params_(params),
      noise_models_(noise_models),
      sensors_(sensors),
      shared_data_(std::make_shared<SharedFormulationData>()) {
  shared_data_->hooks = hooks;
}

void Formulation::setTheta(const gtsam::Values& linearization) {
  shared_data_->threadSafeSetTheta(linearization);
}

void Formulation::updateTheta(const gtsam::Values& linearization) {
  shared_data_->threadSafeInsertOrAssignTheta(linearization);
}

bool Formulation::exists(gtsam::Key key) const {
  const std::lock_guard<std::mutex> lock(shared_data_->theta_mutex);
  const auto& theta = shared_data_->theta;
  return theta.exists(key);
}

BackendLogger::UniquePtr Formulation::makeFullyQualifiedLogger() const {
  return std::make_unique<BackendLogger>(getFullyQualifiedName());
}

Accessor::Ptr Formulation::accessorFromTheta() const {
  if (!accessor_theta_) {
    accessor_theta_ = createAccessor(shared_data_);
  }
  return accessor_theta_;
}

std::string Formulation::setFullyQualifiedName() const {
  // get the derived name of the formulation
  std::string logger_prefix = this->loggerPrefix();
  const std::string suffix = params_.updater_suffix;

  // add suffix to name if required
  if (!suffix.empty()) {
    logger_prefix += ("_" + suffix);
  }
  fully_qualified_name_ = logger_prefix;
  return *fully_qualified_name_;
}

void Formulation::logBackendFromMap(
    const FormulationLoggingParams& backend_info) {
  // TODO:
  std::string logger_prefix = this->getFullyQualifiedName();
  const std::string suffix = backend_info.logging_suffix;

  // add suffix to name if required
  if (!suffix.empty()) {
    logger_prefix += ("_" + suffix);
  }
  BackendLogger::UniquePtr logger =
      std::make_unique<BackendLogger>(logger_prefix);

  auto accessor = this->accessorFromTheta();

  CHECK(hooks().ground_truth_packets_request);

  logFromAccessor(accessor, *logger, hooks().ground_truth_packets_request());

  logger.reset();
}

}  // namespace dyno
