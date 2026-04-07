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

  // TODO: do we still need the full batch hack? Hardly ever use FB anymore but
  //  maybe fore backwards compatability!?

  // TODO: formulation params are now backend params so no longer need to
  //  pass backend params into Formulation with FormulationLoggingParams
  //  CHECK_NOTNULL(backend_info.backend_params);
  //  const auto& backend_params = *backend_info.backend_params;

  // const ObjectPoseMap object_pose_map = accessor->getObjectPoses();

  // for (FrameId frame_k : map->getFrameIds()) {
  //   // TODO: hack - only go up to frames < full batch so we actually only
  //   // include the optimised alues
  //   // TODO: actually should be based on the optimization mode!!
  //   if (params_.optimization_mode == RegularOptimizationType::FULL_BATCH &&
  //       params_.full_batch_frame - 1 == (int)frame_k) {
  //     break;
  //   }

  //   std::stringstream ss;
  //   ss << "Logging data from map at frame " << frame_k;

  //   // get MotionestimateMap
  //   //  const MotionEstimateMap motions = map->getMotionEstimates(frame_k);
  //   {
  //     const MotionEstimateMap motions = accessor->getObjectMotions(frame_k);
  //     auto result =
  //         logger->logObjectMotion(frame_k, motions, ground_truth_packets);
  //     if (result)
  //       ss << " Logged " << *result << " motions from " << motions.size()
  //          << " computed motions.";
  //     else
  //       ss << " Could not log object motions.";
  //   }

  //   StateQuery<gtsam::Pose3> X_k_query = accessor->getSensorPose(frame_k);

  //   if (X_k_query) {
  //     logger->logCameraPose(frame_k, X_k_query.get(), ground_truth_packets);
  //   } else {
  //     LOG(WARNING) << "Could not log camera pose estimate at frame " <<
  //     frame_k;
  //   }

  //   // TODO: log!!
  //   //  logger->logObjectPose(object_pose_map, ground_truth_packets);

  //   if (map->frameExists(frame_k)) {
  //     auto static_map = accessor->getStaticLandmarkEstimates(frame_k);
  //     auto dynamic_map = accessor->getDynamicLandmarkEstimates(frame_k);

  //     CHECK(X_k_query);  // actually not needed for points in world!!
  //     logger->logPoints(frame_k, *X_k_query, static_map);
  //     logger->logPoints(frame_k, *X_k_query, dynamic_map);
  //   }

  //   LOG(INFO) << ss.str();
  // }

  logger.reset();
}

}  // namespace dyno
