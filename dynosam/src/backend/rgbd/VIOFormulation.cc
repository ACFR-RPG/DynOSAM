#include "dynosam/backend/rgbd/VIOFormulation.hpp"

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>

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

struct VIOUpdater {
 public:
  VIOUpdater(VIOFormulation* vio_formulation)
      : vio_formulation_(CHECK_NOTNULL(vio_formulation)) {}
  virtual ~VIOUpdater() = default;

 protected:
  // some helper base functions
  bool isRobust() const {
    return vio_formulation_->params().makeStaticMeasurementsRobust();
  }

  // NOTE: pass pointer by reference as we want to change the object the
  // noise_model points too
  //  since robustifyHuber returns a new model
  void robustifyHuber(gtsam::SharedNoiseModel& noise_model) {
    if (isRobust()) {
      noise_model = factor_graph_tools::robustifyHuber(
          vio_formulation_->params().k_huber_3d_points_, noise_model);
    }
  }

  bool isPointAdded(gtsam::Key point_key) {
    return vio_formulation_->is_other_values_in_map.exists(point_key);
  }

  void markPointAsAdded(gtsam::Key point_key) {
    vio_formulation_->is_other_values_in_map.insert2(point_key, true);
  }

 protected:
  VIOFormulation* vio_formulation_;
};

class VIOUpdaterImpl : public VIOUpdater {
 public:
  using MapTraits = VIOFormulation::MapTraitsType;
  using LmkNode = MapTraits::LandmarkNodePtr;
  using FrameNode = MapTraits::FrameNodePtr;
  using MeasurementType = MapTraits::MeasurementType;
  using MeasurementTraits = measurement_traits<MeasurementType>;

  VIOUpdaterImpl(VIOFormulation* vio_formulation)
      : VIOUpdater(vio_formulation) {}

  /**
   * @brief
   *
   * Return results indicates if any factors/values were added (ie. the point
   * was added).
   *
   * @param lmk Landmark is passed by reference as some internal flags may be
   * changed
   * @param frame
   * @param update_params
   * @param values
   * @param graph
   * @param point_key
   * @param result
   * @param initial
   * @return true
   * @return false
   */
  virtual bool addLandmark(LmkNode& lmk, const FrameNode& frame,
                           const UpdateObservationParams& update_params,
                           gtsam::Values& values,
                           gtsam::NonlinearFactorGraph& graph,
                           gtsam::Key& point_key,
                           UpdateObservationResult& result,
                           std::optional<Landmark>& initial) = 0;
};

class PTPUpdater : public VIOUpdaterImpl {
 public:
  PTPUpdater(VIOFormulation* vio_formulation)
      : VIOUpdaterImpl(vio_formulation) {}

  bool addLandmark(LmkNode& lmk, const FrameNode& frame,
                   const UpdateObservationParams& update_params,
                   gtsam::Values& values, gtsam::NonlinearFactorGraph& graph,
                   gtsam::Key& point_key, UpdateObservationResult& result,
                   std::optional<Landmark>& initial) override {
    point_key = lmk->makeStaticKey();
    const FrameId frame_k = FrameId(frame->getId());

    const auto& params = this->vio_formulation_->params();

    if (this->isPointAdded(point_key)) {
      CHECK(lmk->added_to_opt);
      const auto pose_key = frame->makePoseKey();

      auto [measured_point_local, measurement_covariance] =
          MeasurementTraits::pointWithCovariance(lmk->getMeasurement(frame_k));
      CHECK_NOTNULL(measurement_covariance);

      this->robustifyHuber(measurement_covariance);

      auto factor =
          boost::make_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
              pose_key, point_key, measured_point_local,
              measurement_covariance);
      graph.add(factor);
      result.updateAffectedObject(frame_k, 0);
      return true;
    } else {
      CHECK(!lmk->added_to_opt);

      if (lmk->numObservations() < params.min_static_observations) {
        return false;
      }

      // this condition should only run once per tracklet (ie.e the first time
      // the tracklet has enough observations) we gather the tracklet
      // observations and then initalise it in the new values these should
      // then get added to the map and map_->exists() should return true for
      // all other times
      const auto& seen_frames = lmk->getSeenFrames();
      for (const auto& seen_frame : seen_frames) {
        FrameId seen_frame_id = FrameId(seen_frame->getId());
        // only iterate up to the query frame
        if (seen_frame_id > frame_k) {
          break;
        }

        // if we should not backtrack, only add the current frame!!!
        const auto do_backtrack = update_params.do_backtrack;
        if (!do_backtrack && seen_frame_id < frame_k) {
          continue;
        }

        const gtsam::Key pose_key = seen_frame->makePoseKey();
        auto [measured_point_local, measurement_covariance] =
            MeasurementTraits::pointWithCovariance(
                lmk->getMeasurement(seen_frame_id));
        CHECK_NOTNULL(measurement_covariance);

        this->robustifyHuber(measurement_covariance);

        auto factor = boost::make_shared<
            gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            pose_key, point_key, measured_point_local, measurement_covariance);

        graph.add(factor);
        result.updateAffectedObject(seen_frame_id, 0);
      }

      const Landmark& measured =
          MeasurementTraits::point(lmk->getMeasurement(frame_k));

      // TODO: should use getInitialOrLinearizedSensorPose
      gtsam::Pose3 T_W_X;
      CHECK(
          this->vio_formulation_->map()->hasInitialSensorPose(frame_k, &T_W_X));

      Landmark initial_point = T_W_X * measured;
      initial = initial_point;
      result.updateAffectedObject(frame_k, 0);

      values.insert(point_key, initial_point);
      this->markPointAsAdded(point_key);
      lmk->added_to_opt = true;
      return true;
    }
  }
};

class GenericProjectionUpdater : public VIOUpdaterImpl {
 public:
  GenericProjectionUpdater(VIOFormulation* vio_formulation)
      : VIOUpdaterImpl(vio_formulation) {}

  bool addLandmark(LmkNode& lmk, const FrameNode& frame,
                   const UpdateObservationParams& update_params,
                   gtsam::Values& values, gtsam::NonlinearFactorGraph& graph,
                   gtsam::Key& point_key, UpdateObservationResult& result,
                   std::optional<Landmark>& initial) override {
    LOG(FATAL) << "Not implemented";
  }
};

class StereoProjectionUpdater : public VIOUpdaterImpl {
 public:
  StereoProjectionUpdater(VIOFormulation* vio_formulation)
      : VIOUpdaterImpl(vio_formulation) {
    std::shared_ptr<Camera> camera =
        CHECK_NOTNULL(vio_formulation_->sensors().camera);
    std::shared_ptr<RGBDCamera> rgbd_camera =
        CHECK_NOTNULL(camera->safeGetRGBDCamera());
    K_stereo_ = rgbd_camera->getFakeStereoCalib();
    CHECK_NOTNULL(K_stereo_);
    K_ = camera->getGtsamCalibration();
  }

  bool addLandmark(LmkNode& lmk, const FrameNode& frame,
                   const UpdateObservationParams& update_params,
                   gtsam::Values& values, gtsam::NonlinearFactorGraph& graph,
                   gtsam::Key& point_key, UpdateObservationResult& result,
                   std::optional<Landmark>& initial) override {
    point_key = lmk->makeStaticKey();
    const FrameId frame_k = FrameId(frame->getId());

    // NOTE: not handling the case a lmk becomes an outlier once already added
    // to the opt
    //  hoping the robust cost funcion handles this!
    if (this->isPointAdded(point_key)) {
      CHECK(lmk->added_to_opt);
      const auto pose_key = frame->makePoseKey();

      auto stereo_measurement =
          MeasurementTraits::stereo(lmk->getMeasurement(frame_k));
      // FOR NOW
      CHECK(stereo_measurement);
      auto [measurement, model] = *stereo_measurement;

      this->robustifyHuber(model);

      auto factor = boost::make_shared<GenericStereoFactor>(
          measurement, model, pose_key, point_key, K_stereo_);

      graph.add(factor);
      result.updateAffectedObject(frame_k, 0);
      return true;
    } else {
      CHECK(!lmk->added_to_opt);

      using GtsamCamera = Camera::CameraImpl;
      CameraSet<GtsamCamera> camera_set;
      gtsam::Point2Vector measurements;
      gtsam::SharedNoiseModel model;

      // triangulate stereo measurements by treating each stereocamera as a
      // pair of monocular cameras this is vital for good triangulation!
      const auto& seen_frames = lmk->getSeenFrames();
      for (const auto& frame_node_i : seen_frames) {
        FrameId frame_id_i = frame_node_i->getId();
        // use the initial pose
        // in the IMU case the optimised pose will be not so good until visual
        // odom starts working... or maybe not...
        gtsam::Pose3 X_W_i;
        CHECK(this->vio_formulation_->map()->hasInitialSensorPose(frame_id_i,
                                                                  &X_W_i))
            << "Missing initial pose at k=" << frame_id_i;
        // TODO: hack for now - in the KF case, we sometimes need to add
        // KF in the past so we already have measurements at k+1 but not yet
        //  an initial pose measurement as the frontend has not send it
        //  for now just skip!
        //  if(!this->vio_formulation_->map()->hasInitialSensorPose(
        //    frame_id_i, &X_W_i)) { continue; }

        const gtsam::Pose3 leftPose = X_W_i;
        const gtsam::Cal3_S2 monoCal = K_stereo_->calibration();
        const GtsamCamera leftCamera_i(leftPose, monoCal);
        const gtsam::Pose3 left_Pose_right = gtsam::Pose3(
            gtsam::Rot3(), gtsam::Point3(K_stereo_->baseline(), 0.0, 0.0));
        const gtsam::Pose3 rightPose = leftPose.compose(left_Pose_right);
        const GtsamCamera rightCamera_i(rightPose, monoCal);

        // gtsam::Pose3 X_W_i =
        //     this->vio_formulation_->getInitialOrLinearizedSensorPose(frame_id_i);

        // updates the model each time, just uses the last one!
        // auto [keypoint, model] = MeasurementTraits::keypointWithCovariance(
        //     lmk->getMeasurement(frame_id_i));
        auto stereo_measurement =
            MeasurementTraits::stereo(lmk->getMeasurement(frame_id_i));
        CHECK(stereo_measurement);
        auto [zi, model] = *stereo_measurement;

        camera_set.push_back(leftCamera_i);
        measurements.push_back(Point2(zi.uL(), zi.v()));
        if (!std::isnan(zi.uR())) {  // if right point is valid
          camera_set.push_back(rightCamera_i);
          measurements.push_back(Point2(zi.uR(), zi.v()));
        }
      }

      gtsam::TriangulationParameters triangulation_params;
      // triangulation_params.useLOST = true;
      triangulation_params.noiseModel = model;
      auto triangulation_result = gtsam::triangulateSafe<GtsamCamera>(
          camera_set, measurements, triangulation_params);

      if (triangulation_result) {
        const gtsam::Point3 initial_point = *triangulation_result;
        initial = initial_point;

        double reprojection_error =
            camera_set.reprojectionError(initial_point, measurements).norm();

        // if error is too large, discard point
        if (reprojection_error > 3.0) {
          // mark as outlier for the front-end
          lmk->inlier = false;
          return false;
        }
        // collect factors
        gtsam::NonlinearFactorGraph stereo_factors;
        FrameIds frames_with_good_factors;
        for (const auto& frame_node_i : seen_frames) {
          const auto pose_key_i = frame_node_i->makePoseKey();
          FrameId frame_id_i = frame_node_i->getId();

          auto stereo_measurement =
              MeasurementTraits::stereo(lmk->getMeasurement(frame_id_i));
          // FOR NOW
          CHECK(stereo_measurement);
          auto [measurement, model] = *stereo_measurement;

          double disparity = measurement.uL() - measurement.uR();
          if (disparity > 0.5) {
            this->robustifyHuber(model);
            auto factor = boost::make_shared<GenericStereoFactor>(
                measurement, model, pose_key_i, point_key, K_stereo_);
            stereo_factors += factor;
            frames_with_good_factors.push_back(frame_id_i);
          }
        }

        if (stereo_factors.size() < 2) {
          return false;
        }

        for (const auto good_frame_id : frames_with_good_factors) {
          result.updateAffectedObject(good_frame_id, 0);
        }
        graph += stereo_factors;

        values.insert(point_key, initial_point);
        this->markPointAsAdded(point_key);
        lmk->added_to_opt = true;

        return true;
      } else {
        // mark as outlier for the front-end
        lmk->inlier = false;
        return false;
      }
    }
  }

 private:
  Camera::CalibrationType::shared_ptr K_;
  StereoCalibPtr K_stereo_;
};

VIOFormulation::VIOFormulation(const FormulationParams& params,
                               typename Map::Ptr map,
                               const NoiseModels& noise_models,
                               const Sensors& sensors,
                               const FormulationHooks& hooks)
    : Base(params, map, noise_models, sensors, hooks) {
  init_vel_prior_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, 1e-5);

  gtsam::Vector6 prior_imu_bias_sigmas;
  prior_imu_bias_sigmas.head<3>().setConstant(0.1);
  prior_imu_bias_sigmas.tail<3>().setConstant(0.01);
  init_imu_bias_prior_noise_ =
      gtsam::noiseModel::Diagonal::Sigmas(prior_imu_bias_sigmas);

  const StaticFormulationType& static_method_type = params.static_formulation;
  switch (static_method_type) {
    case StaticFormulationType::PTP:
      VLOG(20) << "Using Point-to-Pose formulation for Visual SLAM";
      static_updater_ = std::make_unique<PTPUpdater>(this);
      break;
    case StaticFormulationType::GENERIC_PROJECTION:
      VLOG(20) << "Using Generic Projection formulation for Visual SLAM";
      static_updater_ = std::make_unique<GenericProjectionUpdater>(this);
      break;
    case StaticFormulationType::STEREO_PROJECTION:
      VLOG(20) << "Using Stereo Projection formulation for Visual SLAM";
      static_updater_ = std::make_unique<StereoProjectionUpdater>(this);
      break;
    default:
      LOG(FATAL) << "Unknown method type for Static Formulation!";
  }
}

UpdateObservationResult VIOFormulation::updateStaticObservations(
    FrameId frame_id_k, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors,
    const UpdateObservationParams& update_params) {
  typename Map::Ptr map = this->map();
  auto accessor = this->accessorFromTheta();

  // keep track of the new factors added in this function
  // these are then appended to the internal factors_ and new_factors
  gtsam::NonlinearFactorGraph internal_new_factors;

  UpdateObservationResult result(update_params);

  const size_t initial_factors_size = new_factors.size();
  const size_t initial_values_size = new_values.size();

  auto frame_node_k = map->getFrame(frame_id_k);
  CHECK_NOTNULL(frame_node_k);

  const auto& static_method_type = params().static_formulation;

  VLOG(20) << "Looping over " << frame_node_k->static_landmarks.size()
           << " static lmks for frame " << frame_id_k;
  for (auto lmk_node : frame_node_k->static_landmarks) {
    gtsam::Key point_key;
    std::optional<Landmark> initial_value;

    switch (static_method_type) {
      case StaticFormulationType::PTP:
        updaterAs<PTPUpdater>()->addLandmark(
            lmk_node, frame_node_k, update_params, new_values,
            internal_new_factors, point_key, result, initial_value);
        break;
      case StaticFormulationType::GENERIC_PROJECTION:
        updaterAs<GenericProjectionUpdater>()->addLandmark(
            lmk_node, frame_node_k, update_params, new_values,
            internal_new_factors, point_key, result, initial_value);
        break;
      case StaticFormulationType::STEREO_PROJECTION:
        updaterAs<StereoProjectionUpdater>()->addLandmark(
            lmk_node, frame_node_k, update_params, new_values,
            internal_new_factors, point_key, result, initial_value);
        break;
      default:
        LOG(FATAL) << "Unknown method type for Static Formulation!";
    }

    CHECK_EQ(point_key, lmk_node->makeStaticKey());
  }

  if (result.debug_info) {
    result.debug_info->num_static_factors =
        internal_new_factors.size() - initial_factors_size;
    result.debug_info->num_new_static_points =
        new_values.size() - initial_values_size;
  }

  // update internal data structures
  shared_data_->threadSafeInsertOrAssignTheta(new_values);
  factors_ += internal_new_factors;
  new_factors += internal_new_factors;

  if (result.debug_info)
    LOG(INFO) << "Num new static points: "
              << result.debug_info->num_new_static_points
              << " Num new static factors "
              << result.debug_info->num_static_factors;
  return result;
}

gtsam::NavState VIOFormulation::addStatesInitalise(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& X_W_k,
    const gtsam::Vector3& V_W_k) {
  this->addSensorPose(new_values, frame_id_k, X_W_k);
  this->addSensorPosePrior(new_factors, frame_id_k, X_W_k,
                           noise_models_.initial_pose_prior);

  gtsam::imuBias::ConstantBias initial_bias;  // TODO: make param
  initial_imu_bias_ = initial_bias;

  initial_nav_state_ = gtsam::NavState(X_W_k, V_W_k);

  // // add body velocity state
  // this->addValue(new_values, V_W_k, velocity_key);
  // // add bias state
  // this->addValue(new_values, initial_bias, imu_bias_key);

  // this->addFactor(new_factors,
  //                 boost::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
  //                     velocity_key, V_W_k, init_vel_prior_noise_));
  // this->addFactor(
  //     new_factors,
  //     boost::make_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
  //         imu_bias_key, initial_bias, init_imu_bias_prior_noise_));

  first_frame_ = frame_id_k;
  last_propogate_frame_ = frame_id_k;
  last_propogate_time_ = timestamp_k;
  // auto nav_state_query = accessor->getNavState();
  // CHECK(nav_state_query);
  // return nav_state_query.value();
  // return DYNO_GET_QUERY_DEBUG(accessor->getNavState(frame_id_k));
  return initial_nav_state_;
}

gtsam::NavState VIOFormulation::addStatesPropogate(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& T_k_1_k,
    const ImuFrontend::PimPtr& pim) {
  const FrameId from_frame = last_propogate_frame_;
  const FrameId to_frame = frame_id_k;

  CHECK_GT(to_frame, 0);
  CHECK_GT(to_frame, from_frame)
      << "State frame id's are not incrementally ascending. Are we dropping "
         "inertial/VO data per frame?";

  gtsam::NavState nav_state_k;
  bool predict_imu = false;
  if (pim) {
    // first propogate frame
    // check if we have IMU and initalise state as such
    if (first_frame_ == last_propogate_frame_) {
      imu_states_initalise_ = true;
      // only now do we add velocity and IMU bias at k=first_frame_
      // this ensures we dont add imu specific states unless we have
      // measurements from the IMU
      addImuStatesFromInitialNavState(new_values, new_factors);
      LOG(INFO) << "Initised VIO Formulation to use IMU";
    }

    CHECK(isImuInitalized())
        << "Inconsistent VisionImu state - Preintegration recieved "
           "at frame "
        << frame_id_k << " but formulation is not IMU initalized!";

    nav_state_k = predictAndAddFactorsIMU(new_values, new_factors, frame_id_k,
                                          timestamp_k, pim);
  } else {
    CHECK(!isImuInitalized());
    nav_state_k = predictAndAddFactorsVO(new_values, new_factors, frame_id_k,
                                         timestamp_k, T_k_1_k);
  }

  // update frame/timestamp data
  last_propogate_frame_ = frame_id_k;
  last_propogate_time_ = timestamp_k;

  return nav_state_k;
}

void VIOFormulation::addSensorPosePrior(
    gtsam::NonlinearFactorGraph& new_factors, FrameId frame_id_k,
    const gtsam::Pose3& X_W_k, gtsam::SharedNoiseModel noise_model) {
  CHECK(noise_model);
  CHECK_EQ(noise_model->dim(), gtsam::traits<gtsam::Pose3>::dimension);

  this->addFactor(new_factors,
                  boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                      CameraPoseSymbol(frame_id_k), X_W_k, noise_model));
}

void VIOFormulation::addSensorPose(gtsam::Values& new_values,
                                   FrameId frame_id_k,
                                   const gtsam::Pose3& X_W_k) {
  this->addValue(new_values, X_W_k, CameraPoseSymbol(frame_id_k));
}

VIOAccessor::Ptr VIOFormulation::getAsVIOAccessor() const {
  VIOAccessor::Ptr vio_accessor = this->derivedAccessor<VIOAccessor>();

  if (!vio_accessor) {
    throw DynosamException(
        "VIOAccessor is null in formulation " + this->getFullyQualifiedName() +
        "."
        " If you have a formulation that derives from VIOFormulation you must "
        "construct an Accessor that inherits"
        " from VIOAccessor. When using AccessorT<MAP, DerivedAccessor> ensure "
        "Derived DerivedAccessor=VIOAccessor"
        " or if extending (DerivedAccessor) with another custom accessor, "
        "ensure the custom accessor derived from VIOAccessor");
  }
  return vio_accessor;
}

StateQuery<gtsam::NavState> VIOFormulation::getNavState(FrameId frame_id) {
  VIOAccessor::Ptr accessor = this->getAsVIOAccessor();
  return accessor->getNavState(frame_id);
};

gtsam::NavState VIOFormulation::predictAndAddFactorsVO(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& T_k_1_k) {
  CHECK(!isImuInitalized());

  const FrameId from_frame = last_propogate_frame_;
  const FrameId to_frame = frame_id_k;

  VIOAccessor::Ptr accessor = this->getAsVIOAccessor();
  const gtsam::NavState nav_state_prev =
      DYNO_GET_QUERY_DEBUG(accessor->getNavState(from_frame));

  VLOG(10) << "Forward predicting k=" << frame_id_k << " t=" << timestamp_k
           << " using VO";
  const gtsam::Pose3 X_W_km1 = nav_state_prev.pose();
  // apply relative pose
  const gtsam::Pose3 X_W_k = X_W_km1 * T_k_1_k;

  const double dt = timestamp_k - last_propogate_time_;
  CHECK_GT(dt, 0);
  // discrete derivative
  const gtsam::Vector3 V_C_k = T_k_1_k.translation() / dt;
  const gtsam::Vector3 V_W_k = X_W_k.rotation().rotate(V_C_k);
  const gtsam::NavState nav_state_k(X_W_k, V_W_k);

  if (params().use_vo) {
    auto odometry_noise = noiseModels().odometry_noise;
    CHECK(odometry_noise);
    CHECK_EQ(odometry_noise->dim(), 6u);

    this->addFactor(new_factors,
                    boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                        CameraPoseSymbol(from_frame),
                        CameraPoseSymbol(to_frame), T_k_1_k, odometry_noise));
    VLOG(30) << "Added Between factor frames " << from_frame << " -> "
             << to_frame << " using VO";
  }

  addSensorPose(new_values, to_frame, nav_state_k.pose());

  return nav_state_k;
}

gtsam::NavState VIOFormulation::predictAndAddFactorsIMU(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    FrameId frame_id_k, Timestamp timestamp_k, const ImuFrontend::PimPtr& pim) {
  CHECK(isImuInitalized());
  CHECK(pim);
  const FrameId from_frame = last_propogate_frame_;
  const FrameId to_frame = frame_id_k;

  VLOG(10) << "Forward predicting frame=" << frame_id_k << " using PIM";

  VIOAccessor::Ptr accessor = this->getAsVIOAccessor();
  const gtsam::NavState nav_state_prev =
      DYNO_GET_QUERY_DEBUG(accessor->getNavState(from_frame));
  const gtsam::imuBias::ConstantBias imu_bias_prev =
      DYNO_GET_QUERY_DEBUG(accessor->getImuBias(from_frame));

  const gtsam::NavState nav_state_k =
      pim->predict(nav_state_prev, imu_bias_prev);
  // add predicted camera value
  addSensorPose(new_values, frame_id_k, nav_state_k.pose());
  // add predicted velocity value
  this->addValue(new_values, nav_state_k.velocity(),
                 CameraVelocitySymbol(frame_id_k));
  // initalise imu bias
  this->addValue(new_values, imu_bias_prev, ImuBiasSymbol(frame_id_k));
  // add IMU factor
  const gtsam::PreintegratedCombinedMeasurements& pim_combined =
      dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(*pim);

  this->addFactor(
      new_factors,
      boost::make_shared<gtsam::CombinedImuFactor>(
          CameraPoseSymbol(from_frame), CameraVelocitySymbol(from_frame),
          CameraPoseSymbol(to_frame), CameraVelocitySymbol(to_frame),
          ImuBiasSymbol(from_frame), ImuBiasSymbol(to_frame), pim_combined));

  return nav_state_k;
}

void VIOFormulation::addImuStatesFromInitialNavState(
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const gtsam::Symbol velocity_key(CameraVelocitySymbol(first_frame_));
  const gtsam::Symbol imu_bias_key(ImuBiasSymbol(first_frame_));

  const gtsam::Point3& V_W_first = initial_nav_state_.velocity();
  const auto& initial_bias = initial_imu_bias_;

  this->addValue(new_values, V_W_first, velocity_key);
  // add bias state
  this->addValue(new_values, initial_bias, imu_bias_key);

  this->addFactor(new_factors,
                  boost::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
                      velocity_key, V_W_first, init_vel_prior_noise_));
  this->addFactor(
      new_factors,
      boost::make_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
          imu_bias_key, initial_bias, init_imu_bias_prior_noise_));
}

}  // namespace dyno
