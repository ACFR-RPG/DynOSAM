#include "dynosam/formulations/VIOFormulation.hpp"

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

// struct VIOUpdater {
//  public:
//   VIOUpdater(VIOFormulation* vio_formulation)
//       : vio_formulation_(CHECK_NOTNULL(vio_formulation)) {}
//   virtual ~VIOUpdater() = default;

//  protected:
//   // some helper base functions
//   bool isRobust() const {
//     return vio_formulation_->params().makeStaticMeasurementsRobust();
//   }

//   // NOTE: pass pointer by reference as we want to change the object the
//   // noise_model points too
//   //  since robustifyHuber returns a new model
//   void robustifyHuber(gtsam::SharedNoiseModel& noise_model) {
//     if (isRobust()) {
//       noise_model = factor_graph_tools::robustifyHuber(
//           vio_formulation_->params().k_huber_3d_points_, noise_model);
//     }
//   }

//   bool isPointAdded(gtsam::Key point_key) {
//     return vio_formulation_->is_other_values_in_map.exists(point_key);
//   }

//   void markPointAsAdded(gtsam::Key point_key) {
//     vio_formulation_->is_other_values_in_map.insert2(point_key, true);
//   }

//  protected:
//   VIOFormulation* vio_formulation_;
// };

// class VIOUpdaterImpl : public VIOUpdater {
//  public:
//   using MapTraits = VIOFormulation::MapTraitsType;
//   using LmkNode = MapTraits::SharedLandmarkNode;
//   using FrameNode = MapTraits::SharedFrameNode;
//   using MeasurementType = MapTraits::MeasurementType;
//   using MeasurementTraits = measurement_traits<MeasurementType>;

//   VIOUpdaterImpl(VIOFormulation* vio_formulation)
//       : VIOUpdater(vio_formulation) {}

//   /**
//    * @brief
//    *
//    * Return results indicates if any factors/values were added (ie. the point
//    * was added).
//    *
//    * @param lmk Landmark is passed by reference as some internal flags may be
//    * changed
//    * @param frame
//    * @param update_params
//    * @param values
//    * @param graph
//    * @param point_key
//    * @param result
//    * @param initial
//    * @return true
//    * @return false
//    */
//   virtual bool addLandmark(LmkNode& lmk, const FrameNode& frame,
//                            const UpdateObservationParams& update_params,
//                            gtsam::Values& values,
//                            gtsam::NonlinearFactorGraph& graph,
//                            gtsam::Key& point_key,
//                            UpdateObservationResult& result,
//                            std::optional<Landmark>& initial) = 0;
// };

// class PTPUpdater : public VIOUpdaterImpl {
//  public:
//   PTPUpdater(VIOFormulation* vio_formulation)
//       : VIOUpdaterImpl(vio_formulation) {}

//   bool addLandmark(LmkNode& lmk, const FrameNode& frame,
//                    const UpdateObservationParams& update_params,
//                    gtsam::Values& values, gtsam::NonlinearFactorGraph& graph,
//                    gtsam::Key& point_key, UpdateObservationResult& result,
//                    std::optional<Landmark>& initial) override {
//     point_key = lmk->makeStaticKey();
//     const FrameId frame_k = FrameId(frame->getId());

//     const auto& params = this->vio_formulation_->params();

//     if (this->isPointAdded(point_key)) {
//       CHECK(lmk->added_to_opt);
//       const auto pose_key = frame->makePoseKey();

//       auto [measured_point_local, measurement_covariance] =
//           MeasurementTraits::pointWithCovariance(lmk->getMeasurement(frame_k));
//       CHECK_NOTNULL(measurement_covariance);

//       this->robustifyHuber(measurement_covariance);

//       auto factor =
//           boost::make_shared<gtsam::PoseToPointFactor<gtsam::Pose3,
//           Landmark>>(
//               pose_key, point_key, measured_point_local,
//               measurement_covariance);
//       graph.add(factor);
//       result.updateAffectedObject(frame_k, 0);
//       return true;
//     } else {
//       CHECK(!lmk->added_to_opt);

//       if (lmk->numObservations() < params.min_static_observations) {
//         return false;
//       }

//       // this condition should only run once per tracklet (ie.e the first
//       time
//       // the tracklet has enough observations) we gather the tracklet
//       // observations and then initalise it in the new values these should
//       // then get added to the map and map_->exists() should return true for
//       // all other times
//       const auto& seen_frames = lmk->getSeenFrames();
//       for (const auto& seen_frame : seen_frames) {
//         FrameId seen_frame_id = FrameId(seen_frame->getId());
//         // only iterate up to the query frame
//         if (seen_frame_id > frame_k) {
//           break;
//         }

//         // if we should not backtrack, only add the current frame!!!
//         const auto do_backtrack = update_params.do_backtrack;
//         if (!do_backtrack && seen_frame_id < frame_k) {
//           continue;
//         }

//         const gtsam::Key pose_key = seen_frame->makePoseKey();
//         auto [measured_point_local, measurement_covariance] =
//             MeasurementTraits::pointWithCovariance(
//                 lmk->getMeasurement(seen_frame_id));
//         CHECK_NOTNULL(measurement_covariance);

//         this->robustifyHuber(measurement_covariance);

//         auto factor = boost::make_shared<
//             gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//             pose_key, point_key, measured_point_local,
//             measurement_covariance);

//         graph.add(factor);
//         result.updateAffectedObject(seen_frame_id, 0);
//       }

//       const Landmark& measured =
//           MeasurementTraits::point(lmk->getMeasurement(frame_k));

//       // TODO: should use getInitialOrLinearizedSensorPose
//       gtsam::Pose3 T_W_X;
//       CHECK(
//           this->vio_formulation_->map()->hasInitialSensorPose(frame_k,
//           &T_W_X));

//       Landmark initial_point = T_W_X * measured;
//       initial = initial_point;
//       result.updateAffectedObject(frame_k, 0);

//       values.insert(point_key, initial_point);
//       this->markPointAsAdded(point_key);
//       lmk->added_to_opt = true;
//       return true;
//     }
//   }
// };

// class GenericProjectionUpdater : public VIOUpdaterImpl {
//  public:
//   GenericProjectionUpdater(VIOFormulation* vio_formulation)
//       : VIOUpdaterImpl(vio_formulation) {}

//   bool addLandmark(LmkNode& lmk, const FrameNode& frame,
//                    const UpdateObservationParams& update_params,
//                    gtsam::Values& values, gtsam::NonlinearFactorGraph& graph,
//                    gtsam::Key& point_key, UpdateObservationResult& result,
//                    std::optional<Landmark>& initial) override {
//     LOG(FATAL) << "Not implemented";
//   }
// };

// class StereoProjectionUpdater : public VIOUpdaterImpl {
//  public:
//   StereoProjectionUpdater(VIOFormulation* vio_formulation)
//       : VIOUpdaterImpl(vio_formulation) {
//     std::shared_ptr<Camera> camera =
//         CHECK_NOTNULL(vio_formulation_->sensors().camera);
//     std::shared_ptr<RGBDCamera> rgbd_camera =
//         CHECK_NOTNULL(camera->safeGetRGBDCamera());
//     K_stereo_ = rgbd_camera->getFakeStereoCalib();
//     CHECK_NOTNULL(K_stereo_);
//     K_ = camera->getGtsamCalibration();
//   }

//   bool addLandmark(LmkNode& lmk, const FrameNode& frame,
//                    const UpdateObservationParams& update_params,
//                    gtsam::Values& values, gtsam::NonlinearFactorGraph& graph,
//                    gtsam::Key& point_key, UpdateObservationResult& result,
//                    std::optional<Landmark>& initial) override {
//     point_key = lmk->makeStaticKey();
//     const FrameId frame_k = FrameId(frame->getId());

//     // NOTE: not handling the case a lmk becomes an outlier once already
//     added
//     // to the opt
//     //  hoping the robust cost funcion handles this!
//     if (this->isPointAdded(point_key)) {
//       CHECK(lmk->added_to_opt);
//       const auto pose_key = frame->makePoseKey();

//       auto stereo_measurement =
//           MeasurementTraits::stereo(lmk->getMeasurement(frame_k));
//       // FOR NOW
//       CHECK(stereo_measurement);
//       auto [measurement, model] = *stereo_measurement;

//       this->robustifyHuber(model);

//       auto factor = boost::make_shared<GenericStereoFactor>(
//           measurement, model, pose_key, point_key, K_stereo_);

//       graph.add(factor);
//       result.updateAffectedObject(frame_k, 0);
//       return true;
//     } else {
//       CHECK(!lmk->added_to_opt);

//       using GtsamCamera = Camera::CameraImpl;
//       CameraSet<GtsamCamera> camera_set;
//       gtsam::Point2Vector measurements;
//       gtsam::SharedNoiseModel model;

//       // triangulate stereo measurements by treating each stereocamera as a
//       // pair of monocular cameras this is vital for good triangulation!
//       const auto& seen_frames = lmk->getSeenFrames();
//       for (const auto& frame_node_i : seen_frames) {
//         FrameId frame_id_i = frame_node_i->getId();
//         // use the initial pose
//         // in the IMU case the optimised pose will be not so good until
//         visual
//         // odom starts working... or maybe not...
//         gtsam::Pose3 X_W_i;
//         CHECK(this->vio_formulation_->map()->hasInitialSensorPose(frame_id_i,
//                                                                   &X_W_i))
//             << "Missing initial pose at k=" << frame_id_i;
//         // TODO: hack for now - in the KF case, we sometimes need to add
//         // KF in the past so we already have measurements at k+1 but not yet
//         //  an initial pose measurement as the frontend has not send it
//         //  for now just skip!
//         //  if(!this->vio_formulation_->map()->hasInitialSensorPose(
//         //    frame_id_i, &X_W_i)) { continue; }

//         const gtsam::Pose3 leftPose = X_W_i;
//         const gtsam::Cal3_S2 monoCal = K_stereo_->calibration();
//         const GtsamCamera leftCamera_i(leftPose, monoCal);
//         const gtsam::Pose3 left_Pose_right = gtsam::Pose3(
//             gtsam::Rot3(), gtsam::Point3(K_stereo_->baseline(), 0.0, 0.0));
//         const gtsam::Pose3 rightPose = leftPose.compose(left_Pose_right);
//         const GtsamCamera rightCamera_i(rightPose, monoCal);

//         // gtsam::Pose3 X_W_i =
//         //
//         this->vio_formulation_->getInitialOrLinearizedSensorPose(frame_id_i);

//         // updates the model each time, just uses the last one!
//         // auto [keypoint, model] =
//         MeasurementTraits::keypointWithCovariance(
//         //     lmk->getMeasurement(frame_id_i));
//         auto stereo_measurement =
//             MeasurementTraits::stereo(lmk->getMeasurement(frame_id_i));
//         CHECK(stereo_measurement);
//         auto [zi, model] = *stereo_measurement;

//         camera_set.push_back(leftCamera_i);
//         measurements.push_back(Point2(zi.uL(), zi.v()));
//         if (!std::isnan(zi.uR())) {  // if right point is valid
//           camera_set.push_back(rightCamera_i);
//           measurements.push_back(Point2(zi.uR(), zi.v()));
//         }
//       }

//       gtsam::TriangulationParameters triangulation_params;
//       // triangulation_params.useLOST = true;
//       triangulation_params.noiseModel = model;
//       auto triangulation_result = gtsam::triangulateSafe<GtsamCamera>(
//           camera_set, measurements, triangulation_params);

//       if (triangulation_result) {
//         const gtsam::Point3 initial_point = *triangulation_result;
//         initial = initial_point;

//         double reprojection_error =
//             camera_set.reprojectionError(initial_point, measurements).norm();

//         // if error is too large, discard point
//         if (reprojection_error > 3.0) {
//           // mark as outlier for the front-end
//           lmk->inlier = false;
//           return false;
//         }
//         // collect factors
//         gtsam::NonlinearFactorGraph stereo_factors;
//         FrameIds frames_with_good_factors;
//         for (const auto& frame_node_i : seen_frames) {
//           const auto pose_key_i = frame_node_i->makePoseKey();
//           FrameId frame_id_i = frame_node_i->getId();

//           auto stereo_measurement =
//               MeasurementTraits::stereo(lmk->getMeasurement(frame_id_i));
//           // FOR NOW
//           CHECK(stereo_measurement);
//           auto [measurement, model] = *stereo_measurement;

//           double disparity = measurement.uL() - measurement.uR();
//           if (disparity > 0.5) {
//             this->robustifyHuber(model);
//             auto factor = boost::make_shared<GenericStereoFactor>(
//                 measurement, model, pose_key_i, point_key, K_stereo_);
//             stereo_factors += factor;
//             frames_with_good_factors.push_back(frame_id_i);
//           }
//         }

//         if (stereo_factors.size() < 2) {
//           return false;
//         }

//         for (const auto good_frame_id : frames_with_good_factors) {
//           result.updateAffectedObject(good_frame_id, 0);
//         }
//         graph += stereo_factors;

//         values.insert(point_key, initial_point);
//         this->markPointAsAdded(point_key);
//         lmk->added_to_opt = true;

//         return true;
//       } else {
//         // mark as outlier for the front-end
//         lmk->inlier = false;
//         return false;
//       }
//     }
//   }

//  private:
//   Camera::CalibrationType::shared_ptr K_;
//   StereoCalibPtr K_stereo_;
// };

}  // namespace dyno
