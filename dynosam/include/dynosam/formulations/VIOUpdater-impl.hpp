#pragma once

#include <gtsam/geometry/triangulation.h>

#include "dynosam/formulations/VIOFormulation.hpp"

namespace dyno {

/// triangulateSafe: extensive checking of the outcome
/// a modifed version that lets use use triangulateLOST
/// while still on gtsam 4.2
template <class CAMERA>
gtsam::TriangulationResult triangulateSafe(
    const gtsam::CameraSet<CAMERA>& cameras,
    const typename CAMERA::MeasurementVector& measured,
    const gtsam::TriangulationParameters& params, const bool useLOST = false);

template <typename MAP>
struct VIOUpdater {
 public:
  typedef VIOFormulation<MAP> VIOFormulationM;

  VIOUpdater(VIOFormulationM* vio_formulation)
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
  VIOFormulationM* vio_formulation_;
};

template <typename MAP>
class VIOUpdaterImpl : public VIOUpdater<MAP> {
 public:
  typedef typename VIOUpdater<MAP>::VIOFormulationM VIOFormulation;
  using MapTraits = typename VIOFormulation::MapTraitsType;
  using LmkNode = typename MapTraits::SharedLandmarkNode;
  using FrameNode = typename MapTraits::SharedFrameNode;
  using MeasurementType = typename MapTraits::MeasurementType;
  using MeasurementTraits = measurement_traits<MeasurementType>;

  VIOUpdaterImpl(VIOFormulation* vio_formulation)
      : VIOUpdater<MAP>(vio_formulation) {}

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
  virtual bool addLandmark(LmkNode&, const FrameNode&,
                           const UpdateObservationParams&, gtsam::Values&,
                           gtsam::NonlinearFactorGraph&, gtsam::Key&,
                           UpdateObservationResult&, std::optional<Landmark>&) {
    throw DynosamException(
        "VIOUpdaterImpl::addLandmark not implemented in derived class");
  }
};

template <typename MAP>
class PTPUpdater : public VIOUpdaterImpl<MAP> {
 public:
  typedef VIOUpdaterImpl<MAP> Base;
  typedef typename Base::VIOFormulation VIOFormulation;
  typedef typename Base::MapTraits MapTraits;
  typedef typename Base::LmkNode LmkNode;
  typedef typename Base::FrameNode FrameNode;
  typedef typename Base::MeasurementTraits MeasurementTraits;

  PTPUpdater(VIOFormulation* vio_formulation)
      : VIOUpdaterImpl<MAP>(vio_formulation) {}

  bool addLandmark(LmkNode& lmk, const FrameNode& frame,
                   const UpdateObservationParams& update_params,
                   gtsam::Values& values, gtsam::NonlinearFactorGraph& graph,
                   gtsam::Key& point_key, UpdateObservationResult& result,
                   std::optional<Landmark>& initial) override {
    point_key = lmk->makeStaticKey();
    const FrameId frame_k = FrameId(frame->getId());

    const auto& params = this->vio_formulation_->params();

    if (this->isPointAdded(point_key)) {
      // CHECK(lmk->added_to_opt);
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

      // initalise using a single measurement at k
      gtsam::Pose3 X_W_k = frame->initialSensorPose();
      Landmark initial_point = X_W_k * measured;

      initial = initial_point;
      result.updateAffectedObject(frame_k, 0);

      values.insert(point_key, initial_point);
      this->markPointAsAdded(point_key);
      return true;
    }
  }
};

template <typename MAP>
class GenericProjectionUpdater : public VIOUpdaterImpl<MAP> {
 public:
  typedef VIOUpdaterImpl<MAP> Base;
  typedef typename Base::VIOFormulation VIOFormulation;
  typedef typename Base::MapTraits MapTraits;
  typedef typename Base::LmkNode LmkNode;
  typedef typename Base::FrameNode FrameNode;

  GenericProjectionUpdater(VIOFormulation* vio_formulation)
      : VIOUpdaterImpl<MAP>(vio_formulation) {}

  bool addLandmark(LmkNode&, const FrameNode&, const UpdateObservationParams&,
                   gtsam::Values&, gtsam::NonlinearFactorGraph&, gtsam::Key&,
                   UpdateObservationResult&,
                   std::optional<Landmark>&) override {
    LOG(FATAL) << "Not implemented";
  }
};

template <typename MAP>
class StereoProjectionUpdater : public VIOUpdaterImpl<MAP> {
 public:
  typedef VIOUpdaterImpl<MAP> Base;
  typedef typename Base::VIOFormulation VIOFormulation;
  typedef typename Base::MapTraits MapTraits;
  typedef typename Base::LmkNode LmkNode;
  typedef typename Base::FrameNode FrameNode;
  typedef typename Base::MeasurementTraits MeasurementTraits;

  StereoProjectionUpdater(VIOFormulation* vio_formulation)
      : VIOUpdaterImpl<MAP>(vio_formulation) {
    std::shared_ptr<Camera> camera =
        CHECK_NOTNULL(this->vio_formulation_->sensors().camera);
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
      // CHECK(lmk->added_to_opt);
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
      // CHECK(!lmk->added_to_opt);

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
        // Pose3Measurement X_W_i = frame_node_i->initialSensorPose();
        gtsam::Pose3 X_W_i =
            this->vio_formulation_->getInitialOrLinearizedSensorPose(
                frame_id_i);

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
      // triangulate with lost
      auto triangulation_result = dyno::triangulateSafe<GtsamCamera>(
          camera_set, measurements, triangulation_params, true);

      if (triangulation_result) {
        const gtsam::Point3 initial_point = *triangulation_result;
        initial = initial_point;

        double reprojection_error =
            camera_set.reprojectionError(initial_point, measurements).norm();

        // if error is too large, discard point
        if (reprojection_error > 3.0) {
          // mark as outlier for the front-end
          // lmk->inlier = false;
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
        // lmk->added_to_opt = true;

        return true;
      } else {
        // mark as outlier for the front-end
        // lmk->inlier = false;
        return false;
      }
    }
  }

 private:
  Camera::CalibrationType::shared_ptr K_;
  StereoCalibPtr K_stereo_;
};

/// @brief A modified of gtsam::triangulateSafe that allows us to parse useLOST
/// as this is not available on the 4.2 version of gtsam currently used.
/// @tparam CAMERA
/// @param cameras
/// @param measured
/// @param params
/// @param useLOST
/// @return
template <class CAMERA>
gtsam::TriangulationResult triangulateSafe(
    const gtsam::CameraSet<CAMERA>& cameras,
    const typename CAMERA::MeasurementVector& measured,
    const gtsam::TriangulationParameters& params, const bool useLOST) {
  size_t m = cameras.size();

  // if we have a single pose the corresponding factor is uninformative
  if (m < 2)
    return gtsam::TriangulationResult::Degenerate();
  else
    // We triangulate the 3D position of the landmark
    try {
      gtsam::Point3 point = gtsam::triangulatePoint3<CAMERA>(
          cameras, measured, params.rankTolerance, params.enableEPI,
          params.noiseModel, useLOST);

      // Check landmark distance and re-projection errors to avoid outliers
      size_t i = 0;
      double maxReprojError = 0.0;
      for (const CAMERA& camera : cameras) {
        const gtsam::Pose3& pose = camera.pose();
        if (params.landmarkDistanceThreshold > 0 &&
            distance3(pose.translation(), point) >
                params.landmarkDistanceThreshold)
          return gtsam::TriangulationResult::FarPoint();
#ifdef GTSAM_THROW_CHEIRALITY_EXCEPTION
        // verify that the triangulated point lies in front of all cameras
        // Only needed if this was not yet handled by exception
        const gtsam::Point3& p_local = pose.transformTo(point);
        if (p_local.z() <= 0) return gtsam::TriangulationResult::BehindCamera();
#endif
        // Check reprojection error
        if (params.dynamicOutlierRejectionThreshold > 0) {
          const typename CAMERA::Measurement& zi = measured.at(i);
          gtsam::Point2 reprojectionError = camera.reprojectionError(point, zi);
          maxReprojError = std::max(maxReprojError, reprojectionError.norm());
        }
        i += 1;
      }
      // Flag as degenerate if average reprojection error is too large
      if (params.dynamicOutlierRejectionThreshold > 0 &&
          maxReprojError > params.dynamicOutlierRejectionThreshold)
        return gtsam::TriangulationResult::Outlier();

      // all good!
      return gtsam::TriangulationResult(point);
    } catch (gtsam::TriangulationUnderconstrainedException&) {
      // This exception is thrown if
      // 1) There is a single pose for triangulation - this should not happen
      // because we checked the number of poses before 2) The rank of the matrix
      // used for triangulation is < 3: rotation-only, parallel cameras (or
      // motion towards the landmark)
      return gtsam::TriangulationResult::Degenerate();
    } catch (TriangulationCheiralityException&) {
      // point is behind one of the cameras: can be the case of
      // close-to-parallel cameras or may depend on outliers
      return gtsam::TriangulationResult::BehindCamera();
    }
}

}  // namespace dyno
