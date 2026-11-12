#include "dynosam2/DynamicSlam.hpp"

#include <glog/logging.h>
#include <tbb/task_group.h>

#include "dynosam2/frontend/GeometryUpdater.hpp"
#include "dynosam_common/Transforms.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno2 {

/// @brief Only retirn the last two messages in the queue to publish the latest
/// data
static constexpr int kVizMaxQueueSize{2};

DynamicSlam::DynamicSlam(const PipelineParameters& parameters,
                         CanonicalSensorRig::Ptr sensor_rig)
    : parameters_(parameters),
      sensor_rig_(sensor_rig),
      camera_(sensor_rig->cannonicalCamera()),
      imu_params_(
          dyno::ImuFrontend::makeCombinedParams(sensor_rig->getImuParams())),
      frontend_(parameters.frontend, camera_),
      estimator_(camera_),
      frontend_viz_(estimator_),
      frontend_viz_data_queue_(kVizMaxQueueSize)

{
  // TODO: set some params for frontend!

  auto& vio_estimator = estimator_.vioEstimator();
  if (sensor_rig_->imuEnabled()) {
    vio_estimator.addImu(imu_params_);
  }

  if (parameters_.estimator.enforce_realtime) {
    vio_estimator.setOptimisationTimeLimit(
        parameters_.estimator.realtime_time_limit,
        parameters_.estimator.realtime_static_min_iterations);
  }

  // start viz thread
  frontend_viz_thread_ = std::thread(&DynamicSlam::visualisationLoop, this);
}

DynamicSlam::~DynamicSlam() {
  LOG(INFO) << "Shutting down DynamicSlam";
  shutdownThreads();
}

StateEstimate::Ptr DynamicSlam::spinOnce(
    dyno::VIFrontendInput::ConstPtr input) {
  LOG(INFO) << "DynamicSlam::spinOnce " << input->getFrameId();
  const auto t1 = dyno::utils::Timer::tic();
  dyno::utils::ChronoTimingStats spin_t("slam.spin");

  const dyno::FrameId frame_id = input->getFrameId();
  const dyno::Timestamp timestamp = input->getTimestamp();

  dyno::ImageContainer::Ptr image_container = input->image_container_;
  FeatureBlockContainer* features_ptr{nullptr};
  dyno::ObjectDetectionResult object_detection;

  // If no messages are provided just init with an empty vector
  // this is the equivalenent of having no measurements
  const dyno::ImuMeasurements imu_measurements =
      input->imu_measurements.value_or(dyno::ImuMeasurements{});

  if (first_frame_) {
    dyno::utils::ChronoTimingStats detect_t("slam.detect_and_track");
    frontend_.detectAndTrack(features_ptr, object_detection, image_container,
                             gtsam::Rot3::Identity());
    CHECK_NOTNULL(features_ptr);
    detect_t.stop();
    FeatureBlockContainer& features = *features_ptr;

    FeatureBlockGeometryMap geometries;
    GeometryUpdater(camera_, *image_container, features).calcPoints(geometries);

    for (const auto& [object_id, geometry] : geometries) {
      if (object_id > background_label) {
        DoFrame::Ptr frame =
            std::make_shared<DoFrame>(frame_id, timestamp, object_id);
        frame->fill(geometry, features);

        auto& do_estimator = estimator_.allocateDoEstimator(object_id);
        do_estimator.initaliseFrame(frame);

        if (parameters_.estimator.enforce_realtime) {
          do_estimator.setOptimisationTimeLimit(
              parameters_.estimator.realtime_time_limit,
              parameters_.estimator.realtime_dynamic_min_iterations);
        }

      } else {
        ViFrame::Ptr vi_frame = std::make_shared<ViFrame>(frame_id, timestamp);
        vi_frame->fill(geometry, features);

        // initial camera pose (opencv coordinate convention);
        // NOT for dataset data as we assume that we are already algined to the
        // world-frame with the ground truth
        gtsam::Pose3 T_WS;
        if (sensor_rig_->imuEnabled() &&
            parameters_.frontend.attempt_initial_pose_alignment) {
          // align with the world frame (z-down) using accelerometer
          // measurements
          bool success = initPoseImu(imu_measurements, T_WS);
          CHECK(success)
              << "pose could not be initalized from imu measurements.";
          LOG(INFO) << "Init from IMU measurement!";
          (void)(success);
        } else {
          T_WS = gtsam::Pose3::Identity();
        }

        VioGraphEstimator& vio_estimator = estimator_.vioEstimator();
        vio_estimator.initaliseStates(vi_frame, T_WS);
        // first frame must be keyframe
        // measurements SHOULD get added in the solveVisualOdometry step
        // after outlier rejection
        vio_estimator.setKeyframe(frame_id, true);
      }
    }

    first_frame_ = false;

    PublicationData::Ptr publication_data = std::make_shared<PublicationData>();
    publication_data->frame_id = frame_id;
    publication_data->timestamp = timestamp;
    publication_data->updated_states.vo.push_back(frame_id);
    pushPublicationDataToQueue(publication_data);

    return nullptr;
  }

  VioGraphEstimator& vio_estimator = estimator_.vioEstimator();
  // propogate the next pose from the most recent state using IMU or constant
  // velocity
  const VioMotionPropagation vio_propagation = Frontend::predictFromMotion(
      frame_id, timestamp, vio_estimator, imu_measurements);
  if (!vio_propagation.success) {
    LOG(FATAL) << "No IMU data and could not "
                  "successfully progate motion!";
  }
  VLOG(30) << "Done motion propogation with source: " << vio_propagation.source;
  // forward rotation prediction using either constant motion or IMU
  // rottes from k-1 to k
  const gtsam::Rot3 R_k_km1 =
      vio_propagation.relative_motion_error->deltaXij().rotation();

  dyno::utils::ChronoTimingStats detect_t("slam.detect_and_track");
  frontend_.detectAndTrack(features_ptr, object_detection, image_container,
                           R_k_km1);
  CHECK_NOTNULL(features_ptr);
  detect_t.stop();
  FeatureBlockContainer& features = *features_ptr;

  FeatureBlockGeometryMap geometries;
  GeometryUpdater(camera_, *image_container, features).calcPoints(geometries);

  struct ObjectContext {
    FeatureBlockGeometry* geometry;
    DoGraphEstimator* estimator;
  };

  std::vector<ObjectContext> object_contexts;
  // only include object ids here
  std::vector<ObjectId> object_ids;

  FeatureBlockGeometry* vo_geometry{nullptr};
  // make sure we have solvers for detected objects
  // what if we have not solved for the geometries (ie VO with no features but
  // we have IMU?)
  for (auto& [object_id, geometry] : geometries) {
    if (object_id > background_label &&
        parameters_.debug.estimate_dynamic_objects) {
      if (!estimator_.hasObjectEstimator(object_id)) {
        auto& do_estimator = estimator_.allocateDoEstimator(object_id);
        if (parameters_.estimator.enforce_realtime) {
          do_estimator.setOptimisationTimeLimit(
              parameters_.estimator.realtime_time_limit,
              parameters_.estimator.realtime_dynamic_min_iterations);
        }
      }
      ObjectContext context;
      context.geometry = CHECK_NOTNULL(&geometry);
      context.estimator = CHECK_NOTNULL(&estimator_.doEstimator(object_id));
      object_contexts.push_back(context);
      object_ids.push_back(object_id);

    } else if (object_id == background_label) {
      vo_geometry = &geometry;
    }
  }

  PublicationData::Ptr publication_data =
      std::make_shared<PublicationData>(frame_id, timestamp);
  std::vector<dyno::FrameId>& updated_vo_states =
      publication_data->updated_states.vo;

  // setup context to write visual odometry into this alue
  gtsam::Pose3 X_W_k;
  AyncVO async_context;
  async_context.vo = &X_W_k;
  CHECK_NOTNULL(async_context.vo);

  bool run_vo{true};
  bool vo_success{false};

  // handle the case we have no VO geometry but we have IMU
  if (!vo_geometry) {
    ViFrame::Ptr vi_frame = std::make_shared<ViFrame>(frame_id, timestamp);
    vi_frame->setRelativePose(vio_propagation.relative_motion_error);

    // TODO: in this failure case its probbaly better we dont add it as a state
    //  but instead init it as a frame
    //  1. This means that the logic in predictFromMotion which checks the last
    //  valid state
    //     will work because the last valid state was one we had correspondances
    //     for
    //  2. The logic is more sound beciase we should only have states poses we
    //  have tracks for I guess we could get around this by changing it to like
    //  "last optimized state" becuase we check for covisability but this
    //  becomes rahter convoluted....
    // TODO: NO! becuase we need all states for example so the IMU is always
    // between the last image and the current image
    //  I guess we could merge IMU error terms if we dont have a valid solve so
    //  the last state is still the last state we had correspondences in! new
    //  points should get created then as there will be 0% overlap with the last
    //  keyframe! but basically if we dont add states for each frame we then
    //  need to accumulate IMU error terms becuase ImuMeasurements should
    //  correspond to k-1 to k Todo this we would then also need to propogate
    //  using the RelativePoseError term which we could do, arguably.. in
    //  practive we probably dont need to bother for the PhD and for the
    //  datasets we will test.
    vio_estimator.initaliseStates(vi_frame, vio_propagation.predicted_pv);
    run_vo = false;
    vo_success = true;
    // still try and solve for objects, just immedidately by pass the waiting on
    // the mutex since we have odometry by IMU propagation!
    async_context.vo_ready = true;
    async_context.vo_valid = true;
  }

  const size_t num_objects = object_ids.size();
  std::vector<uchar> do_solve_success(num_objects, 0);
  std::vector<std::vector<FrameId>> updated_object_frames(num_objects);
  std::vector<std::vector<TrackletId>> object_correspondences(num_objects);

  // vector of keypoints which had measurements added to the bundle adjustment
  std::vector<KeypointIdentifier> vo_measurements_added;
  std::vector<TrackletId> vo_correspondences;
  dyno::utils::ChronoTimingStats solve_t("slam.solve");
  tbb::task_group group;
  if (run_vo) {
    group.run([&] {
      dyno::utils::ChronoTimingStats vo_t("slam.vo");
      VioGraphEstimator& vio_estimator = estimator_.vioEstimator();
      frontend_.solveVisualOdometry(
          vio_estimator, features, *vo_geometry, async_context,
          updated_vo_states, vo_measurements_added, vo_correspondences,
          vo_success, *image_container, vio_propagation);
    });
  }

  dyno::utils::ChronoTimingStats doo_t("slam.doo");
  for (size_t i = 0; i < num_objects; i++) {
    group.run([&, i] {
      ObjectId object_id = object_ids[i];
      ObjectContext object_context = object_contexts[i];
      uchar& solve_success = do_solve_success.at(i);

      std::vector<FrameId>& updated_frames = updated_object_frames.at(i);
      std::vector<TrackletId>& correspondences = object_correspondences.at(i);
      frontend_.solveObjectOdometry(
          object_id, vio_estimator, parameters_.estimator, *image_container,
          *object_context.estimator, features, *object_context.geometry,
          async_context, updated_frames, correspondences, solve_success);
    });
  }
  group.wait();
  solve_t.stop();
  doo_t.stop();

  std::vector<dyno::TemporalObjectId>& objects_with_update =
      publication_data->updated_states.objects;

  // tracking state information
  dyno::ViTrackingState& vi_tracking_state =
      publication_data->vi_tracking_state;
  vi_tracking_state.frame_id = frame_id;
  // TODO: quality etc

  std::vector<dyno::DoTrackingState>& do_tracking_states =
      publication_data->do_tracking_states;

  if (!vo_success) {
    LOG(FATAL) << "Could not produce a VO!";
    // VioGraphEstimator& estimator = estimator_.vioEstimator();
    // vi_tracking_state.is_keyframe = estimator.isKeyframe(frame_id);
    // // prune keyframes
    // estimator.pruneKeyframes(5, 3);
  }

  VioGraphEstimator& estimator = estimator_.vioEstimator();
  vi_tracking_state.is_keyframe = estimator.isKeyframe(frame_id);
  // prune keyframes
  estimator.pruneKeyframes(parameters_.estimator.num_static_keyframes, 3);

  if (vi_tracking_state.is_keyframe && parameters_.debug.use_fullgraph) {
    estimator_.addStaticMeasurements(vo_measurements_added);
  }

  FrontendViz::Data::Ptr visualization_data =
      std::make_shared<FrontendViz::Data>();
  visualization_data->images = image_container;
  visualization_data->object_detection = object_detection;
  visualization_data->vi_tracks.frame = estimator.getFramePtr(frame_id);
  visualization_data->vi_tracks.correspondences = vo_correspondences;
  visualization_data->vi_tracks.measurements = vo_measurements_added;

  for (size_t i = 0; i < num_objects; i++) {
    const ObjectId object_id = object_ids[i];
    LOG(INFO) << "j= " << object_id
              << " solve success= " << static_cast<int>(do_solve_success[i]);
    if (!do_solve_success[i]) {
      continue;
    }

    const ObjectContext object_context = object_contexts[i];
    const DoGraphEstimator& estimator = *object_context.estimator;

    const auto& updated_frames = updated_object_frames.at(i);
    for (FrameId k0 : updated_frames) {
      objects_with_update.emplace_back(k0, object_id);
    }

    do_tracking_states.emplace_back();
    do_tracking_states.back().frame_id = frame_id;
    do_tracking_states.back().object_id = object_id;
    do_tracking_states.back().keyframe_type =
        estimator.keyframeStatus(frame_id);

    auto& do_track_viz = visualization_data->do_tracks[object_id];
    do_track_viz.frame = estimator.getFramePtr(frame_id);
    do_track_viz.correspondences = object_correspondences.at(i);
  }

  if (estimator_.needsFullGraphOptimisation()) {
    LOG(INFO) << "Starting full graph optimisation";
    estimator_.startFullGraphOptimisation();
  }

  if (estimator_.updateAvaialbleFromFullGraph()) {
    LOG(INFO) << "Update availanble from full graph!";
    PublicationData::Ptr publication_data =
        std::make_shared<PublicationData>(frame_id, timestamp);
    estimator_.synchroniseRealtimeAndFullGraph(
        publication_data->updated_states);
    pushPublicationDataToQueue(publication_data);
  }

  const auto t2 = dyno::utils::Timer::tic();
  // const double dt_seconds = dyno::utils::Timer::

  frontend_viz_data_queue_.push(visualization_data);

  features.reduceToInliersInplace();

  // pushImageToDisplayQueue("Tracks", features.draw(rgb, true));
  pushPublicationDataToQueue(publication_data);

  return nullptr;
}

bool DynamicSlam::initPoseImu(const dyno::ImuMeasurements& imu_measurements,
                              gtsam::Pose3& T_WC) const {
  const size_t size = imu_measurements.size();

  if (size == 0) {
    return false;
  }

  const dyno::ImuAccGyrs& acc_gyrs = imu_measurements.acc_gyr_;
  // acceleration vector
  Eigen::Vector3d acc_I = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < size; i++) {
    acc_I += acc_gyrs.col(i).head<3>();
  }

  acc_I /= double(size);
  Eigen::Vector3d e_acc = acc_I.normalized();
  LOG(INFO) << e_acc;

  // align to OpenCV coordinate convention (ie z down!)
  const Eigen::Vector3d ez_W(0.0, -1.0, 0.0);

  gtsam::Pose3 T_WI = gtsam::Pose3::Identity();
  Eigen::Matrix<double, 6, 1> poseIncrement;
  poseIncrement.tail<3>() = Eigen::Vector3d::Zero();
  poseIncrement.head<3>() = ez_W.cross(e_acc).normalized();
  double angle = std::acos(ez_W.transpose() * e_acc);
  poseIncrement.head<3>() *= angle;

  T_WI = T_WI.expmap(-poseIncrement);

  // This composes the aligned IMU pose with the camera-to-IMU transform.
  const gtsam::Pose3& T_CI = sensor_rig_->getImuParams().T_CI;

  T_WC = T_WI * T_CI.inverse();
  return true;
}

void DynamicSlam::shutdownThreads() {
  is_shutdown_ = true;

  // shutdown queues first
  frontend_viz_data_queue_.shutdown();

  // TODO: force asynchronous backend to also shutdown!

  if (frontend_viz_thread_.joinable()) {
    frontend_viz_thread_.join();
  }
}

void DynamicSlam::visualisationLoop() {
  while (!is_shutdown_) {
    FrontendViz::Data::Ptr viz_data;
    if (frontend_viz_data_queue_.popBlocking(viz_data)) {
      cv::Mat viz = frontend_viz_.draw(viz_data);
      pushImageToDisplayQueue("Tracks", viz);
    }
  }
}

}  // namespace dyno2
