#include "dynosam2/DynamicSlam.hpp"

#include <glog/logging.h>
#include <tbb/task_group.h>

#include "dynosam2/frontend/GeometryUpdater.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno2 {

DynamicSlam::DynamicSlam(const PipelineParameters& parameters,
                         CanonicalSensorRig::Ptr sensor_rig)
    : parameters_(parameters),
      sensor_rig_(sensor_rig),
      camera_(sensor_rig->cannonicalCamera()),
      frontend_(parameters.frontend, camera_),
      estimator_(camera_)

{
  // TODO: set some params for frontend!
}

DynamicSlam::~DynamicSlam() {}

StateEstimate::Ptr DynamicSlam::spinOnce(
    dyno::VIFrontendInput::ConstPtr input) {
  LOG(INFO) << "DynamicSlam::spinOnce " << input->getFrameId();
  dyno::utils::ChronoTimingStats spin_t("slam.spin");

  const dyno::FrameId frame_id = input->getFrameId();
  const dyno::Timestamp timestamp = input->getTimestamp();

  dyno::ImageContainer::Ptr image_container = input->image_container_;
  FeatureBlockContainer* features_ptr{nullptr};
  dyno::ObjectDetectionResult object_detection;

  dyno::utils::ChronoTimingStats detect_t("slam.detect_and_track");
  frontend_.detectAndTrack(features_ptr, object_detection, image_container);
  CHECK_NOTNULL(features_ptr);
  detect_t.stop();
  FeatureBlockContainer& features = *features_ptr;

  FeatureBlockGeometryMap geometries;
  GeometryUpdater(camera_, *image_container, features).calcPoints(geometries);

  if (first_frame_) {
    for (const auto& [object_id, geometry] : geometries) {
      if (object_id > background_label) {
        DoFrame::Ptr frame =
            std::make_shared<DoFrame>(frame_id, timestamp, object_id);
        frame->fill(geometry, features);
        estimator_.allocateDoEstimator(object_id).initaliseFrame(frame);
      } else {
        ViFrame::Ptr vi_frame = std::make_shared<ViFrame>(frame_id, timestamp);
        vi_frame->fill(geometry, features);

        VioGraphEstimator& vio_estimator = estimator_.vioEstimator();
        // TODO: add measurements!
        vio_estimator.initaliseStates(vi_frame, gtsam::Pose3::Identity());
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
    if (object_id > background_label) {
      if (!estimator_.hasObjectEstimator(object_id)) {
        estimator_.allocateDoEstimator(object_id);
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

  CHECK(vo_geometry);

  bool vo_success{false};

  // setup context to write visual odometry into this alue
  gtsam::Pose3 X_W_k;
  AyncVO async_context;
  async_context.vo = &X_W_k;
  CHECK_NOTNULL(async_context.vo);

  const size_t num_objects = object_ids.size();
  std::vector<uchar> do_solve_success(num_objects, 0);

  std::vector<std::vector<FrameId>> updated_object_frames(num_objects);

  PublicationData::Ptr publication_data =
      std::make_shared<PublicationData>(frame_id, timestamp);
  std::vector<dyno::FrameId>& updated_vo_states =
      publication_data->updated_states.vo;

  dyno::utils::ChronoTimingStats solve_t("slam.solve");
  tbb::task_group group;
  group.run([&] {
    dyno::utils::ChronoTimingStats vo_t("slam.vo");
    VioGraphEstimator& vio_estimator = estimator_.vioEstimator();
    frontend_.solveVisualOdometry(vio_estimator, features, *vo_geometry,
                                  async_context, updated_vo_states, vo_success,
                                  *image_container);
  });

  const VioGraphEstimator& vio_estimator = estimator_.vioEstimator();
  dyno::utils::ChronoTimingStats doo_t("slam.doo");
  for (size_t i = 0; i < num_objects; i++) {
    group.run([&, i] {
      ObjectId object_id = object_ids[i];
      ObjectContext object_context = object_contexts[i];
      uchar& solve_success = do_solve_success.at(i);

      std::vector<FrameId>& updated_frames = updated_object_frames.at(i);
      frontend_.solveObjectOdometry(object_id, vio_estimator, *image_container,
                                    *object_context.estimator, features,
                                    *object_context.geometry, async_context,
                                    updated_frames, solve_success);
    });
  }
  group.wait();
  solve_t.stop();
  doo_t.stop();

  std::vector<dyno::TemporalObjectIdentifier>& objects_with_update =
      publication_data->updated_states.objects;

  // tracking state information
  dyno::ViTrackingState& vi_tracking_state =
      publication_data->vi_tracking_state;
  vi_tracking_state.frame_id = frame_id;
  // TODO: quality etc

  std::vector<dyno::DoTrackingState>& do_tracking_states =
      publication_data->do_tracking_states;

  if (vo_success) {
    VioGraphEstimator& estimator = estimator_.vioEstimator();
    vi_tracking_state.is_keyframe = estimator.isKeyframe(frame_id);
    // prune keyframes
    estimator.pruneKeyframes(5, 3);
  }

  for (size_t i = 0; i < num_objects; i++) {
    const ObjectId object_id = object_ids[i];
    LOG(INFO) << "j= " << object_id
              << " solve success= " << static_cast<int>(do_solve_success[i]);
    if (!do_solve_success[i]) {
      continue;
    }

    const ObjectContext object_context = object_contexts[i];
    const DoGraphEstimator& estimator = *object_context.estimator;

    const std::vector<FrameId>& updated_frames = updated_object_frames.at(i);
    for (FrameId k0 : updated_frames) {
      objects_with_update.emplace_back(k0, object_id);
    }

    do_tracking_states.emplace_back();
    do_tracking_states.back().frame_id = frame_id;
    do_tracking_states.back().object_id = object_id;
    do_tracking_states.back().keyframe_type =
        estimator.keyframeStatus(frame_id);
  }

  const dyno::WrappedRGBMono wrapped_rgb = image_container->rgb();
  const cv::Mat rgb = wrapped_rgb.image();

  features.reduceToInliersInplace();

  pushImageToDisplayQueue("rgb", features.draw(rgb, true));
  pushPublicationDataToQueue(publication_data);

  return nullptr;
}

}  // namespace dyno2
