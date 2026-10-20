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
      estimator_(camera_) {}

DynamicSlam::~DynamicSlam() {}

StateEstimate::Ptr DynamicSlam::spinOnce(
    dyno::VIFrontendInput::ConstPtr input) {
  LOG(INFO) << "DynamicSlam::spinOnce " << input->getFrameId();
  dyno::utils::ChronoTimingStats spin_t("spin");

  const dyno::FrameId frame_id = input->getFrameId();
  const dyno::Timestamp timestamp = input->getTimestamp();

  dyno::ImageContainer& image_container = *input->image_container_;
  FeatureBlockContainer* features_ptr{nullptr};
  dyno::ObjectDetectionResult object_detection;

  frontend_.detectAndTrack(features_ptr, object_detection, image_container);
  CHECK_NOTNULL(features_ptr);
  FeatureBlockContainer& features = *features_ptr;

  FeatureBlockGeometryMap geometries;
  GeometryUpdater(camera_, image_container, features).calcPoints(geometries);

  if (first_frame_) {
    for (const auto& [object_id, geometry] : geometries) {
      Frame* frame_ptr{nullptr};
      if (object_id > background_label) {
        DoFrame& do_frame = do_frames_[object_id][frame_id];
        do_frame.object_id = object_id;
        frame_ptr = &do_frame;
      } else {
        ViFrame& vi_frame = vi_frames_[frame_id];
        frame_ptr = &vi_frame;
      }

      CHECK_NOTNULL(frame_ptr);
      frame_ptr->frame_id = frame_id;
      frame_ptr->timestamp = timestamp;
      frame_ptr->fill(geometry, features);
    }

    VioGraphEstimator& vio_estimator = estimator_.vioEstimator();
    // TODO: add measurements!
    vio_estimator.initaliseStates(frame_id, StereoMeasurementStatusVector{},
                                  gtsam::Pose3::Identity());
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
    DoFrames* frames;
    DoGraphEstimator* estimator;
  };

  std::vector<ObjectContext> object_contexts;
  // only include object ids here
  std::vector<ObjectId> object_ids;
  // new do frames that will be filled by each Doo solve
  std::vector<DoFrame*> do_frames;

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
      context.frames = CHECK_NOTNULL(&do_frames_[object_id]);
      context.estimator = CHECK_NOTNULL(&estimator_.doEstimator(object_id));
      object_contexts.push_back(context);

      DoFrame& do_frame = (*context.frames)[frame_id];
      do_frame.frame_id = frame_id;
      do_frame.timestamp = timestamp;
      do_frame.object_id = object_id;
      do_frames.push_back(&do_frame);

      object_ids.push_back(object_id);
    } else if (object_id == background_label) {
      vo_geometry = &geometry;
    }
  }

  CHECK(vo_geometry);

  bool vo_success{false};
  // allocate new vi_frame before thead
  ViFrame& vi_frame = vi_frames_[frame_id];
  vi_frame.frame_id = frame_id;
  vi_frame.timestamp = timestamp;

  gtsam::Pose3 X_W_k;
  AyncVO async_context;
  // setup context to write visual odometry into this alue
  async_context.vo = &X_W_k;
  CHECK_NOTNULL(async_context.vo);

  const size_t num_objects = object_ids.size();
  std::vector<uchar> do_solve_success(num_objects, 0);

  std::vector<std::vector<FrameId>> updated_object_frames(num_objects);

  PublicationData::Ptr publication_data =
      std::make_shared<PublicationData>(frame_id, timestamp);
  std::vector<dyno::FrameId>& updated_vo_states =
      publication_data->updated_states.vo;

  tbb::task_group group;
  group.run([&] {
    VioGraphEstimator& vio_estimator = estimator_.vioEstimator();
    frontend_.solveVisualOdometry(vio_estimator, features, *vo_geometry,
                                  vi_frame, async_context, updated_vo_states,
                                  vo_success, vi_frames_, image_container);
  });

  const VioGraphEstimator& vio_estimator = estimator_.vioEstimator();
  for (size_t i = 0; i < num_objects; i++) {
    group.run([&, i] {
      ObjectId object_id = object_ids[i];
      ObjectContext object_context = object_contexts[i];
      uchar& solve_success = do_solve_success.at(i);
      DoFrame& do_frame = *do_frames.at(i);

      std::vector<FrameId>& updated_frames = updated_object_frames.at(i);

      frontend_.solveObjectOdometry(
          object_id, vio_estimator, image_container, *object_context.frames,
          *object_context.estimator, features, *object_context.geometry,
          do_frame, async_context, updated_frames, solve_success);
    });
  }

  group.wait();

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

  // dyno::ImageContainer::Ptr image_container = input->image_container_;
  const dyno::WrappedRGBMono wrapped_rgb = image_container.rgb();
  const cv::Mat rgb = wrapped_rgb.image();

  features.reduceToInliersInplace();

  pushImageToDisplayQueue("rgb", features.draw(rgb, true));

  pushPublicationDataToQueue(publication_data);

  return nullptr;
}

}  // namespace dyno2
