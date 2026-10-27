#include "dynosam2/estimator/FullGraph.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/StereoFactor.h>

#include "dynosam_opt/FactorGraphTools.hpp"

namespace dyno2 {

using GenericStereoFactor =
    gtsam::GenericStereoFactor<gtsam::Pose3, dyno::Landmark>;

FullGraph::FullGraph(dyno::Camera::Ptr camera) {
  std::shared_ptr<dyno::RGBDCamera> rgbd_camera =
      CHECK_NOTNULL(camera->safeGetRGBDCamera());
  K_stereo_ = rgbd_camera->getFakeStereoCalib();
  CHECK_NOTNULL(K_stereo_);

  gtsam::ISAM2Params params;
  isam_ = std::make_unique<gtsam::ISAM2>(params);
}

void FullGraph::addStaticMeasurements(
    const VioGraph& estimator,
    const std::vector<KeypointIdentifier>& vo_measurements) {
  dyno::StereoMeasurementStatusVector stereo_measurements;
  stereo_measurements.resize(vo_measurements.size());

  std::unordered_set<dyno::FrameId> involved_frames;
  for (const auto& kpi : vo_measurements) {
    StereoMeasurement stereo_measurement;
    CHECK(estimator.getMeasurement(kpi, stereo_measurement));

    const dyno::TrackletId& tid = kpi.trackletId();
    CHECK(estimator.landmarkExists(tid));

    auto [measurement, model] = stereo_measurement;
    model = dyno::factor_graph_tools::robustifyHuber(0.01, model);

    const dyno::FrameId fid = kpi.frameId();
    const gtsam::Symbol pose_sym('x', fid);
    const gtsam::Symbol lmk_sym('l', tid);
    involved_frames.insert(fid);

    // stereo_measurements.push_back(createStereoMeasurement(frame,
    // kpi.trackletId()));
    update_.new_factors.emplace_shared<GenericStereoFactor>(
        measurement, model, pose_sym, lmk_sym, K_stereo_);

    // not in the estimator and not yet added
    if (!isam_->valueExists(tid) && !update_.new_values.exists(lmk_sym)) {
      // for some gross reason we cannot see the getLandmark function in the
      // estimator that is available as part of the LandmarkMap
      const dyno::Landmark& lmk = estimator.getLandmark(tid);
      update_.new_values.insert(lmk_sym, lmk);
    }
  }

  update_.involved_states.vo.insert(update_.involved_states.vo.begin(),
                                    involved_frames.begin(),
                                    involved_frames.end());
  LOG(INFO) << "Added " << stereo_measurements.size()
            << " factors for full estimation";
}

}  // namespace dyno2
