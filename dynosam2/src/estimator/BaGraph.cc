#include "dynosam2/estimator/BaGraph.hpp"

#include <gtsam/nonlinear/DoglegOptimizer.h>

namespace dyno2 {

void BaGraph::addMeasurement(Frame::Ptr frame, dyno::TrackletId id) {
  addStereoMeasurements(
      dyno::StereoMeasurementStatusVector{createStereoMeasurement(frame, id)});
}
void BaGraph::addMeasurement(dyno::KeypointIdentifier id) {
  const dyno::FrameId frame_id = id.frameId();
  const dyno::TrackletId tracklet_id = id.trackletId();

  const Frame::Ptr frame = getFrame(frame_id);
  CHECK_NOTNULL(frame);
  addMeasurement(frame, tracklet_id);
}
void BaGraph::addMeasurements(
    const std::vector<dyno::KeypointIdentifier>& ids) {
  // takes a long time since we have to do lots of lookups!
  for (const auto& id : ids) {
    addMeasurement(id);
  }
}
void BaGraph::addMeasurements(Frame::Ptr frame,
                              const std::vector<dyno::TrackletId>& ids) {
  StereoMeasurementStatusVector measurements;
  measurements.reserve(ids.size());
  for (dyno::TrackletId id : ids) {
    measurements.push_back(createStereoMeasurement(frame, id));
  }
  addStereoMeasurements(measurements);
}

// bool BaOptInterface::optimize(gtsam::Values& optimised_values, const
// gtsam::Values& values, const gtsam::NonlinearFactorGraph& graph) {
// optimised_values = values;

//   dyno::NonlinearOptimizerSummary summary;
//   // Dog leg is just way faster for will defined problems
//   gtsam::DoglegParams opt_params;
//   opt_params.setMaxIterations(2);

//   dyno::NonlinearOptimizer<gtsam::DoglegOptimizer> solver(graph, values,
//   opt_params); if(!solver.solve(optimised_values, options_, &summary)) {
//     return false;
//   }

//   VLOG(10) << "Initial error: " << summary.initial_error << " final error "
//            << summary.final_error << " time[s] "
//            << summary.cumulative_time_in_seconds
//            << " #iterations= " << summary.numIterations();
//   return true;

// }

}  // namespace dyno2
