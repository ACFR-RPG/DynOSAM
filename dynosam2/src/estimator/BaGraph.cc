#include "dynosam2/estimator/BaGraph.hpp"

#include <gtsam/nonlinear/DoglegOptimizer.h>

namespace dyno2 {

BaGraph::BaGraph(dyno::Camera::Ptr camera)
    : camera_(camera),
      K_stereo_(dyno::RGBDCamera::Calibration(camera)),
      observations_(StereoBaMap::create()) {
  setMeasurementSigma(2.0);
}

void BaGraph::addMeasurement(Frame::Ptr frame, dyno::TrackletId id) {
  addStereoMeasurements(
      dyno::StereoMeasurementStatusVector{createStereoMeasurement(frame, id)});
}
void BaGraph::addMeasurement(dyno::KeypointIdentifier id) {
  const Frame::Ptr frame = getFrame(id.temoralObjectIdentifier());
  CHECK_NOTNULL(frame);
  addMeasurement(frame, id.trackletId());
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

NonlinearOptimizerSummary BaProblemInterface::optimize(
    gtsam::Values& optimised_values, int max_iterations, bool verbose) const {
  gtsam::DoglegParams params;
  params.setMaxIterations(max_iterations);

  dyno::NonlinearOptimizer<gtsam::DoglegOptimizer> solver(graph_, values_,
                                                          params);

  dyno::NonlinearOptimizerSummary summary;
  optimised_values = values_;

  CHECK(solver.solve(optimised_values, options_, &summary));

  return summary;
}

bool BaProblemInterface::setOptimisationTimeLimit(double time_limit_s,
                                                  int min_iterations) {
  if (callback_ != nullptr) {
    // clamp time to zero
    // in this case we just use the iteration count
    double time_limit = std::max(time_limit_s, 0.0);
    callback_->setTimeLimit(time_limit);
    callback_->setMinimumIterations(min_iterations);
    return true;
  } else if (time_limit_s >= 0.0) {
    callback_ = std::unique_ptr<dyno::TimeBudgetOptimizationCallback>(
        new dyno::TimeBudgetOptimizationCallback(time_limit_s, min_iterations));
    options_.callbacks.push_back(callback_.get());
    return true;
  }
  // no callback yet registered with ceres.
  // but given time limit is lower than 0, so no callback needed
  return true;
}

bool BaProblemInterface::isTimeLimitSet() const { return callback_ != nullptr; }

void BaProblemInterface::removeOptimisationTimeLimit() {
  if (isTimeLimitSet()) {
    auto* ptr = callback_.get();
    auto& callbacks = options_.callbacks;
    // Remove the dangling pointer from the vector.
    callbacks.erase(std::remove(callbacks.begin(), callbacks.end(), ptr),
                    callbacks.end());
    callback_.reset();
  }
}

void BaProblemInterface::resetProblem() {
  graph_ = gtsam::NonlinearFactorGraph();
  values_.clear();
}

}  // namespace dyno2
