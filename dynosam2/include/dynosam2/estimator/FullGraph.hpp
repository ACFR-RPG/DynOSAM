#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "dynosam2/estimator/BaGraph.hpp"
#include "dynosam2/estimator/DoGraph.hpp"
#include "dynosam2/estimator/VioGraph.hpp"

namespace dyno2 {

struct FullGraphState {};

// TODO: this probably doesn't even need both the frames internally (which are
// usually used to build the measurements) or the map since we solve
// incrementally we should just be getting measurements directly from the
// estimator's map and

/**
 * @brief Solves the full Dynamic SLAM problem via incremental inference using
 * iSAM2
 *
 */
class FullGraph /*: public BaGraphStates<FullGraphState>*/ {
 public:
  FullGraph(dyno::Camera::Ptr camera);
  ~FullGraph() = default;

  void addStaticMeasurements(
      const VioGraph& estimator,
      const std::vector<KeypointIdentifier>& vo_measurements);

 private:
  dyno::StereoCalibPtr K_stereo_;

  /// @brief Incremental bundle to be added to the smoother as part of a single
  /// update
  struct IncrementalUpdate {
    // States that are involved in this bundle
    UpdatedStates involved_states;
    gtsam::NonlinearFactorGraph new_factors;
    gtsam::Values new_values;
  };
  IncrementalUpdate update_;

  std::unique_ptr<gtsam::ISAM2> isam_;
};

}  // namespace dyno2
