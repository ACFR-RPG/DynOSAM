#pragma once

#include "dynosam2/estimator/BaGraph.hpp"
#include "dynosam2/estimator/DoGraph.hpp"
#include "dynosam2/estimator/VioGraph.hpp"

namespace dyno2 {

struct FullGraphState {};

/**
 * @brief Solves the full Dynamic SLAM problem via incremental inference using
 * iSAM2
 *
 */
class FullGraph : public BaGraphStates<FullGraphState> {
 public:
  FullGraph() = default;
  ~FullGraph() = default;

  // void addStaticMeasurements(const VioGraphEstimator& estimator,
  // dyno::FrameId frame_id, const std::vector<TrackletId>& ids);
};

}  // namespace dyno2
