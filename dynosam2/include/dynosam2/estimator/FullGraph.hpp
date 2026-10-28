#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <optional>
#include <thread>

#include "dynosam2/estimator/BaGraph.hpp"
#include "dynosam2/estimator/DoGraph.hpp"
#include "dynosam2/estimator/VioGraph.hpp"

namespace dyno2 {

struct FullGraphState {
  // std::optional<
};

typedef std::unordered_map<dyno::ObjectId, DoGraph*> DoGraphPtrMap;

// TODO: this probably doesn't even need both the frames internally (which are
// usually used to build the measurements) or the map since we solve
// incrementally we should just be getting measurements directly from the
// estimator's map and

/**
 * @brief Solves the full Dynamic SLAM problem via incremental inference using
 * iSAM2
 *
 */
class FullGraph : public States<FullGraphState>, public LandmarkMap {
 public:
  FullGraph(dyno::Camera::Ptr camera);
  ~FullGraph();

  void addStaticMeasurements(
      const VioGraph& estimator,
      const std::vector<KeypointIdentifier>& vo_measurements);

  inline bool needsFullGraphOptimisation() const {
    return needs_full_graph_opt_;
  }

  inline bool isRunningFullGraphOptimisation() const {
    return is_running_full_opt_;
  }

  inline bool isUpdateAvailable() const {
    return has_update_ && !is_running_full_opt_;
  }

  /// @brief All VIO (key) frames involved in the optimisation
  /// @return
  const std::set<dyno::FrameId>& vioKeyframes() const { return vio_keyframes_; }

  bool synchronise(VioGraph* vio_graph, const DoGraphPtrMap& do_graphs,
                   UpdatedStates& involved_states);

  // optimizes in own thread
  void optimize();

 private:
  /// @brief Incremental bundle to be added to the smoother as part of a single
  /// update
  struct IncrementalUpdate {
    // States that are involved in this bundle
    UpdatedStates involved_states;
    gtsam::NonlinearFactorGraph new_factors;
    gtsam::Values new_values;
  };

 private:
  // NOTE: we take the update by COPY so we can continue modify the (next)
  // latest update asynchronously!
  void optimizeImpl(IncrementalUpdate update);

  dyno::StereoCalibPtr K_stereo_;
  std::atomic_bool is_running_full_opt_;
  std::atomic_bool has_update_;
  std::atomic_bool needs_full_graph_opt_;
  std::unique_ptr<gtsam::ISAM2> isam_;

  // Current incremental bundle that will be added to the optimizer upon next
  // call to optimize
  IncrementalUpdate update_;
  //! States involved each time optimize was called
  std::vector<UpdatedStates> involved_states_;
  //! Index to the involved_states indicating which states have an update
  //! There may be multiple if we run multiple optimisations without consuming
  //! an update Clearned once the update is consumed
  std::vector<size_t> involved_states_to_sync_;

  std::set<dyno::FrameId> vio_keyframes_;

  std::thread full_graph_opt_thread_;
};

}  // namespace dyno2
