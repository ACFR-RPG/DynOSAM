#pragma once

#include "dynosam2/common/Types.hpp"
#include "dynosam_common/SensorModels.hpp"
#include "dynosam_common/StructuredContainers.hpp"
#include "dynosam_opt/Map.hpp"
#include "dynosam_opt/NonlinearOptimizer.hpp"

namespace dyno2 {

struct StereoBaNodeTypes {
  using Measurement = dyno::StereoMeasurement;
  using ObjectNodeT = dyno::ObjectNodeBase<StereoBaNodeTypes>;
  using FrameNodeT = dyno::FrameNodeBase<StereoBaNodeTypes>;
  using LandmarkNodeT = dyno::LandmarkNodeBase<StereoBaNodeTypes>;
};

/// @brief A map only containing measurements and nodes ready for BA
/// does not store any state information as this is contained within the BAGraph
class StereoBaMap : public dyno::Map<StereoBaNodeTypes> {
  struct Private {};

 public:
  using This = StereoBaMap;
  using Base = Map<StereoBaNodeTypes>;
  DYNO_POINTER_TYPEDEFS(This)

  // Constructor is only usable by this class
  StereoBaMap(Private) {}

  static std::shared_ptr<This> create() {
    return std::make_shared<This>(Private());
  }
};

template <typename STATE>
class States {
 public:
  typedef STATE State;

  // TODO: remove state if not valid

  size_t numStates() const noexcept { return states_.size(); }

  bool stateExists(dyno::FrameId frame_id) const {
    return states_.exists(frame_id);
  }
  dyno::FrameId currentFrameId() const { return states_.rbegin()->first; }

  dyno::FrameId frameIdbyAge(size_t age) const {
    CHECK_LT(age, states_.size()) << "Age exceeds available states";
    auto iter = states_.rbegin();
    std::advance(iter, age);
    return iter->first;
  }

 protected:
  State& lastState() { return states_.rbegin()->second; }

  const State& lastState() const { return states_.rbegin()->second; }

  State& state(dyno::FrameId frame_id) { return states_[frame_id]; }
  const State& state(dyno::FrameId frame_id) const {
    return states_.at(frame_id);
  }

  bool stateByAge(dyno::FrameId frame_id, size_t age, State*& state) {
    auto it = states_.find(frame_id);
    if (it == states_.end()) {
      return false;
    }

    for (size_t i = 0; i < age; ++i) {
      if (it == states_.begin()) {
        return false;
      }
      --it;
    }

    state = &it->second;
    return true;
  }

 protected:
  gtsam::FastMap<dyno::FrameId, State> states_;
};

class BaGraph {
 public:
  DYNO_POINTER_TYPEDEFS(BaGraph)

  using FrameCovisibilities =
      dyno::FastUnorderedMap<dyno::FrameId, dyno::TrackletIds>;
  using Covisibilities =
      dyno::FastUnorderedMap<dyno::FrameId, FrameCovisibilities>;

  BaGraph() : observations_(StereoBaMap::create()) {}
  virtual ~BaGraph() = default;

  bool hasObservation(dyno::FrameId frame_id,
                      dyno::TrackletId tracklet_id) const {
    // way faster to lookup by tracklet id
    auto lmk_node = observations_->getLandmark(tracklet_id);
    if (!lmk_node) {
      return false;
    }
    return lmk_node->seenAtFrame(frame_id);
  }

  const StereoBaMap& getObservations() const { return *observations_; }

  bool computeCovisibilities();
  int covisibilities(dyno::FrameId frame_i, dyno::FrameId frame_j) const;

  // all measurements should exists for states?
  inline void addMeasurements(
      const dyno::StereoMeasurementStatusVector& measurements) {
    covisibilities_computed_ = false;
    observations_->updateObservations(measurements);
  }

 protected:
  StereoBaMap::Ptr observations_;

  bool covisibilities_computed_{false};
  Covisibilities covisibilities_;
  std::set<dyno::FrameId> visible_frames_;
};

/// @brief Convenience struct to hold all observations and states together
/// @tparam STATE
template <typename STATE>
struct BaGraphStates : public BaGraph,
                       public States<STATE>,
                       public LandmarkMap {
  BaGraphStates() = default;
};

// class BaOptInterface {
// public:
//   BaOptInterface() = default;

// protected:
//   bool optimize(gtsam::Values& optimised_values, const gtsam::Values& values,
//   const gtsam::NonlinearFactorGraph& graph);

// protected:
//   dyno::NonlinearOptimizerOptions options_;

// };

}  // namespace dyno2
