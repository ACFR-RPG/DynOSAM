#pragma once

#include "dynosam2/common/Types.hpp"
#include "dynosam2/frontend/Frame.hpp"
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

  BaGraph();
  virtual ~BaGraph() = default;

  /**
   * @brief If an observation has been turned into a measurement for the BA
   * graph, which (maybe) will included in the local BA problem
   *
   * @param frame_id
   * @param tracklet_id
   * @return true
   * @return false
   */
  bool hasMeasurement(dyno::FrameId frame_id,
                      dyno::TrackletId tracklet_id) const {
    // way faster to lookup by tracklet id
    auto lmk_node = observations_->getLandmark(tracklet_id);
    return lmk_node ? lmk_node->seenAtFrame(frame_id) : false;
  }

  inline bool hasMeasurement(const dyno::KeypointIdentifier& kpi) const {
    return this->hasMeasurement(kpi.frameId(), kpi.trackletId());
  }

  inline bool getMeasurement(const dyno::KeypointIdentifier& kpi,
                             StereoMeasurement& measurement) const {
    auto lmk_node = observations_->getLandmark(kpi.trackletId());
    if (!lmk_node) {
      return false;
    }

    return lmk_node->getMeasurement(kpi.frameId(), measurement);
  }

  const StereoBaMap& getObservations() const { return *observations_; }

  // Measurement frames, not frames in the BaMeasurementGraph
  size_t numFrames() const { return frames_.size(); }

  void setMeasurementSigma(double sigma) {
    CHECK_GT(sigma, 0);
    measurement_model_ = gtsam::noiseModel::Isotropic::Sigma(3, sigma);
  }

  gtsam::SharedGaussian getMeasurementModel() const {
    return measurement_model_;
  }

  // adds a measurement that exists in the Frame to the measurement graph,
  // indicating that it will be considered for BA!
  void addMeasurement(Frame::Ptr frame, dyno::TrackletId id);
  void addMeasurement(dyno::KeypointIdentifier id);
  void addMeasurements(const std::vector<dyno::KeypointIdentifier>& ids);
  // this is faster if all observations are for the same frame
  void addMeasurements(Frame::Ptr frame,
                       const std::vector<dyno::TrackletId>& ids);

  bool computeCovisibilities();
  int covisibilities(dyno::FrameId frame_i, dyno::FrameId frame_j) const;

 protected:
  inline Frame::Ptr getFrame(dyno::TemporalObjectIdentifier id) const {
    auto it = frames_.find(id);
    if (it == frames_.end()) {
      return nullptr;
    }
    return it->second;
  }

  template <typename FRAME>
  std::shared_ptr<FRAME> getFrameAs(dyno::TemporalObjectIdentifier id) const {
    Frame::Ptr frame = this->getFrame(id);
    return std::static_pointer_cast<FRAME>(frame);
  }

  inline void addStereoMeasurements(
      const dyno::StereoMeasurementStatusVector& measurements) {
    covisibilities_computed_ = false;
    observations_->updateObservations(measurements);
  }

  inline StereoMeasurementStatus createStereoMeasurement(
      const Frame::Ptr& frame, dyno::TrackletId id) const {
    // debug check frame has measurement!?
    // TODO: observation model robust?
    StereoMeasurement stereo_measurement(frame->stereoPointById(id),
                                         measurement_model_);
    return StereoMeasurementStatus(stereo_measurement, frame->getFrameId(),
                                   frame->getTimestamp(), id,
                                   frame->getObjectId(), ReferenceFrame::LOCAL);
  }

 protected:
  //! Represents measurements that have been accepted into the BA problem
  //! Not all keypoint observations, these are stored in Frame::Ptr
  StereoBaMap::Ptr observations_;

  bool covisibilities_computed_{false};
  Covisibilities covisibilities_;
  std::set<dyno::FrameId> visible_frames_;

  //! Measurement model used when constructing measurements
  gtsam::SharedGaussian measurement_model_;
  //! All frames added so far. Holds all keypoints.
  //! We use a TemporalObjectIdentifier rather than a frame_id so we can use
  //! this same structure to hold ViFrames and DoFrames
  dyno::FastUnorderedMap<dyno::TemporalObjectIdentifier, Frame::Ptr> frames_;
  // dyno::FastUnorderedMap<dyno::FrameId, Frame::Ptr> frames_;
};

class BaProblemInterface {
 public:
  BaProblemInterface() = default;

 protected:
  // NOTE: max iterations will only be used if setOptimisationTimeLimit has not
  // been set, otherwise the time-budget will be used
  NonlinearOptimizerSummary optimize(gtsam::Values& optimised_values,
                                     int max_iterations,
                                     bool verbose = false) const;

  // must perform at least min_iterations regardless of time budget
  bool setOptimisationTimeLimit(double time_limit_s, int min_iterations);
  bool isTimeLimitSet() const;
  void removeOptimisationTimeLimit();

  gtsam::NonlinearFactorGraph& getProblemGraph() { return graph_; }
  gtsam::Values& getProblemValues() { return values_; }

  void resetProblem();

 protected:
  dyno::NonlinearOptimizerOptions options_;

 private:
  std::unique_ptr<dyno::TimeBudgetOptimizationCallback> callback_{nullptr};
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values values_;
};

/// @brief Convenience struct to hold all observations and states together
/// @tparam STATE
template <typename STATE>
struct BaGraphStates : public BaGraph,
                       public States<STATE>,
                       public LandmarkMap {
  BaGraphStates() = default;
};

}  // namespace dyno2
