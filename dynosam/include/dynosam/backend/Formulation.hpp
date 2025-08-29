/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include <gtsam/nonlinear/ISAM2Result.h>
#include <gtsam/nonlinear/ISAM2UpdateParams.h>

#include "dynosam/backend/Accessor.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/common/Types.hpp"

namespace dyno {

template <typename MAP>
struct MapTraits {
  using Map = MAP;
  using MeasurementType = typename Map::Measurement;
  using FrameNode = typename Map::FrameNodeM;
  using ObjectNode = typename Map::ObjectNodeM;
  using LandmarkNode = typename Map::LandmarkNodeM;

  using FrameNodePtr = typename FrameNode::Ptr;
  using ObjectNodePtr = typename ObjectNode::Ptr;
  using LandmarkNodePtr = typename LandmarkNode::Ptr;
};

struct UpdateObservationParams {
  //! If true, vision related updated will backtrack to the start of a new
  //! tracklet and all the measurements to the graph should make false in batch
  //! case where we want to be explicit about which frames are added!
  bool do_backtrack = false;
  bool enable_debug_info = true;
  bool enable_incremental_detail = false;
};

struct UpdateObservationResult {
  gtsam::FastMap<ObjectId, std::set<FrameId>>
      objects_affected_per_frame;  // per frame
  DebugInfo::Optional debug_info{};

  // Incremental interface
  // TODO: should be optional
  // Doubles as any fixed lag interface
  gtsam::ISAM2UpdateParams isam_update_params;

  struct BatchUpdateParams {
    //! Since no batch algorithm (even fixed-lag smoothers) have a 1-to-1 factor
    //! index like ISAM2, this interface will find the factors manually
    gtsam::NonlinearFactorGraph factors_to_remove;
  };

  //! Batch update params.
  //! TODO: should be optional for only when running with batch!!
  BatchUpdateParams batch_update_params;

  //! A custom set of keys that wont be added to the timestamp map of a fixed
  //! lag smoother (when used) This prevents the keys being marginalized out
  //! when they leave the lag horizin; Should be keys added during 'this' update
  //! step
  gtsam::KeySet keys_to_not_marginalize;

  UpdateObservationResult() {}

  // TODO: use the UpdateObservationParams to set if we have an incremental
  // update or batch update params!!
  UpdateObservationResult(const UpdateObservationParams& update_params) {
    if (update_params.enable_debug_info) {
      this->debug_info = DebugInfo();
    }
  }

  void updateAffectedObject(FrameId frame_id, ObjectId object_id) {
    if (!objects_affected_per_frame.exists(object_id)) {
      objects_affected_per_frame.insert2(object_id, std::set<FrameId>{});
    }
    objects_affected_per_frame[object_id].insert(frame_id);
  }
};

template <typename MAP>
struct PointUpdateContext {
  using MapTraitsType = MapTraits<MAP>;

  typename MapTraitsType::LandmarkNodePtr lmk_node;
  typename MapTraitsType::FrameNodePtr frame_node_k_1;
  typename MapTraitsType::FrameNodePtr frame_node_k;

  //! Camera pose from measurement (or initial)
  gtsam::Pose3 X_k_measured;
  //! Camera pose from measurement (or initial)
  gtsam::Pose3 X_k_1_measured;

  //! When an update starts only a subset of the factors are provided to the
  //! update This value indicates the factor slot offset (ie the total graph
  //! size before any update)
  Slot starting_factor_slot = -1;

  //! If true then this frame is the first frame where a motion is available
  //! (i.e we have a pair of valid frames) e.g k-1 is the FIRST frame for this
  //! object and now, since we are at k, we can create a motion from k-1 to k
  bool is_starting_motion_frame{false};

  inline ObjectId getObjectId() const {
    return lmk_node->template getObjectId();
  }
  inline TrackletId getTrackletId() const {
    return lmk_node->template tracklet_id;
  }
};

template <typename MAP>
struct ObjectUpdateContext {
  using MapTraitsType = MapTraits<MAP>;
  //! Frame that is part of the update context. Shared pointer to a frame node
  //! as defined by the Map type
  typename MapTraitsType::FrameNodePtr frame_node_k;
  //! Object that is part of the update context. Shared pointer to a object node
  //! as defined by the Map type
  typename MapTraitsType::ObjectNodePtr object_node;
  //! Indicates that we have a valid motion pair from k-1 to k (this frame)
  //! and therefore k is at least the second frame for which this object has
  //! been consequatively tracked When this is false, it means that the frame
  //! k-1 (getFrameId() - 1u) did not track/observe this object and therfore no
  //! motion can be created between them. This happens on the first observation
  //! of this object.
  bool has_motion_pair{false};

  inline FrameId getFrameId() const { return frame_node_k->template getId(); }
  inline ObjectId getObjectId() const { return object_node->template getId(); }
};

template <typename MAP>
struct OtherUpdateContext {
  using MapTraitsType = MapTraits<MAP>;
  //! Frame that is part of the update context. Shared pointer to a frame node
  //! as defined by the Map type
  typename MapTraitsType::FrameNodePtr frame_node_k;

  // TODO: hack for now as we need timestamp in dynosam mpc
  // make easily avaiable for all contexts/formulations
  Timestamp timestamp;

  inline FrameId getFrameId() const { return frame_node_k->template getId(); }
};

// forward declare
class BackendParams;

/**
 * @brief Metadata struct used for the Formulation::preUpdate hook.
 *
 */
struct PreUpdateData {
  FrameId frame_id;

  PreUpdateData() {}
  PreUpdateData(FrameId _frame_id) : frame_id(_frame_id) {}
};

/**
 * @brief Metdata struct used for the Formulation::postUpdate hook.
 *
 */
struct PostUpdateData {
  FrameId frame_id;
  Timestamp timestamp;
  UpdateObservationResult dynamic_update_result;
  UpdateObservationResult static_update_result;
  UpdateObservationResult other_update_result;

  struct IncrementalResult {
    //! Result from the incremental update
    gtsam::ISAM2Result isam2;
    //! Current set of factors from the (incremental) smoother
    gtsam::NonlinearFactorGraph factors;
  };

  // TODO: batch result!!!

  std::optional<IncrementalResult> incremental_result = {};

  PostUpdateData() {}
  PostUpdateData(FrameId _frame_id, Timestamp _timestamp)
      : frame_id(_frame_id), timestamp(_timestamp) {}
};

/**
 * @brief Basic parameters for a Formulation.
 * Right now is just directly the BackendParams since there is lots of
 * cross-over, but later we may add formulation specific ones
 *
 */
struct FormulationParams : public BackendParams {
  FormulationParams() {}

  FormulationParams(const BackendParams& b) { FormulationParams::operator=(b); }

  FormulationParams& operator=(const BackendParams& b) {
    static_cast<BackendParams&>(*this) = b;
    return *this;
  }
};

class Camera;

/**
 * @brief Encapsulation of various sensor types, giving the Formulation access
 * to different models
 *
 */
struct Sensors {
  std::shared_ptr<Camera> camera;

  Sensors() {}
};

/**
 * @brief Base class for a formulation that defines the structure and
 * implementation for a factor-graph based Dynamic SLAM solution. Derived
 * formulations construct the specific factor-graph which are solved
 * independandtly to the formulation. The formulation is responsible for the
 * construction of a single factor-graph with a single set of initial values and
 * has an associated Accessor which knows how to extract these values into a
 * common format. In the DynoSAM paper, this is referred to as the \mathbf{O}_k,
 * which is the output of the backend. See Accessor.hpp for more details.
 *
 * The base formulation manages all high-level book keeping associated with any
 * Dynamic SLAM factor graph and decided when points and objects should be added
 * to the graph in a general sense (e.g. points must be seen multiple times, has
 * this point already been added the graph in a general way). This is achieved
 * in two ways:
 * 1. Key virtual functions which provides the base formulation with some
 * knowledge of the dervied formulations actions and
 * 2. Access to the MAP which defines the structure of frames, objects and
 * points directly from measurements.
 *
 * At a high-level the formulation has the following functionalities:
 *  - Addition of odometry and pose priors
 *  - Static point updates (which adds factors and values for static points)
 *  - Dynamic point updates (which adds factors and values for dynamic points
 * AND dynamic objects).
 *
 * The derived class must implement the dynamicPointUpdateCallback and
 * objectUpdateContext pure-virtual functions and updated the
 * UpdateObservationResult,gtsam::Values and gtsam::NonlinearFactorGraph objects
 * based on the additional context that is parsed in. The context is build by
 * the base-formulation class as part of the high-level bookeeping and indicates
 * that this dynamic point/object has new observations associated with it and
 * therefore new values/factors should be added. The new values/factors will
 * eventually be added to the base-formualtions internal theta_ and factors_.
 * These virtual functions are called in the corresponding update static/dynamic
 * observation functions when the base class has determined new obsevrations
 * have occurned.
 *
 * @tparam MAP
 */
template <typename MAP>
class Formulation {
 public:
  using Map = MAP;
  using This = Formulation<Map>;
  using MapTraitsType = MapTraits<Map>;
  using MeasurementType = typename MapTraitsType::MeasurementType;

  using MeasurementTraits = measurement_traits<MeasurementType>;

  using PointUpdateContextType = PointUpdateContext<Map>;
  using ObjectUpdateContextType = ObjectUpdateContext<Map>;
  using OtherUpdateContextType = OtherUpdateContext<Map>;
  using AccessorType = Accessor<MAP>;
  using AccessorTypePointer = typename Accessor<MAP>::Ptr;

  DYNO_POINTER_TYPEDEFS(This)

  Formulation(const FormulationParams& params, typename Map::Ptr map,
              const NoiseModels& noise_models, const Sensors& sensors,
              const FormulationHooks& hooks = FormulationHooks());
  virtual ~Formulation() = default;

 protected:
  /**
   * @brief Virtual function to be implemented by derived class and indicates
   * that a specific landmark has new measurements and new values/factors should
   * be added. Details on the landmark to be updated are provied by the context.
   * In general, this is used to add projection/motion factors and initial
   * values for landmarks.
   *
   * This function is called when, for a frame k, a point has new measurements
   * (from the front-end) and the point is well-tracked (e.g has been seen at
   * least n times).
   *
   * If the point is new the gtsam::Values should be updated and
   * gtsam::NonlinearFactorGraph should be updated with new factors. Note that
   * the PointUpdateContextType is constructed in-terms of a MOTION so that
   * information is provided about the k-1 and k'th observation. In the case
   * that this point is first observed, this construction means that the point
   * has been seen twice (at k-1 and k).
   *
   * UpdateObservationResult MUST also be updated if a new value or factor is
   * added as this tells the base formulation that new information has been
   * added to the graph. The result is used to determine which object's to then
   * form a ObjectUpdateContextType for.
   *
   * This function should also set any internal logic to ensure that after a new
   * point has been initalised/added, isDynamicTrackletInMap returns true as
   * this is used internally.
   *
   * For all factor-graph construction functions, all functions additionally
   * return (via out-args) all new factors and values added during the function.
   * These new values/factors will ALSO be added to the internal
   * theta_/factors_. This allows the Formulation to be used for batch
   * optimisation, by accessing the full graph (getTheta(), getGraph()) or
   * incremental optimisation by updating the incremental optimser using just
   * the new values/factors returned from the update functions.
   *
   * NOTE all point update callbacks are triggered before any
   * objectUpdateContext, although the order is arbitrary.
   *
   *
   * @param context const PointUpdateContextType&
   * @param result UpdateObservationResult&
   * @param new_values gtsam::Values&
   * @param new_factors gtsam::NonlinearFactorGraph&
   */
  virtual void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) = 0;

  /**
   * @brief Virtual function to be implemented by derived class and indicates
   * that a specific object j and time-step k has been observed and its dynamic
   * points updated in the graph, and associated values/factors should be added.
   * Details on the object to be updated are provied by the context. In general,
   * this is used to add smoothing factors and initial values for object
   * motion/pose.
   *
   * This function is called when, for a frame k, the object has been seen at
   * least m times and there is at least n points on this object that have new
   * values/factors since the last iteration (as determined by the
   * UpdateObservationResult from dynamicPointUpdateCallback).
   *
   * NOTE this function is called independanly for all objects and is triggered
   * after all dynamicPointUpdateCallback's been called.
   *
   * @param context const ObjectUpdateContextType& context
   * @param result UpdateObservationResult&
   * @param new_values gtsam::Values&
   * @param new_factors gtsam::NonlinearFactorGraph&
   */
  virtual void objectUpdateContext(
      const ObjectUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) = 0;

  // TODO: how to handle static vs dynamic (ie in parallel!!!)
  /**
   * @brief Virtual function that allows any other updates to the factor graph
   * to occur at a specific time-step k.
   *
   * These updates should not non vison related as these are handled by the
   * objectUpdate and dynamicPointUpdateCallback functions.
   *
   * NOTE: this function should not be called directly and instead should be
   * called via the updateOtherObservations function.
   *
   */
  virtual void otherUpdatesContext(const OtherUpdateContextType&,
                                   UpdateObservationResult&, gtsam::Values&,
                                   gtsam::NonlinearFactorGraph&) {}

  /**
   * @brief Virtual function to be implemented by derived class and indicates
   * that the specified landmark exists within the graph already. This is used
   * as part of the internal bookeeping.
   *
   * @param lmk_node typename MapTraitsType::LandmarkNodePtr&. Pointer to a
   * landmark node.
   * @return true
   * @return false
   */
  virtual bool isDynamicTrackletInMap(
      const typename MapTraitsType::LandmarkNodePtr& lmk_node) const = 0;

  /**
   * @brief Defines the base name of this formulation. This name will appear as
   * part of the output logs and may be augmented with the
   * FormulationParams::suffix, if given. To get the fully qualified name (the
   * actual name that will appear in the logs), use
   * This::getFullyQualifiedName().
   *
   * @return std::string
   */
  virtual std::string loggerPrefix() const = 0;

  /**
   * @brief Creates the associated Accessor for the derived formulation. The
   * accessor will contain a pointer to This::theta_, so that when the values
   * are updated the accessor has access to the latest values.
   *
   * @param values const gtsam::Values*
   * @return AccessorType::Ptr
   */
  virtual typename AccessorType::Ptr createAccessor(
      const SharedFormulationData& shared_data) const = 0;

 public:
  /**
   * @brief Get the map used by the formulation
   *
   * @return Map::Ptr
   */
  typename Map::Ptr map() const { return map_; }

  /**
   * @brief Get the current linearization point for this formulation. These
   * values can be updated after external optimisation and are constructed by
   * the derived formulation during update.
   *
   * @return const gtsam::Values&
   */
  const gtsam::Values& getTheta() const { return theta_; }

  /**
   * @brief Sets the full set of theta values (overrides them). This is used
   * after external optimisation to update the values for the accessor which has
   * a internal reference (pointer) to theta.
   *
   * @param linearization const gtsam::Values&
   */
  void setTheta(const gtsam::Values& linearization);

  /**
   * @brief Updates and assigns new values to theta. This is used after external
   * optimisation to update the values for the accessor which has a internal
   * reference (pointer) to theta.
   *
   * @param linearization const gtsam::Values&
   */
  void updateTheta(const gtsam::Values& linearization);

  /**
   * @brief Gets the current graph associated with the current theta and
   * constructed by the derived formulation. Along theta, this should be used by
   * an external optimisation routine.
   *
   * @return const gtsam::NonlinearFactorGraph&
   */
  const gtsam::NonlinearFactorGraph& getGraph() const { return factors_; }

  const FormulationHooks& hooks() const { return hooks_; }
  const NoiseModels& noiseModels() const { return noise_models_; }
  const Sensors& sensors() const { return sensors_; }

  /**
   * @brief Custom gtsam::Key formatter for this formulation.
   * Defaults to DynoLikeKeyFormatter.
   *
   * @return gtsam::KeyFormatter
   */
  virtual gtsam::KeyFormatter formatter() const { return DynoLikeKeyFormatter; }

  /**
   * @brief Get the fully qualified name of this formulation which is derived
   * from the loggerPrefix and optionally the FormulationParams::suffix if
   * given.
   *
   * @return const std::string
   */
  const std::string getFullyQualifiedName() const {
    return fully_qualified_name_.value_or(setFullyQualifiedName());
  }

  /**
   * @brief Public fucntion to access the Accessor for the derived formulation.
   *
   * @return AccessorType::Ptr
   */
  typename AccessorType::Ptr accessorFromTheta() const;

  template <typename Derived>
  std::shared_ptr<Derived> derivedAccessor() const {
    if (!accessor_theta_) {
      accessorFromTheta();
    }
    CHECK(accessor_theta_);
    return std::dynamic_pointer_cast<Derived>(accessor_theta_);
  }

  /**
   * @brief Makes a logger using the fully-qualified name for this formulation.
   *
   * @return BackendLogger::UniquePtr
   */
  BackendLogger::UniquePtr makeFullyQualifiedLogger() const;

  void addValuesFunctional(std::function<void(gtsam::Values&)> callback,
                           gtsam::Values& new_values);
  void addFactorsFunctional(
      std::function<void(gtsam::NonlinearFactorGraph&)> callback,
      gtsam::NonlinearFactorGraph& new_factors);

  // Factor Graph build functions.
  void addSensorPoseValue(const gtsam::Pose3& X_W_k, FrameId frame_id_k,
                          gtsam::Values& new_values);
  void addSensorPosePriorFactor(const gtsam::Pose3& X_W_k,
                                gtsam::SharedNoiseModel noise_model,
                                FrameId frame_id_k,
                                gtsam::NonlinearFactorGraph& new_factors);

  /**
   * @brief Adds a gtsam::Pose3 as a value in the graph at frame k.
   *
   * @param T_world_camera const gtsam::Pose3&
   * @param frame_id_k FrameId
   * @param new_values gtsam::Values&
   */
  void setInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k,
                      gtsam::Values& new_values);
  /**
   * @brief Adds a pose prior on a value in the graph at frame k using the given
   * pose value as the prior's mean.
   *
   * @param T_world_camera const gtsam::Pose3&
   * @param frame_id_k FrameId
   * @param new_factors gtsam::NonlinearFactorGraph&
   */
  void setInitialPosePrior(const gtsam::Pose3& T_world_camera,
                           FrameId frame_id_k,
                           gtsam::NonlinearFactorGraph& new_factors);

  /**
   * @brief Updates the static-point part of the factor graph for frame k.
   * This contains all the bookeeping logic to update the factor graph with
   * static points.
   *
   * @param frame_id_k FrameId
   * @param new_values gtsam::Values&
   * @param new_factors gtsam::NonlinearFactorGraph&
   * @param update_params const UpdateObservationParams&
   * @return UpdateObservationResult
   */
  UpdateObservationResult updateStaticObservations(
      FrameId frame_id_k, gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors,
      const UpdateObservationParams& update_params);

  /**
   * @brief Updates the dyanmic-point part of the factor graph for frame k.
   * This contains all the big-boi bookeeping logic to update the factor graph
   * with dyanmic points and objects. This is where the virtual
   * dynamicPointUpdateCallback and objectUpdateContext functions are called.
   *
   * @param frame_id_k
   * @param new_values
   * @param new_factors
   * @param update_params
   * @return UpdateObservationResult
   */
  UpdateObservationResult updateDynamicObservations(
      FrameId frame_id_k, gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors,
      const UpdateObservationParams& update_params);

  // TODO: update all to have timestamps easily available!!
  /**
   * @brief Fills new values and factors based on any other miscelaneous
   * observations for frame k. This allows a formulation to be constructed that
   * contains additional information other than just static/dynamic point
   * variables.
   *
   * This function is called after all static/dynamic point measurements but
   * before handleExternalMeasurements. This function is meant to differ from
   * the handling of external measurements as this allows values to be
   * constructed from the current set of values in the formulation while
   * external measurements are fixed in their behaviour.
   *
   * Internally this calls the otherUpdatesContext function.
   *
   * @param frame_id_k
   * @param new_values
   * @param new_factors
   */
  UpdateObservationResult updateOtherObservations(
      Timestamp timestamp_k, FrameId frame_id_k, gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors);

  /**
   * @brief Logs all frames and values to file using Accessor and BackendLogger.
   * The BackendMetaData provides meta-data and ground truth information for
   * logging.
   *
   * @param backend_info const BackendMetaData&
   */
  void logBackendFromMap(const BackendMetaData& backend_info);

  /**
   * @brief Pre update hook called in the RegularBackend after the map is
   * updated with new measurements but before new values/factors are constructed
   * from this formulation.
   *
   */
  virtual void preUpdate(const PreUpdateData&){};

  /**
   * @brief Post-update hook called in the RegulatBackend after graph
   * construction and optimization!
   *
   */
  virtual void postUpdate(const PostUpdateData&){};

 protected:
  gtsam::Pose3 getInitialOrLinearizedSensorPose(FrameId frame_id) const;

  void clearGraph() { factors_.resize(0); }

 private:
  std::string setFullyQualifiedName() const;  // but isnt actually const ;)

 protected:
  const FormulationParams params_;
  typename Map::Ptr map_;
  const NoiseModels noise_models_;
  Sensors sensors_;
  FormulationHooks hooks_;
  //! the set of (static related) values managed by this updater. Allows
  //! checking if values have already been added over successive function calls
  gtsam::FastMap<gtsam::Key, bool> is_other_values_in_map;
  //! Current linearisation that will be associated with the current graph
  gtsam::Values theta_;
  gtsam::NonlinearFactorGraph factors_;

 private:
  mutable typename AccessorType::Ptr accessor_theta_;
  //! Full name of the formulation and accounts for the additional configuration
  //! from the FormulationParams
  mutable std::optional<std::string> fully_qualified_name_{std::nullopt};
};

}  // namespace dyno

#include "dynosam/backend/Formulation-impl.hpp"
