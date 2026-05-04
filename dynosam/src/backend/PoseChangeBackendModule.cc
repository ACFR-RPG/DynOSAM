#include "dynosam/backend/PoseChangeBackendModule.hpp"

#include <gtsam/nonlinear/ISAM2Params.h>

#include "dynosam_common/PointCloudProcess.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

void BatchPoseChangeInput::Builder::add(
    const SinglePoseChangeInput::ConstPtr& entry) {
  batch_.push_back(entry);
}
BatchPoseChangeInput::Ptr BatchPoseChangeInput::Builder::finalise() {
  auto output = BatchPoseChangeInput::Ptr(new BatchPoseChangeInput());
  output->batch_ = batch_;

  output->starting_frame_ = std::numeric_limits<FrameId>::max();
  output->ending_frame_ = std::numeric_limits<FrameId>::min();
  output->is_camera_involved_ = false;

  auto& latest_keyframes = output->latest_keyframes_;

  auto insertOrUpdateLatestKeyframes = [&latest_keyframes](ObjectId object_id,
                                                           FrameId frame_id) {
    if (!latest_keyframes.exists(object_id)) {
      latest_keyframes.insert2(object_id, frame_id);
    } else {
      // take max of previous entry and this frame
      FrameId previous_max_kf = latest_keyframes.at(object_id);
      latest_keyframes[object_id] = std::max(frame_id, previous_max_kf);
    }
  };

  std::unordered_set<ObjectId> objects_set;
  for (const auto& entry : output->batch_) {
    FrameId frame_id = entry->frame_id;
    const KeyframeInfo kf_info = entry->keyframe_info;

    output->starting_frame_ = std::min(output->starting_frame_, frame_id);
    output->ending_frame_ = std::max(output->ending_frame_, frame_id);

    if (kf_info.camera_keyframe) {
      output->is_camera_involved_ = true;
      insertOrUpdateLatestKeyframes(0, frame_id);
    }

    for (const auto& okf : kf_info.object_keyframes) {
      objects_set.insert(okf.object_id);
      insertOrUpdateLatestKeyframes(okf.object_id, frame_id);
    }
  }

  output->involved_objects_ = ObjectIds{objects_set.begin(), objects_set.end()};
  return output;
}

FrameId BatchPoseChangeInput::startingFrame() const { return starting_frame_; }

FrameId BatchPoseChangeInput::endingFrame() const { return ending_frame_; }

const ObjectIds& BatchPoseChangeInput::involvedObjects() const {
  return involved_objects_;
}

bool BatchPoseChangeInput::isCameraInvolved() const {
  return is_camera_involved_;
}

const gtsam::FastMap<ObjectId, FrameId>& BatchPoseChangeInput::latestKeyframes()
    const {
  return latest_keyframes_;
}

PoseChangeVIBackendModule::PoseChangeVIBackendModule(
    const BackendParams& params, Camera::Ptr camera,
    HybridFormulationKeyFrame::Ptr formulation,
    const SharedGroundTruth& shared_ground_truth)
    : Base(params, camera, shared_ground_truth),
      formulation_(CHECK_NOTNULL(formulation)) {
  hybrid_accessor_ =
      formulation_->derivedAccessor<HybridFormulationKeyFrameAccessor>();
  CHECK_NOTNULL(hybrid_accessor_);

  gtsam::ISAM2Params isam2_params;
  isam2_params.relinearizeThreshold = 0.01;
  isam2_params.relinearizeSkip = 1;
  // isam2_params.relinearizeSkip = FLAGS_regular_backend_relinearize_skip;
  isam2_params.keyFormatter = DynosamKeyFormatter;
  // isam2_params.enablePartialRelinearizationCheck = true;
  isam2_params.enablePartialRelinearizationCheck = false;
  isam2_params.evaluateNonlinearError = true;
  smoother_ = std::make_unique<gtsam::ISAM2>(isam2_params);

  smoother_interface_ = SmootherInterface(smoother_.get());
  smoother_interface_.setMaxExtraIterations(0);

  error_hooks_ = formulation_->getCustomErrorHooks();
  error_hooks_.identifier = "pc-backend";

  auto afs = std::make_shared<ApplyFunctionalSymbol>();
  afs->dynamicLandmark([&](TrackletId tracklet_id, const DynamicPointSymbol&) {
    auto map = formulation_->map();
    LOG(INFO) << "Debig ILS exception with landmark: " << tracklet_id;

    auto lmk = map->getLandmark(tracklet_id);
    CHECK_NOTNULL(lmk);
    LOG(INFO) << lmk->verboseInfo();
  });
  auto ils_debug_callback = [afs](gtsam::Key key) {
    LOG(ERROR) << DynosamKeyFormatter(key);
    afs->operator()(key);
  };
  error_hooks_.ils_debug_callbacks.push_back(ils_debug_callback);
}

PoseChangeVIBackendModule::~PoseChangeVIBackendModule() {}

DynoState::Ptr PoseChangeVIBackendModule::spinOnce(
    PoseChangeInput::ConstPtr input) {
  LOG(INFO) << "In PoseChangeVIBackendModule";

  // expect pipeline to provide a batch input
  BatchPoseChangeInput::ConstPtr batch_input =
      safeCast<PoseChangeInput, BatchPoseChangeInput>(input);
  CHECK_NOTNULL(batch_input);

  utils::ChronoTimingStats timer(formulation_->getFullyQualifiedName() +
                                 ".update_incremental");
  SharedModuleStates* shared_module_states =
      formulation_->map()->getSharedModuleStates();
  shared_module_states->is_backend_optimizing = true;

  // all values and factors accumulated from the batch input
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;
  // includes the camera and represents all involved keyframes
  gtsam::FastMap<ObjectId, KeyFrameIndexMap> new_kf_indices_per_object;
  // global mapping of {frame_id, object_id} -> keyframe index
  FrameKeyframeIndexMapping frame_to_frameIndex;

  prepareArgumentsForUpdate(batch_input, new_values, new_factors,
                            new_kf_indices_per_object, frame_to_frameIndex);

  gtsam::ISAM2Result result;
  if (!optimize(&result, new_values, new_factors, new_kf_indices_per_object)) {
    LOG(FATAL) << "Failed...";
  }

  // update the shared module state with the latest updated frames per object
  // this indicates up to which frame variables have been optimized for since
  // backend lags compared to the frontend which adds variables as each frame is
  // processed
  shared_module_states->updateLatestOptFramePerObject(
      batch_input->latestKeyframes());
  shared_module_states->is_backend_optimizing = false;

  LOG(INFO) << "ISAM2 result. Error before " << result.getErrorBefore()
            << " error after " << result.getErrorAfter();
  gtsam::Values optimised_values = smoother_interface_.calculateEstimate();
  // TODO: testing marginalisation - calling calculateEstimate causes an
  // internal theta = theta + delta
  //  in isam and this is currently breaking...
  //  get linearization point just returns the current theta without updating
  //  the current linearization gtsam::Values optimised_values =
  //  smoother_interface_.getLinearizationPoint();
  formulation_->updateTheta(optimised_values);

  DynoState::Ptr state = makeOutput();

  // alert frontend
  if (update_callback_) {
    PoseChangeUpdateComplete event;
    // TODO: actually look at and see which variables were marked as an object
    // may not be part of the input
    //  keyframes but variables did change due to other variables changing. This
    //  is likely to happen particularily with the camera!
    createPoseChangeUpdateComplete(batch_input, state, event);
    update_callback_(event);
  }

  return state;
}

void PoseChangeVIBackendModule::prepareArgumentsForUpdate(
    const BatchPoseChangeInput::ConstPtr& batch_input,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
    gtsam::FastMap<ObjectId, KeyFrameIndexMap>& new_kf_indices_per_object,
    FrameKeyframeIndexMapping& frame_to_frameIndex) {
  utils::ChronoTimingStats compute_kf_index_timer("pc_backend.compute_kf_index",
                                                  20);
  // compute the keyframe index for objects and camera
  // this is a little gross we have to do it every time but first check it
  // works and then optimize
  auto map = formulation_->map();
  const auto& all_ckfs = map->getCameraKeyFrames();
  const auto& all_okfs = map->getObjectKeyFrames();

  // index represents the relative positon of a keyframe (which is stored by
  // global frame k) to the other keyframes ie. if CKF's = {2, 4} then k=2 would
  // have a CKF index = 0 and k=4 has an CKF index 1 store objects and camera
  // (j=0) in the same data-structure and compure the index's independantly
  int kf_index = 0;
  for (const auto& ckf : all_ckfs) {
    const FrameObjectPair key{ckf->frameId(), 0};
    frame_to_frameIndex.insert2(key, kf_index);
    kf_index++;
  }

  gtsam::FastMap<ObjectId, int> per_object_count;
  for (const auto& [object_id, oks_j] : all_okfs) {
    per_object_count.insert2(object_id, 0);

    for (const auto& okf : oks_j) {
      int& count_j = per_object_count.at(object_id);
      const FrameObjectPair key{okf->frameId(), object_id};
      frame_to_frameIndex.insert2(key, count_j);
      count_j++;
    }
  }
  compute_kf_index_timer.stop();

  // PROBLEM: since each keyframe index (for camera/objects) has a different
  // scale the greatest one will dictact when the others are deleted ie if
  // camera frame is up to frame 25 and an object up to with less keyframes will
  // be have all variables delted!

  auto egofilterFunc = [](gtsam::Key key) -> bool {
    return gtsam::Symbol::ChrTest(kPoseSymbolChar)(key) ||
           gtsam::Symbol::ChrTest(kVelocitySymbolChar)(key) ||
           gtsam::Symbol::ChrTest(kImuBiasSymbolChar)(key);
  };

  // new key->kf indices for each object (including camera j=0)
  // that were involved in this batch update.
  gtsam::FastMap<ObjectId, KeyFrameIndexMap>& involved_kf_indices_per_object =
      new_kf_indices_per_object;

  const ObjectIds involved_objects = batch_input->involvedObjects();
  for (ObjectId object_id : involved_objects) {
    involved_kf_indices_per_object.insert2(object_id, KeyFrameIndexMap{});
  }

  if (batch_input->isCameraInvolved()) {
    involved_kf_indices_per_object.insert2(0, KeyFrameIndexMap{});
  }

  for (const auto& entry : *batch_input) {
    const KeyframeInfo& kf_info = entry->keyframe_info;
    const FrameId frame_id_k = entry->frame_id;

    // handle static only variables + factors
    const gtsam::Values& new_static_values = entry->new_static_fg_input.values;
    const gtsam::NonlinearFactorGraph& new_static_factors =
        entry->new_static_fg_input.factors;

    if (kf_info.camera_keyframe) {
      const FrameObjectPair frame_object_pair{frame_id_k, 0};
      // mark and ego-state poses to be marginalized based on their keyframe
      // index
      for (const auto& key_value : new_static_values) {
        const auto key = key_value.key;
        if (egofilterFunc(key)) {
          CHECK(frame_to_frameIndex.exists(frame_object_pair));
          int ckf_index = frame_to_frameIndex.at(frame_object_pair);
          involved_kf_indices_per_object.at(0).emplace(key, ckf_index);
          // timestamps[key] = static_cast<double>(ckf_index);
          // LOG(INFO) << "Adding static key " << DynosamKeyFormatter(key) << "
          // with kf index=" << ckf_index;
        }
      }
    }

    new_values.insert_or_assign(new_static_values);
    new_factors += new_static_factors;

    // do what we did in the frontend only here to delay the initalisation
    // of object motions
    formulation_->addObjects(frame_id_k, entry->kf_pose_change_infos);

    UpdateObservationParams update_params;
    update_params.enable_debug_info = true;
    update_params.do_backtrack = false;

    // generate new factors for dynamic objects based on latest measurements
    // and object keyframe states
    gtsam::Values new_dynamic_values;
    gtsam::NonlinearFactorGraph new_dynamic_factors;
    formulation_->updateDynamicObservations(frame_id_k, new_dynamic_values,
                                            new_dynamic_factors, update_params);

    // // // handle dynamic only variables + factors
    // const gtsam::Values& new_dynamic_values =
    //     entry->new_dynamic_fg_input.values;
    // const gtsam::NonlinearFactorGraph& new_dynamic_factors =
    //     entry->new_dynamic_fg_input.factors;

    // TODO: not actually even using the keyframe info
    for (const auto& key_value : new_dynamic_values) {
      const auto key = key_value.key;

      ObjectId object_id;
      FrameId frame_id;
      if (reconstructMotionInfo(key, object_id, frame_id)) {
        // if key is an object motion then frame_id will correspond with the to
        // motion for this value
        const FrameObjectPair frame_object_pair{frame_id, object_id};
        CHECK(frame_to_frameIndex.exists(frame_object_pair));
        int okf_index = frame_to_frameIndex.at(frame_object_pair);
        involved_kf_indices_per_object.at(object_id).emplace(key, okf_index);
        // LOG(INFO) << "Adding dynamic key " << DynosamKeyFormatter(key) << "
        // with kf index=" << okf_index;
      }
      // else if()
    }

    new_values.insert_or_assign(new_dynamic_values);
    new_factors += new_dynamic_factors;
  }
}

bool PoseChangeVIBackendModule::optimize(
    gtsam::ISAM2Result* result, const gtsam::Values& new_values,
    const gtsam::NonlinearFactorGraph& new_factors,
    const gtsam::FastMap<ObjectId, KeyFrameIndexMap>& new_kf_indices_per_object,
    const gtsam::ISAM2UpdateParams& update_params) {
  // all keys to be marginalized
  // gtsam::KeyVector marginalizable_keys;
  // // merged constrained keys
  // gtsam::FastMap<gtsam::Key, int> all_constrained_keys;

  // // Record so that we can delete these keys from the relevant
  // data-structures
  // // later
  // gtsam::FastMap<ObjectId, gtsam::KeyVector> marginalizable_keys_per_object;

  // // Do a pass over all new keyframe-indicies for each involved object
  // // (including camera)
  // // This is equivalent to parsing the timestamp (key->timestamp) to a
  // // regular fixed-lag-smoother
  // // where we update the temporal mapping and then get the set of keys to be
  // // marginalized
  // // in our case we do this independantly for each object/camera and
  // // construct a full set of
  // // keys to be marginalized for the entire problem
  // for(const auto& [object_id, new_frame_indices] : new_kf_indices_per_object)
  // {
  //   // also create boookeping KeyframeIndexBookkeeping for object if new
  //   if(!key_keyframe_indices_.exists(object_id)) {
  //     key_keyframe_indices_.insert2(object_id,
  //     KeyframeIndexBookkeeping(fixed_lag_));
  //   }

  //   //TODO: for testing
  //   // dont marginalize camera poses
  //   // since objects may be involved with the camera poses at non-CKF's
  //   // arguably we can probably get away with not marginalising camera stuff
  //   if(object_id == 0) {
  //     continue;
  //   }

  //   KeyframeIndexBookkeeping& lag_bk_j = key_keyframe_indices_.at(object_id);
  //   LOG(INFO) << "updating key->keframe indices for j=" << object_id;

  //   std::cout << "New frame indices ";
  //   for (const auto& [key, index] : new_frame_indices) {
  //     std::cout << DynosamKeyFormatter(key) << ":" << index << " ";
  //   }
  //   std::cout << std::endl;

  //   lag_bk_j.updateKeyFrameIndexMap(new_frame_indices);

  //   int current_keyframe_index = lag_bk_j.getCurrentFrameIndex();
  //   LOG(INFO) << "current KF index for j=" << object_id << " is " <<
  //   current_keyframe_index;

  //   gtsam::KeyVector marginalizable_keys_j =
  //     lag_bk_j.findMarginalizableKeys();
  //   marginalizable_keys_per_object[object_id] = marginalizable_keys_j;

  //   // create ordering for these keys and merge to the full data-structure
  //   gtsam::FastMap<gtsam::Key, int> constrained_keys_j;
  //   // Force iSAM2 to put the marginalizable variables at the beginning
  //   lag_bk_j.createOrderingConstraints(marginalizable_keys_j,
  //   constrained_keys_j);

  //   // NOTE existing keys are not overwittten!
  //   all_constrained_keys.merge(constrained_keys_j);

  //   marginalizable_keys.insert(marginalizable_keys.end(),
  //     marginalizable_keys_j.begin(), marginalizable_keys_j.end());

  //   std::cout << "Gets to marginalize due to filter j=" << object_id;
  //   for (const auto& key : marginalizable_keys_j) {
  //     std::cout << DynosamKeyFormatter(key) << " ";
  //   }
  //   std::cout << std::endl;

  // }

  // std::cout << "Constrained keys ";
  //   for (const auto& [key, order] : all_constrained_keys) {
  //     std::cout << DynosamKeyFormatter(key) << ": " << order << " ";
  //   }
  //   std::cout << std::endl;

  // std::unordered_set<gtsam::Key> additionalKeys =
  //     BayesTreeMarginalizationHelper<
  //         gtsam::ISAM2>::gatherAdditionalKeysToReEliminate(*smoother_,
  //                                                         marginalizable_keys);

  // gtsam::KeyList additionalMarkedKeys(additionalKeys.begin(),
  //                                     additionalKeys.end());

  gtsam::ISAM2UpdateParams mutable_update_params = update_params;
  // if (!mutable_update_params.extraReelimKeys) {
  //   mutable_update_params.extraReelimKeys = gtsam::KeyList{};
  // }
  // mutable_update_params.extraReelimKeys->insert(
  //     mutable_update_params.extraReelimKeys->begin(),
  //     additionalMarkedKeys.begin(), additionalMarkedKeys.end());

  // if (all_constrained_keys.size() > 0) {
  //   mutable_update_params.constrainedKeys.emplace(all_constrained_keys);
  // }

  using SmootherT = SmootherInterface::Smoother;
  using ArgumentsT = SmootherInterface::UpdateArguments;

  bool is_smoother_ok = smoother_interface_.optimize(
      result,
      [&](const SmootherT&, ArgumentsT& update_arguments) {
        update_arguments.new_values = new_values;
        update_arguments.new_factors = new_factors;
        update_arguments.update_params = mutable_update_params;
      },
      error_hooks_);

  // // // // Marginalize out any needed variables
  // if (marginalizable_keys.size() > 0) {
  //   gtsam::FastList<gtsam::Key> leafKeys(marginalizable_keys.begin(),
  //                                        marginalizable_keys.end());
  //   // utils::ChronoTimingStats marginalize_timer(
  //   //     logger_prefix_ + ".marginalize_leaves", 10);
  //   smoother_->marginalizeLeaves(leafKeys);
  // }

  // for(const auto& [object_id, marginalizable_keys_j] :
  // marginalizable_keys_per_object) {
  //   KeyframeIndexBookkeeping& lag_bk_j = key_keyframe_indices_.at(object_id);
  //   lag_bk_j.eraseKeyFrameIndexMap(marginalizable_keys_j);
  // }

  return is_smoother_ok;
}

void PoseChangeVIBackendModule::createPoseChangeUpdateComplete(
    const BatchPoseChangeInput::ConstPtr& batch_input,
    const DynoState::Ptr& state, PoseChangeUpdateComplete& event) const {
  event.starting_frame_id = batch_input->startingFrame();
  event.ending_frame_id = batch_input->endingFrame();

  for (const auto& entry : *batch_input) {
    event.keyframe_infos.insert2(entry->frame_id, entry->keyframe_info);
  }

  auto hybrid_accessor =
      formulation_->derivedAccessor<HybridFormulationKeyFrameAccessor>();

  // should be up to lates ckd
  event.camera.trajectory = state->camera_trajectory;
  // only send an update with the involved objects
  const ObjectIds involved_objects = batch_input->involvedObjects();
  const auto object_points_m_L =
      formulation_->getObjectPoints(involved_objects);

  // motion is not up to latest optimized variable!
  const auto& object_trajectories = state->object_trajectories;

  for (ObjectId object_id : involved_objects) {
    PoseChangeUpdateComplete::Object object;
    object.trajectory = object_trajectories.at(object_id);
    object.points_m_L = object_points_m_L.at(object_id);
    event.objects.insert2(object_id, object);
  };
}

void PoseChangeVIBackendModule::registerUpdateCallback(
    const PoseChangeUpdateCompleteCallback& callback) {
  update_callback_ = callback;
}

void PoseChangeVIBackendModule::KeyframeIndexBookkeeping::
    updateKeyFrameIndexMap(const KeyFrameIndexMap& new_frames) {
  // Loop through each key and add/update it in the map
  for (const auto& key_frameid : new_frames) {
    // Check to see if this key already exists in the database
    KeyFrameIndexMap::iterator key_iter =
        key_kf_index_map_.find(key_frameid.first);

    // If the key already exists
    if (key_iter != key_kf_index_map_.end()) {
      // Find the entry in the FrameIndex-Key database
      std::pair<FrameIndexKeyMap::iterator, FrameIndexKeyMap::iterator> range =
          kf_index_key_map_.equal_range(key_iter->second);
      FrameIndexKeyMap::iterator frame_id_iter = range.first;
      while (frame_id_iter->second != key_frameid.first) {
        ++frame_id_iter;
      }
      // remove the entry in the FrameIndex-Key database
      kf_index_key_map_.erase(frame_id_iter);
      // insert an entry at the new time
      kf_index_key_map_.insert(
          FrameIndexKeyMap::value_type(key_frameid.second, key_frameid.first));
      // update the Key-FrameIndex database
      key_iter->second = key_frameid.second;
    } else {
      // Add the Key-FrameIndex database
      key_kf_index_map_.insert(key_frameid);
      // Add the key to the FrameIndex-Key database
      kf_index_key_map_.insert(
          FrameIndexKeyMap::value_type(key_frameid.second, key_frameid.first));
    }
  }
}

void PoseChangeVIBackendModule::KeyframeIndexBookkeeping::eraseKeyFrameIndexMap(
    const gtsam::KeyVector& keys) {
  for (gtsam::Key key : keys) {
    // Erase the key from the index->Key map
    int frame_index = key_kf_index_map_.at(key);

    FrameIndexKeyMap::iterator iter =
        kf_index_key_map_.lower_bound(frame_index);
    while (iter != kf_index_key_map_.end() && iter->first == frame_index) {
      if (iter->second == key) {
        kf_index_key_map_.erase(iter++);
      } else {
        ++iter;
      }
    }
    // Erase the key from the Key->FrameId map
    key_kf_index_map_.erase(key);
  }
}

int PoseChangeVIBackendModule::KeyframeIndexBookkeeping::getCurrentFrameIndex()
    const {
  if (kf_index_key_map_.size() > 0) {
    return kf_index_key_map_.rbegin()->first;
  } else {
    return -std::numeric_limits<int>::max();
  }
}

gtsam::KeyVector
PoseChangeVIBackendModule::KeyframeIndexBookkeeping::findKeysBefore(
    int kf_index) const {
  gtsam::KeyVector keys;
  FrameIndexKeyMap::const_iterator end =
      kf_index_key_map_.lower_bound(kf_index);
  for (FrameIndexKeyMap::const_iterator iter = kf_index_key_map_.begin();
       iter != end; ++iter) {
    keys.push_back(iter->second);
  }
  return keys;
}

gtsam::KeyVector
PoseChangeVIBackendModule::KeyframeIndexBookkeeping::findKeysAfter(
    int kf_index) const {
  gtsam::KeyVector keys;
  FrameIndexKeyMap::const_iterator end =
      kf_index_key_map_.upper_bound(kf_index);
  for (FrameIndexKeyMap::const_iterator iter = kf_index_key_map_.begin();
       iter != end; ++iter) {
    keys.push_back(iter->second);
  }
  return keys;
}

gtsam::KeyVector
PoseChangeVIBackendModule::KeyframeIndexBookkeeping::findMarginalizableKeys()
    const {
  return findKeysBefore(getCurrentFrameIndex() - lag_);
}

void PoseChangeVIBackendModule::KeyframeIndexBookkeeping::
    createOrderingConstraints(
        const gtsam::KeyVector& marginalizableKeys,
        gtsam::FastMap<gtsam::Key, int>& constrainedKeys) const {
  if (marginalizableKeys.size() > 0) {
    // Generate ordering constraints so that the marginalizable variables will
    // be eliminated first Set all variables to Group1
    for (const FrameIndexKeyMap::value_type& frameIndex_key :
         kf_index_key_map_) {
      constrainedKeys[frameIndex_key.second] = 1;
    }
    // Set marginalizable variables to Group0
    for (gtsam::Key key : marginalizableKeys) {
      constrainedKeys[key] = 0;
    }
  }
}

}  // namespace dyno
