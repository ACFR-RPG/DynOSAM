/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/backend/ParallelObjectISAM.hpp"

#include <pcl/io/pcd_io.h>  //for dynamic map IO

#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"

DEFINE_bool(save_per_frame_dynamic_cloud, false,
            "To save the local map of each object after every update! WARNING: "
            "may become memory intensive!!");

namespace dyno {

ParallelObjectISAM::ParallelObjectISAM(
    const Params& params, ObjectId object_id, const NoiseModels& noise_models,
    const Sensors& sensors, const FormulationHooks& formulation_hooks)
    : params_(params),
      object_id_(object_id),
      map_(Map::create()),
      expected_style_(MotionRepresentationStyle::F2F) {
  smoother_ = std::make_shared<gtsam::ISAM2>(params_.isam);

  FormulationParams formulation_params = params_.formulation;
  formulation_params.updater_suffix = "object_" + std::to_string(object_id);
  // HACK for now so that we get object motions at every frame!!!?
  formulation_params.min_dynamic_observations = 2u;

  decoupled_formulation_ = std::make_shared<HybridFormulation>(
      formulation_params, map_, noise_models, sensors, formulation_hooks);
  accessor_ = std::dynamic_pointer_cast<HybridAccessor>(
      decoupled_formulation_->accessorFromTheta());
  CHECK_NOTNULL(accessor_);

  // setup error hooks for the optimisation
  setupErrorHandlingHooks();
}

StateQuery<Motion3ReferenceFrame> ParallelObjectISAM::getFrame2FrameMotion(
    FrameId frame_id) const {
  // this is in the form of our accessor
  StateQuery<Motion3ReferenceFrame> H_W_km1_k =
      accessor_->getObjectMotionReferenceFrame(frame_id, object_id_);
  if (!H_W_km1_k) return H_W_km1_k;
  CHECK(H_W_km1_k->style() == MotionRepresentationStyle::F2F);
  CHECK(H_W_km1_k->origin() == ReferenceFrame::GLOBAL);
  CHECK(H_W_km1_k->to() == frame_id);

  return H_W_km1_k;
}

Motion3ReferenceFrame ParallelObjectISAM::getKeyFramedMotion(
    FrameId frame_id) const {
  StateQuery<Motion3ReferenceFrame> e_H_k_world =
      decoupled_formulation_->getEstimatedMotion(object_id_, frame_id);
  CHECK(e_H_k_world);
  CHECK(e_H_k_world->style() == MotionRepresentationStyle::KF);
  CHECK(e_H_k_world->origin() == ReferenceFrame::GLOBAL);
  CHECK(e_H_k_world->to() == frame_id);
  return e_H_k_world.get();
}

ObjectMotionMap ParallelObjectISAM::getFrame2FrameMotions() const {
  ObjectMotionMap motions;
  for (const FrameId& frame_id : map()->getFrameIds()) {
    auto f2f_motion = this->getFrame2FrameMotion(frame_id);
    if (f2f_motion) motions.insert22(object_id_, frame_id, f2f_motion.get());
  }
  return motions;
}

ObjectMotionMap ParallelObjectISAM::getKeyFramedMotions() const {
  ObjectMotionMap motions;
  for (const FrameId& frame_id : map()->getFrameIds()) {
    motions.insert22(object_id_, frame_id, this->getKeyFramedMotion(frame_id));
  }
  return motions;
}

StatusLandmarkVector ParallelObjectISAM::getDynamicLandmarks(
    FrameId frame_id) const {
  return accessor_->getDynamicLandmarkEstimates(frame_id, object_id_);
}

std::pair<FrameId, gtsam::Pose3> ParallelObjectISAM::insertNewKeyFrame(
    FrameId frame_id) {
  // this will clear internal factors and meta-data
  // TODO: what about values that exist between across keyframes - there will be
  // bugs in the Formulation as it currently cannot handle this!!
  // when we make a new keyframe we delete all old point variables (m) for this
  // object
  // this will mean we cannot get them even from the accessor as the
  // gtsam::Values used to store all variables is shared
  const auto new_key_frame =
      decoupled_formulation_->forceNewKeyFrame(frame_id, object_id_);

  // TrackletIds tracklets_in_key_frame
  // is_dynamic_tracklet_in_map_ shoulld be cleared!!!
  // smoother_.reset(new gtsam::ISAM2(params_.isam));
  // clear isam2 estimator
  return new_key_frame;
}

void ParallelObjectISAM::updateFormulation(
    FrameId frame_k, const Pose3Measurement& X_W_k,
    gtsam::NonlinearFactorGraph& new_factors, gtsam::Values& new_values) {
  // no need to update and static things ;)
  // TODO: hack to ensure we add the pose to the internal values on the first
  // run for the previous frame!!
  if (frame_k > 0) {
    auto frame_km1 = frame_k - 1u;

    if (!accessor_->exists(CameraPoseSymbol(frame_km1))) {
      VLOG(5) << "Previous camera pose does not exist!! k=" << frame_km1
              << "j=" << object_id_;
      Pose3Measurement T_W_cam_km1;
      CHECK(this->map()->hasInitialSensorPose(frame_km1, &T_W_cam_km1));

      const gtsam::Pose3& initial_X_W_km1 = T_W_cam_km1.measurement();
      CHECK(T_W_cam_km1.hasModel());
      const gtsam::SharedGaussian& uncertainty_X_W_km1 = T_W_cam_km1.model();

      decoupled_formulation_->addSensorPoseValue(initial_X_W_km1, frame_km1,
                                                 new_values);
      decoupled_formulation_->addSensorPosePriorFactor(
          initial_X_W_km1, uncertainty_X_W_km1, frame_km1, new_factors);
    }
  }

  const gtsam::Pose3& initial_X_W_k = X_W_k.measurement();
  CHECK(X_W_k.hasModel());
  const gtsam::SharedGaussian& uncertainty_X_W_k = X_W_k.model();

  decoupled_formulation_->addSensorPoseValue(initial_X_W_k, frame_k,
                                             new_values);
  decoupled_formulation_->addSensorPosePriorFactor(
      initial_X_W_k, uncertainty_X_W_k, frame_k, new_factors);

  UpdateObservationParams update_params;
  // TODO: setting this to true breaks the formulation as it does not understand
  // object path segments ie. the first seen frame of an object is not
  // necessarily what we want to backtrack too!!!
  update_params.do_backtrack = false;
  update_params.enable_debug_info = true;
  VLOG(10) << "ParallelObjectISAM: Starting formulation update k=" << frame_k
           << " j= " << object_id_;
  decoupled_formulation_->updateDynamicObservations(frame_k, new_values,
                                                    new_factors, update_params);
  VLOG(10) << "Done updateDynamicObservations";
  // TODO: use update result - if the object is not updated, should we remove
  // the frame node at k-1...?
}

bool ParallelObjectISAM::updateSmoother(FrameId frame_k,
                                        const Pose3Measurement& X_W_k) {
  // only clear result when we update the smoother...
  // result_ = Result();
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  updateFormulation(frame_k, X_W_k, new_factors, new_values);

  // do first optimisation
  utils::StatsCollector stats("parallel_object_sam.optimize." +
                              decoupled_formulation_->getFullyQualifiedName() +
                              " [ms]");
  auto tic = utils::Timer::tic();
  VLOG(5) << "ParallelObjectISAM: starting update k=" << frame_k
          << " j= " << object_id_;

  using SmootherInterface = IncrementalInterface<gtsam::ISAM2>;
  SmootherInterface smoother_interface(smoother_.get());

  const auto& max_extra_iterations = static_cast<size_t>(params_.num_optimzie);
  smoother_interface.setMaxExtraIterations(max_extra_iterations);

  bool is_smoother_ok = smoother_interface.optimize(
      &result_.isam_result,
      [&](const gtsam::ISAM2&,
          SmootherInterface::UpdateArguments& update_arguments) {
        update_arguments.new_values = new_values;
        update_arguments.new_factors = new_factors;
      },
      error_hooks_);

  auto toc = utils::Timer::toc<std::chrono::nanoseconds>(tic);
  int64_t milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(toc).count();
  stats.AddSample(static_cast<double>(milliseconds));

  result_.timing = milliseconds;
  result_.was_smoother_ok = is_smoother_ok;

  VLOG(5) << "ParallelObjectISAM: update complete k=" << frame_k
          << " j= " << object_id_
          << " error before: " << result_.isam_result.errorBefore.value_or(NaN)
          << " error after: " << result_.isam_result.errorAfter.value_or(NaN);
  return is_smoother_ok;
}

void ParallelObjectISAM::updateStates() {
  gtsam::Values previous_estimate = this->getEstimate();
  gtsam::Values estimate = smoother_->calculateEstimate();
  // gtsam::Values estimate = previous_estimate;

  // frame ids at which a motion had a large change
  // TODO: this does not include the new motions?
  FrameIds motions_changed;

  const double motion_delta = 0.1;  // TODO: ehhhhhh magic number for now!!

  const auto previous_motions = previous_estimate.extract<gtsam::Pose3>(
      gtsam::Symbol::ChrTest(kObjectMotionSymbolChar));
  for (const auto& [motion_key, previous_motion_estimate] : previous_motions) {
    if (estimate.exists(motion_key)) {
      gtsam::Pose3 new_motion_estimate = estimate.at<gtsam::Pose3>(motion_key);
      if (!new_motion_estimate.equals(previous_motion_estimate, motion_delta)) {
        ObjectId object_id;
        FrameId frame_id;
        CHECK(reconstructMotionInfo(motion_key, object_id, frame_id));
        CHECK_EQ(object_id, object_id_);

        motions_changed.push_back(frame_id);
      }
    }
  }

  // update statistics
  result_.motions_with_large_change = motions_changed;
  result_.large_motion_change_delta = motion_delta;
  std::tie(result_.max_clique_size, result_.average_clique_size) =
      factor_graph_tools::getCliqueSize(*smoother_);
  result_.num_factors = smoother_->getFactorsUnsafe().size();
  result_.num_variables = estimate.size();
  result_.motion_variable_status.clear();

  if (result_.was_smoother_ok) {
    // update with detailed results
    auto detailed_results = result_.isam_result.details();
    if (detailed_results) {
      // get all motion symbols
      const auto motion_symbols = estimate.extract<gtsam::Pose3>(
          gtsam::Symbol::ChrTest(kObjectMotionSymbolChar));
      for (const auto& [motion_key, _] : motion_symbols) {
        // TODO: unclear if variables not involved with anything will be in the
        // variable status at all!!
        if (detailed_results->variableStatus.exists(motion_key)) {
          ObjectId object_id;
          FrameId frame_id;
          CHECK(reconstructMotionInfo(motion_key, object_id, frame_id));

          // LOG(INFO) << "Motion key" << DynosamKeyFormatter(motion_key) << "
          // at k=" << frame_id << " key= "<< motion_key;
          CHECK_EQ(object_id, object_id_);
          // add variable status associated with the frame id of this motion
          result_.motion_variable_status.insert2(
              frame_id, detailed_results->variableStatus.at(motion_key));
        }
      }
    }

    // calculate what type of variables were involved in the update (marked
    // variables)
    size_t num_points_involved = 0;
    size_t num_motions_involved = 0;
    const gtsam::KeySet& marked_keys = result_.isam_result.markedKeys;

    ApplyFunctionalSymbol afs;
    afs.dynamicLandmark(
           [&num_points_involved](TrackletId, const DynamicPointSymbol&) {
             num_points_involved++;
           })
        .objectMotion([&num_motions_involved](FrameId, ObjectId,
                                              const gtsam::LabeledSymbol&) {
          num_motions_involved++;
        });
    for (const gtsam::Key& key : marked_keys) {
      afs(key);
    }

    result_.num_landmarks_marked = num_points_involved;
    result_.num_motions_marked = num_motions_involved;

    // TODO: only flag - no paramter struct!
    if (FLAGS_save_per_frame_dynamic_cloud) {
      VLOG(10) << "Saving per frame dynamic cloud "
               << info_string(result_.frame_id, object_id_);
      StatusLandmarkVector object_map =
          accessor_->getLocalDynamicLandmarkEstimates(object_id_);

      pcl::PointCloud<pcl::PointXYZRGB> object_map_cloud;
      convert(object_map, object_map_cloud);

      std::string path = dyno::getOutputFilePath(
          "object_map_k" + std::to_string(result_.frame_id) + "_j" +
          std::to_string(object_id_) + ".pcd");
      VLOG(10) << "Writing object map of size " << object_map_cloud.size()
               << " - " << path;
      pcl::io::savePCDFileASCII(path, object_map_cloud);
    }
  }

  VLOG(5) << "Motion change at frames " << container_to_string(motions_changed)
          << " for j=" << object_id_;

  decoupled_formulation_->updateTheta(estimate);
}

void ParallelObjectISAM::setupErrorHandlingHooks() {
  // setup error
  error_hooks_.handle_ils_exception = [](const gtsam::Values& current_values,
                                         gtsam::Key problematic_key) {
    ErrorHandlingHooks::HandleILSResult ils_handle_result;
    // a little gross that I need to set this up in this function
    gtsam::NonlinearFactorGraph& prior_factors = ils_handle_result.pior_factors;

    ApplyFunctionalSymbol afs;
    afs.cameraPose([&prior_factors, &current_values](FrameId,
                                                     const gtsam::Symbol& sym) {
         const gtsam::Key& key = sym;
         gtsam::Pose3 pose = current_values.at<gtsam::Pose3>(key);
         gtsam::Vector6 sigmas;
         sigmas.head<3>().setConstant(0.001);  // rotation
         sigmas.tail<3>().setConstant(0.01);   // translation
         gtsam::SharedNoiseModel noise =
             gtsam::noiseModel::Diagonal::Sigmas(sigmas);
         prior_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
             key, pose, noise);
       })
        .
        operator()(problematic_key);
    return ils_handle_result;
  };
}

void to_json(json& j, const ParallelObjectISAM::Result& result) {
  j["was_smoother_ok"] = result.was_smoother_ok;
  j["frame_id"] = result.frame_id;
  j["isam_result"] = result.isam_result;
  j["timing"] = result.timing;
  j["motions_with_large_change"] = result.motions_with_large_change;
  j["large_motion_change_delta"] = result.large_motion_change_delta;
  j["motion_variable_status"] = result.motion_variable_status;
  j["average_clique_size"] = result.average_clique_size;
  j["max_clique_size"] = result.max_clique_size;
  j["num_factors"] = result.num_factors;
  j["num_variables"] = result.num_variables;
  j["num_landmarks_marked"] = result.num_landmarks_marked;
  j["num_motions_marked"] = result.num_motions_marked;
  j["dynamic_map"] = result.dynamic_map;
}

}  // namespace dyno
