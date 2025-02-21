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

#include "dynosam/backend/rgbd/impl/DecoupledObjectSAM.hpp"

#include "dynosam/utils/TimingStats.hpp"

namespace dyno {

DecoupledObjectSAM::DecoupledObjectSAM(
    const Params& params, ObjectId object_id, const NoiseModels& noise_models,
    const FormulationHooks& formulation_hooks)
    : params_(params),
      object_id_(object_id),
      map_(Map::create()),
      expected_style_(MotionRepresentationStyle::F2F) {
  smoother_ = std::make_shared<gtsam::ISAM2>(params_.isam);

  FormulationParams formulation_params;
  formulation_params.suffix = "object_" + std::to_string(object_id);
  // HACK for now so that we get object motions at every frame!!!?
  formulation_params.min_dynamic_observations = 1u;

  decoupled_formulation_ =
      std::make_shared<keyframe_object_centric::DecoupledFormulation>(
          formulation_params, map_, noise_models, formulation_hooks);
  accessor_ = decoupled_formulation_->accessorFromTheta();
}

Motion3ReferenceFrame DecoupledObjectSAM::getFrame2FrameMotion(
    FrameId frame_id) const {
  // this is in the form of our accessor
  StateQuery<Motion3ReferenceFrame> H_W_km1_k =
      accessor_->getObjectMotionReferenceFrame(frame_id, object_id_);
  CHECK(H_W_km1_k);
  CHECK(H_W_km1_k->style() == MotionRepresentationStyle::F2F);
  CHECK(H_W_km1_k->origin() == ReferenceFrame::GLOBAL);
  CHECK(H_W_km1_k->to() == frame_id);

  return H_W_km1_k.get();
}

// currently no way of checking (with the object) the type of motion we have!!
Motion3ReferenceFrame DecoupledObjectSAM::getKeyFramedMotion(
    FrameId frame_id) const {
  Motion3ReferenceFrame H_W_s0_k =
      decoupled_formulation_->getEstimatedMotion(object_id_, frame_id);
  CHECK(H_W_s0_k.style() == MotionRepresentationStyle::KF);
  CHECK(H_W_s0_k.origin() == ReferenceFrame::GLOBAL);
  CHECK(H_W_s0_k.to() == frame_id);
  return H_W_s0_k;
}

void DecoupledObjectSAM::updateFormulation(
    FrameId frame_k, gtsam::NonlinearFactorGraph& new_factors,
    gtsam::Values& new_values) {
  // no need to update and static or odometry things ;)
  UpdateObservationParams update_params;
  update_params.do_backtrack = false;
  update_params.enable_debug_info = true;
  VLOG(10) << "DecoupledObjectSAM: Starting formulation update k=" << frame_k
           << " j= " << object_id_;

  GraphUpdateResult graph_update;
  decoupled_formulation_->updateDynamicObservations(frame_k, graph_update,
                                                    update_params);

  new_values = graph_update.values();
  new_factors = graph_update.factors();
}

bool DecoupledObjectSAM::updateSmoother(FrameId frame_k) {
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  updateFormulation(frame_k, new_factors, new_values);

  // do first optimisation
  dyno::utils::TimingStatsCollector timer(
      "decoupled_object_sam.optimize." +
      decoupled_formulation_->getFullyQualifiedName());
  bool is_smoother_ok = optimize(&result_, new_factors, new_values);

  if (is_smoother_ok) {
    // use dummy isam result when running optimize without new values/factors
    // as we want to use the result to determine which values were
    // changed/marked
    // TODO: maybe we actually need to append results together?
    static gtsam::ISAM2Result dummy_result;
    const auto& max_extra_iterations =
        static_cast<size_t>(params_.num_optimzie);
    VLOG(30) << "Doing extra iteration nr: " << max_extra_iterations;
    for (size_t n_iter = 0; n_iter < max_extra_iterations && is_smoother_ok;
         ++n_iter) {
      is_smoother_ok = optimize(&dummy_result);
    }
  }

  VLOG(5) << "DecoupledObjectSAM: update complete k=" << frame_k
          << " j= " << object_id_
          << "  error before: " << result_.errorBefore.value_or(NaN)
          << " error after: " << result_.errorAfter.value_or(NaN);
  return is_smoother_ok;
}

bool DecoupledObjectSAM::optimize(
    gtsam::ISAM2Result* result, const gtsam::NonlinearFactorGraph& new_factors,
    const gtsam::Values& new_values, const ISAM2UpdateParams& update_params) {
  CHECK_NOTNULL(result);
  CHECK(smoother_);

  try {
    *result = smoother_->update(new_factors, new_values, update_params);
  } catch (gtsam::IndeterminantLinearSystemException& e) {
    LOG(FATAL) << "gtsam::IndeterminantLinearSystemException with variable "
               << DynoLikeKeyFormatter(e.nearbyVariable());
  }
  return true;
}

void DecoupledObjectSAM::updateStates() {
  gtsam::Values previous_estimate = this->getEstimate();
  gtsam::Values estimate = smoother_->calculateEstimate();

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

  LOG(INFO) << "Motion change at frames "
            << container_to_string(motions_changed) << " for j=" << object_id_;

  decoupled_formulation_->updateTheta(estimate);
}

}  // namespace dyno
