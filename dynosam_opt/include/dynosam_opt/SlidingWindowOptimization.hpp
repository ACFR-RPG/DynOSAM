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

#pragma once

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "dynosam_common/Types.hpp"

namespace dyno {

struct SWOptimizationResult {
  bool optimized = false;
  gtsam::Values result;
  gtsam::NonlinearFactorGraph prior;
};

class SlidingWindowOptimization {
 public:
  DYNO_POINTER_TYPEDEFS(SlidingWindowOptimization)

  struct Params {
    FrameId window_size;
    FrameId overlap;
    bool filter_invalid_factors = true;
  };

  SlidingWindowOptimization(const Params& params);

  SWOptimizationResult update(const gtsam::NonlinearFactorGraph& new_factors,
                              const gtsam::Values& new_values,
                              FrameId frame_id);

 private:
  SWOptimizationResult optimizeWindow();

  bool isRecentKey(gtsam::Key key) const;

  // check if new factors connect to marginalized keys
  gtsam::NonlinearFactorGraph filterValidFactors(
      const gtsam::NonlinearFactorGraph& factors) const;

  static gtsam::GaussianFactorGraph CalculateMarginalFactors(
      const gtsam::GaussianFactorGraph& graph, const gtsam::KeyVector& keys,
      const gtsam::GaussianFactorGraph::Eliminate& eliminateFunction =
          gtsam::EliminatePreferCholesky);

  static gtsam::NonlinearFactorGraph CalculateMarginalFactors(
      const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& theta,
      const gtsam::KeyVector& keys,
      const gtsam::GaussianFactorGraph::Eliminate& eliminateFunction =
          gtsam::EliminatePreferCholesky);

 private:
  const Params params_;
  FrameId current_frame_ = 0;

  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values values_;
  gtsam::NonlinearFactorGraph priorFactors_;

  std::deque<FrameId> frame_window_;
  gtsam::FastMap<gtsam::Key, FrameId> key_frameid_map_;

  //! Keys that have been successfully marginalized
  gtsam::KeySet marginalized_keys_;
};

}  // namespace dyno
