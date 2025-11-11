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

#include "dynosam_opt/SlidingWindowOptimization.hpp"

namespace dyno {

SlidingWindowOptimization::SlidingWindowOptimization(
    const SlidingWindowOptimization::Params& params)
    : params_(params) {
  CHECK(params_.filter_invalid_factors)
      << "Option must be true - dense marginaliation has not been implemented!";
}

SWOptimizationResult SlidingWindowOptimization::update(
    const gtsam::NonlinearFactorGraph& new_factors,
    const gtsam::Values& new_values, FrameId frame_id) {
  // initalise timestamps
  for (const auto& [key, _] : new_values) {
    key_frameid_map_[key] = frame_id;
  }

  current_frame_ = frame_id;

  // Add new data
  graph_.add(new_factors);
  values_.insert(new_values);
  frame_window_.push_back(frame_id);

  if (frame_window_.size() > params_.window_size) {
    LOG(INFO) << "Starting SW Opt at frame " << frame_id;
    return optimizeWindow();
  }
  return SWOptimizationResult{};
}

SWOptimizationResult SlidingWindowOptimization::optimizeWindow() {
  SWOptimizationResult output;

  // TODO: if params
  auto filtered_graph = filterValidFactors(graph_);
  filtered_graph.add(priorFactors_);

  gtsam::LevenbergMarquardtParams params;
  gtsam::LevenbergMarquardtOptimizer optimizer(filtered_graph, values_, params);
  gtsam::Values result = optimizer.optimize();

  // Retain only variables with recent timestamps
  gtsam::KeyVector retainedKeys;
  gtsam::KeySet retainedSet;
  gtsam::Values retainedValues;
  for (const auto& [key, val] : result) {
    if (isRecentKey(key)) {
      retainedKeys.push_back(key);
      retainedSet.insert(key);
      retainedValues.insert(key, val);
    }
  }

  // Identify keys to marginalize = all - retained
  gtsam::KeySet allKeys;
  for (const auto& [key, _] : result) {
    allKeys.insert(key);
  }

  std::stringstream ss;
  ss << "Keys to marginalize: ";
  gtsam::KeyVector keysToMarginalize;
  for (const auto& key : allKeys) {
    if (!retainedSet.exists(key)) {
      keysToMarginalize.push_back(key);
      ss << key;
    }
  }

  LOG(INFO) << ss.str();

  // Linearize and marginalize old variables
  priorFactors_ =
      CalculateMarginalFactors(filtered_graph, result, keysToMarginalize);

  marginalized_keys_.insert(keysToMarginalize.begin(), keysToMarginalize.end());

  // Drop old times
  while (frame_window_.size() > params_.overlap) {
    frame_window_.pop_front();
  }

  graph_ = gtsam::NonlinearFactorGraph();
  values_ = retainedValues;

  output.optimized = true;
  output.result = result;
  output.prior = priorFactors_;
  return output;
}

bool SlidingWindowOptimization::isRecentKey(gtsam::Key key) const {
  if (key_frameid_map_.exists(key)) {
    return key_frameid_map_.at(key) > (current_frame_ - params_.overlap);
  }
  return false;
}

gtsam::NonlinearFactorGraph SlidingWindowOptimization::filterValidFactors(
    const gtsam::NonlinearFactorGraph& factors) const {
  gtsam::NonlinearFactorGraph valid_factors;

  for (const auto& factor : factors) {
    if (!factor) continue;

    bool valid = true;
    for (gtsam::Key key : factor->keys()) {
      if (marginalized_keys_.exists(key)) {
        valid = false;
        LOG(WARNING) << "Discarding factor involving marginalized key "
                     << gtsam::DefaultKeyFormatter(key);
        break;
      }
    }

    if (valid) {
      valid_factors.add(factor);
    }
  }

  return valid_factors;
}

gtsam::GaussianFactorGraph SlidingWindowOptimization::CalculateMarginalFactors(
    const gtsam::GaussianFactorGraph& graph, const gtsam::KeyVector& keys,
    const gtsam::GaussianFactorGraph::Eliminate& eliminateFunction) {
  if (keys.size() == 0) {
    // There are no keys to marginalize. Simply return the input factors
    return graph;
  } else {
    // .first is the eliminated Bayes tree, while .second is the remaining
    // factor graph
    return *graph.eliminatePartialMultifrontal(keys, eliminateFunction).second;
  }
}

gtsam::NonlinearFactorGraph SlidingWindowOptimization::CalculateMarginalFactors(
    const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& theta,
    const gtsam::KeyVector& keys,
    const gtsam::GaussianFactorGraph::Eliminate& eliminateFunction) {
  if (keys.size() == 0) {
    // There are no keys to marginalize. Simply return the input factors
    return graph;
  } else {
    // Create the linear factor graph
    const auto linearFactorGraph = graph.linearize(theta);

    const auto marginalLinearFactors =
        CalculateMarginalFactors(*linearFactorGraph, keys, eliminateFunction);

    // Wrap in nonlinear container factors
    return gtsam::LinearContainerFactor::ConvertLinearGraph(
        marginalLinearFactors, theta);
  }
}

}  // namespace dyno
