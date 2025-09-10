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
#include <gtsam_unstable/nonlinear/FixedLagSmoother.h>

#include <queue>

namespace dyno {

class BatchFixedLagSmoother : public gtsam::FixedLagSmoother {
 public:
  /// Typedef for a shared pointer to an Incremental Fixed-Lag Smoother
  typedef std::shared_ptr<BatchFixedLagSmoother> shared_ptr;

  struct UpdateParams {
    std::map<gtsam::Key, double> timestamps;
    gtsam::FactorIndices factors_to_remove = gtsam::FactorIndices();
    gtsam::Values values_relin = gtsam::Values();
  };

  /**
   * Construct with parameters
   *
   * @param smootherLag The length of the smoother lag. Any variable older than
   * this amount will be marginalized out.
   * @param parameters The L-M optimization parameters
   * @param enforceConsistency A flag indicating if the optimizer should enforce
   * probabilistic consistency by maintaining the linearization point of all
   * variables involved in linearized/marginal factors at the edge of the
   * smoothing window.
   */
  BatchFixedLagSmoother(double smootherLag = 0.0,
                        const gtsam::LevenbergMarquardtParams& parameters =
                            gtsam::LevenbergMarquardtParams(),
                        bool enforceConsistency = true)
      : FixedLagSmoother(smootherLag),
        parameters_(parameters),
        enforceConsistency_(enforceConsistency) {}

  /** destructor */
  ~BatchFixedLagSmoother() override {}

  /** Print the factor for debugging and testing (implementing Testable) */
  void print(const std::string& s = "BatchFixedLagSmoother:\n",
             const gtsam::KeyFormatter& keyFormatter =
                 gtsam::DefaultKeyFormatter) const override;

  /** Check if two IncrementalFixedLagSmoother Objects are equal */
  bool equals(const gtsam::FixedLagSmoother& rhs,
              double tol = 1e-9) const override;

  /** Add new factors, updating the solution and relinearizing as needed. */
  Result update(const gtsam::NonlinearFactorGraph& newFactors =
                    gtsam::NonlinearFactorGraph(),
                const gtsam::Values& newTheta = gtsam::Values(),
                const KeyTimestampMap& timestamps = KeyTimestampMap(),
                const gtsam::FactorIndices& factorsToRemove =
                    gtsam::FactorIndices()) override;

  Result update(const gtsam::NonlinearFactorGraph& newFactors,
                const gtsam::Values& newTheta,
                const UpdateParams& update_params);

  /** Compute an estimate from the incomplete linear delta computed during the
   * last update. This delta is incomplete because it was not updated below
   * wildfire_threshold.  If only a single variable is needed, it is faster to
   * call calculateEstimate(const KEY&).
   */
  gtsam::Values calculateEstimate() const override {
    return theta_.retract(delta_);
  }

  /** Compute an estimate for a single variable using its incomplete linear
   * delta computed during the last update.  This is faster than calling the
   * no-argument version of calculateEstimate, which operates on all variables.
   * @param key
   * @return
   */
  template <class VALUE>
  VALUE calculateEstimate(gtsam::Key key) const {
    const gtsam::Vector delta = delta_.at(key);
    return gtsam::traits<VALUE>::Retract(theta_.at<VALUE>(key), delta);
  }

  /** read the current set of optimizer parameters */
  const gtsam::LevenbergMarquardtParams& params() const { return parameters_; }

  /** update the current set of optimizer parameters */
  gtsam::LevenbergMarquardtParams& params() { return parameters_; }

  /** Access the current set of factors */
  const gtsam::NonlinearFactorGraph& getFactors() const { return factors_; }

  /** Access the current linearization point */
  const gtsam::Values& getLinearizationPoint() const { return theta_; }

  /** Access the current ordering */
  const gtsam::Ordering& getOrdering() const { return ordering_; }

  /** Access the current set of deltas to the linearization point */
  const gtsam::VectorValues& getDelta() const { return delta_; }

  /// Calculate marginal covariance on given variable
  gtsam::Matrix marginalCovariance(gtsam::Key key) const;

  /// Marginalize specific keys from a linear graph.
  /// Does not check whether keys actually exist in graph.
  /// In that case will fail somewhere deep within elimination
  static gtsam::GaussianFactorGraph CalculateMarginalFactors(
      const gtsam::GaussianFactorGraph& graph, const gtsam::KeyVector& keys,
      const gtsam::GaussianFactorGraph::Eliminate& eliminateFunction =
          gtsam::EliminatePreferCholesky);

  /// Marginalize specific keys from a nonlinear graph, wrap in LinearContainers
  static gtsam::NonlinearFactorGraph CalculateMarginalFactors(
      const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& theta,
      const gtsam::KeyVector& keys,
      const gtsam::GaussianFactorGraph::Eliminate& eliminateFunction =
          gtsam::EliminatePreferCholesky);

 protected:
  /** A typedef defining an Key-Factor mapping **/
  typedef std::map<gtsam::Key, gtsam::FactorIndexSet> FactorIndex;

  /** The L-M optimization parameters **/
  gtsam::LevenbergMarquardtParams parameters_;

  /** A flag indicating if the optimizer should enforce probabilistic
   * consistency by maintaining the linearization point of all variables
   * involved in linearized/marginal factors at the edge of the smoothing
   * window. This idea is from ??? TODO: Look up paper reference **/
  bool enforceConsistency_;

  /** The nonlinear factors **/
  gtsam::NonlinearFactorGraph factors_;

  /** The current linearization point **/
  gtsam::Values theta_;

  /** The set of values involved in current linear factors. **/
  gtsam::Values linearValues_;

  /** The current ordering */
  gtsam::Ordering ordering_;

  /** The current set of linear deltas */
  gtsam::VectorValues delta_;

  /** The set of available factor graph slots. These occur because we are
   * constantly deleting factors, leaving holes. **/
  std::queue<size_t> availableSlots_;

  /** A cross-reference structure to allow efficient factor lookups by key **/
  FactorIndex factorIndex_;

  /** Augment the list of factors with a set of new factors */
  void insertFactors(const gtsam::NonlinearFactorGraph& newFactors);

  /** Remove factors from the list of factors by slot index */
  void removeFactors(const std::set<size_t>& deleteFactors);

  /** Erase any keys associated with timestamps before the provided time */
  void eraseKeys(const gtsam::KeyVector& keys);

  /** Use colamd to update into an efficient ordering */
  void reorder(const gtsam::KeyVector& marginalizeKeys = gtsam::KeyVector());

  /** Optimize the current graph using a modified version of L-M */
  Result optimize();

  /** Marginalize out selected variables */
  void marginalize(const gtsam::KeyVector& marginalizableKeys);

 private:
  /** Private methods for printing debug information */
  static void PrintKeySet(const std::set<gtsam::Key>& keys,
                          const std::string& label);
  static void PrintKeySet(const gtsam::KeySet& keys, const std::string& label);
  static void PrintSymbolicFactor(
      const gtsam::NonlinearFactor::shared_ptr& factor);
  static void PrintSymbolicFactor(
      const gtsam::GaussianFactor::shared_ptr& factor);
  static void PrintSymbolicGraph(const gtsam::NonlinearFactorGraph& graph,
                                 const std::string& label);
  static void PrintSymbolicGraph(const gtsam::GaussianFactorGraph& graph,
                                 const std::string& label);
};  // BatchFixedLagSmoother

}  // namespace dyno
