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

#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include "MRBayesTree.h"

namespace gtsam {

struct MRISAM2Params {
  GaussianFactorGraph::Eliminate eliminate_function;
  double relinearization_threshold = 0.1;
  double delta_update_threshold = 0.1;
  double marginal_update_threshold = 0.1;
  bool show_details = false;
  bool marginal_change_use_gradient = true;

  MRISAM2Params(const GaussianFactorGraph::Eliminate _eliminate_function =
                    EliminateCholesky)
      : eliminate_function(_eliminate_function) {}
};

struct MRISAM2Result;

/**
 * @brief Implementation of the full multi-root ISAM2 algorithm for multi-robot
 * incremental nonlinear optimization.
 */
class MRISAM2 : public MRBayesTree<GaussianBayesTree, GaussianFactorGraph> {
 public:
  typedef MRBayesTree<GaussianBayesTree, GaussianFactorGraph> MRBT;

  MRISAM2() : MRBT(), params_(MRISAM2Params()) {}

  /** constructor
   * @param graph   nonlinear factor graph
   * @param values  values of variables
   * @param order   elimination order to create the tree
   * @param root_id this root id
   * @param other_root_keys_map   map from root_id to keys in other roots
   * @param params  mrisam2 params
   */
  MRISAM2(const NonlinearFactorGraph& graph, const Values& values,
          const Ordering& order, const RootID root_id,
          const RootKeySetMap& other_root_keys_map,
          const MRISAM2Params& params = MRISAM2Params())
      : MRISAM2(graph, *graph.linearize(values), values, order, root_id,
                other_root_keys_map, params) {}

  MRISAM2(const NonlinearFactorGraph& nonlinear_factors,
          const GaussianFactorGraph& linear_factors, const Values& values,
          const Ordering& order, const RootID root_id,
          const RootKeySetMap& other_root_keys_map, const MRISAM2Params& params)
      : MRBT(linear_factors, order, root_id, other_root_keys_map, false,
             params.eliminate_function),
        params_(params),
        nonlinear_factors_(nonlinear_factors),
        linear_factors_(linear_factors),
        variable_index_(VariableIndex(linear_factors)),
        theta_(values),
        delta_(values.zeroVectors()) {
    SharedClique root_clique = roots_.begin()->second;
    computeDelta(root_clique, root_clique, linear_factors_, variable_index_,
                 delta_);
  }

  /** mark cliques that has any of specified keys as frontal variables, or
   * include relinearized variables, and all cliques on the path from such
   * cliques to root
   * @param root_id         specified root
   * @param involved_keys   keys involved in new factors
   * @param relin_keys      keys to relinearize
   * @return set of marked cliques
   */
  CliqueSet markCliques(const SharedClique root_clique,
                        const KeySet& involved_keys,
                        const KeySet& relin_keys) const;

  /** remove top at corresponding root, and return the factor graph and orphans
   * @param root_id        specified root
   * @param top_cliques    the cliques to remove
   * @param top_factor_indices  (return) indices of factors in top
   * @param top_keys       (return) keys ONLY in top of tree
   * @return the edges cut (points toward orphans)
   */
  EdgeVector extractTop(const SharedClique root_clique,
                        const CliqueSet& top_cliques,
                        FactorIndices& top_factor_indices,
                        KeySet& top_keys) const;

  /** connect orphans to the new top
   * @param root_id         specified root
   * @param new_top_mrbt    newly created top part of tree
   * @param orphan_edges    old edges connecting to orphan subtrees
   */
  EdgeVector connectOrphans(const RootID root_id, MRBT& new_top_mrbt,
                            const EdgeVector& orphan_edges);

  static Key findMostRecentStateKey(const RootID root_id,
                                    const SharedClique& root_clique);

  /** recreate the top part of the tree by re-eliminating the factor graph
   * (Note: the marginals, conditionals and deltas for the newly created cliques
   * and edges are not computed)
   * @param root_id         specified root
   * @param old_top_cliques cliques in old top tree
   * @param top_graph       the new factors corresponding to the new top graph
   * @param orphan_edges    old edges connecting to orphan subtrees
   * @param new_factor_keys keys associated with newly added factors
   * @return new edges connected to orphans
   */
  EdgeVector recreateTop(const RootID root_id, CliqueSet& old_top_cliques,
                         const FactorIndices& top_factor_indices,
                         const EdgeVector& orphan_edges,
                         const KeySet& new_factor_keys,
                         MRISAM2Result& update_result);

  /** perform elimination in top part of the tree, set the marginals,
   * conditionals and deltas in top. (Note: for the boundary edges pointing
   * toward top, the marginals are not computed)
   * @param root_id         specified root
   * @param boundary_edges  new edges created to connect orphans (point in the
   * direction toward orphans)
   */
  void eliminateTop(const RootID root_id, const EdgeVector& boundary_edges);

  double marginalChangeByGradient(const SharedFactor& old_marginal,
                                  const SharedFactor& new_marginal) const;

  double marginalChangeByKLD(const SharedFactor& old_marginal,
                             const SharedFactor& new_marginal) const;

  /** compare two marginals by their KL Divergence */
  bool marginalChanged(const SharedFactor& old_marginal,
                       const SharedFactor& new_marginal) const;

  void propagateDeltaRecursive(const SharedEdge& edge,
                               MRISAM2Result& update_result);

  void propagateMarginalsRecursive(const SharedEdge& edge,
                                   MRISAM2Result& update_result);

  /** propagate marginals outside from the top tree
   * @param root_id         specified root
   * @param boundary_edges  new edges created to connect orphans (point in the
   * direction toward orphans)
   */
  void propagateMarginals(const RootID root_id,
                          const EdgeVector& boundary_edges,
                          MRISAM2Result& update_result);

  /** compute delta with a wildfire spread from the boundary edges */
  void propagateDeltas(const RootID root_id, const EdgeVector& boundary_edges,
                       MRISAM2Result& update_result);

  // VectorValues computeEstimates(const RootID root_id,
  //                               const CliqueSet& top_cliques);

  /** check the condition of inputs are correct: root id should be valid; new
   * values should be provided for all new appeared variables, while no other
   * variables are included */
  void updateRootCheck(const RootID root_id,
                       const NonlinearFactorGraph& new_factors,
                       const Values& new_theta);

  /** update at a root with new factors and new values
   * @param root_id       specified root
   * @param new_factors   new measurements gathered by the robot
   * @param new_theta     initial estimate for new variables
   * @param do_relinearization  perform relinearization in this update step
   */
  MRISAM2Result updateRoot(const RootID root_id,
                           const NonlinearFactorGraph& new_factors,
                           const Values& new_theta,
                           const bool do_relinearization = false);

  /** recursively find the keys that need relinearization
   * @param parent        parent clique
   * @param clique        current clique
   * @param relin_keys    (return) keys to be relinearized
   */
  void checkRelinearizationRecursive(const SharedClique& parent,
                                     const SharedClique& clique,
                                     KeySet& relin_keys) const;

  /** gather keys that need relinearization */
  KeySet gatherRelinearizeKeys(const SharedClique& root_clique) const;

  /** gather existing keys involved in new factors */
  KeySet gatherInvolvedKeys(const NonlinearFactorGraph& new_factors) const;

  /** relinearize factors in top */
  void relinearizeTop(const KeySet& top_keys,
                      const FactorIndices& top_factor_indices);

  void addVariables(const Values& new_theta);

  FactorIndices addFactors(const NonlinearFactorGraph& new_factors);

  const NonlinearFactorGraph& nonlinearFactors() { return nonlinear_factors_; }

  const GaussianFactorGraph& linearFactors() { return linear_factors_; }

  const VariableIndex& variableIndex() { return variable_index_; }

  const Values& theta() { return theta_; }

  const VectorValues& delta() { return delta_; }

  MRISAM2Params& params() { return params_; }

  Values calculateBestEstimate();

 protected:
  MRISAM2Params params_;

  NonlinearFactorGraph nonlinear_factors_;
  GaussianFactorGraph linear_factors_;
  VariableIndex variable_index_;
  Values theta_;
  VectorValues delta_;
};

struct MRISAM2Result {
  KeySet top_keys;
  FactorIndices top_factor_indices;
  KeySet relin_keys;
  KeySet involved_keys;
  size_t old_top_clique_size = 0;
  size_t new_top_clique_size = 0;
  size_t propagated_delta = 0;
  size_t propagated_marginal = 0;
  std::set<size_t> roots_in_top;
  std::vector<double> durations;
  size_t largest_clique_size;
  KeySet variables_reeliminated;
  MRISAM2::CliqueSet top_cliques;
};

}  // namespace gtsam
