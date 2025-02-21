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

#include "MRISAM2.h"

#include <gtsam/inference/LabeledSymbol.h>
#include <time.h>

#include <queue>

#include "dynosam/backend/BackendDefinitions.hpp"

namespace gtsam {

/* ************************************************************************* */
MRISAM2::CliqueSet MRISAM2::markCliques(const SharedClique root_clique,
                                        const KeySet& affected_keys,
                                        const KeySet& relin_keys) const {
  CliqueSet top_cliques;
  if (affected_keys.empty() && relin_keys.empty()) {
    return top_cliques;
  }

  KeySet affected_keys_remain = affected_keys;
  KeySet relin_keys_remain = relin_keys;

  // perform bfs until all keys are found (TODO: maintain a variable-clique map
  // to avoid the search)
  std::queue<std::pair<SharedClique, SharedClique>> bfs;
  bfs.push(std::make_pair(root_clique, root_clique));
  std::map<SharedClique, SharedClique> parent_map;  // for tracing the path

  while (!bfs.empty()) {
    SharedClique current_clique;
    SharedClique parent_clique;
    boost::tie(current_clique, parent_clique) = bfs.front();
    bfs.pop();
    parent_map[current_clique] = parent_clique;

    KeySet frontal_keys;
    if (current_clique == root_clique) {
      frontal_keys = current_clique->allKeys();
    } else {
      frontal_keys = current_clique->parentEdge(parent_clique)->frontalKeys();
    }
    bool has_affected = false;
    bool has_relin = false;
    // check if any of the frontal variables is in affected keys
    for (Key key : frontal_keys) {
      if (affected_keys_remain.exists(key)) {
        has_affected = true;
        affected_keys_remain.erase(key);
      }
    }
    // check if any variables in clique is in relin keys
    for (Key key : current_clique->allKeys()) {
      if (relin_keys.exists(key)) {
        has_relin = true;
        if (relin_keys_remain.exists(key)) {
          relin_keys_remain.erase(key);
        }
      }
    }
    if (has_affected || has_relin) {
      // add all cliques on the path
      top_cliques.insert(current_clique);
      SharedClique ancester_clique = parent_clique;
      while (top_cliques.find(ancester_clique) == top_cliques.end()) {
        top_cliques.insert(ancester_clique);
        ancester_clique = parent_map[ancester_clique];
      }
    }

    if (!affected_keys_remain.empty() || !relin_keys_remain.empty() ||
        has_relin) {
      CliqueVector neighbor_cliques = current_clique->neighborCliques();
      for (SharedClique neighbor_clique : neighbor_cliques) {
        if (neighbor_clique != parent_clique) {
          bfs.push(std::make_pair(neighbor_clique, current_clique));
        }
      }
    }
  }
  return top_cliques;
}

/* ************************************************************************* */
MRISAM2::EdgeVector MRISAM2::extractTop(const SharedClique root_clique,
                                        const CliqueSet& top_cliques,
                                        FactorIndices& top_factor_indices,
                                        KeySet& top_keys) const {
  EdgeVector orphan_edges;

  if (top_cliques.empty()) {
    return orphan_edges;
  }

  std::queue<std::pair<SharedClique, SharedClique>> bfs;
  bfs.push(std::make_pair(root_clique, root_clique));
  while (!bfs.empty()) {
    SharedClique current_clique;
    SharedClique parent_clique;
    boost::tie(current_clique, parent_clique) = bfs.front();
    bfs.pop();

    if (current_clique == root_clique) {
      for (FactorIndex i : root_clique->factorIndicesInClique(
               linear_factors_, variable_index_)) {
        top_factor_indices.push_back(i);
      }
    } else {
      SharedEdge edge = parent_clique->childEdge(current_clique);
      for (FactorIndex i :
           edge->elimFactorIndices(linear_factors_, variable_index_)) {
        top_factor_indices.push_back(i);
      }
    }
    top_keys.merge(current_clique->allKeys());

    CliqueVector neighbor_cliques = current_clique->neighborCliques();
    for (SharedClique neighbor_clique : neighbor_cliques) {
      if (neighbor_clique != parent_clique) {
        if (top_cliques.find(neighbor_clique) != top_cliques.end()) {
          bfs.push(std::make_pair(neighbor_clique, current_clique));
        } else {
          orphan_edges.push_back(current_clique->childEdge(neighbor_clique));
          top_keys.merge(Edge(neighbor_clique, current_clique).frontalKeys());
        }
      }
    }
  }

  for (const SharedEdge& edge : orphan_edges) {
    for (Key key : edge->separatorKeys()) {
      if (top_keys.exists(key)) {
        top_keys.erase(key);
      }
    }
  }
  return orphan_edges;
}

/* ************************************************************************* */
MRISAM2::EdgeVector MRISAM2::connectOrphans(const RootID root_id,
                                            MRBT& top_mrbt,
                                            const EdgeVector& orphan_edges) {
  // find the cliques to connect
  FastVector<KeySet> separator_keys_vec;
  for (const SharedEdge& edge : orphan_edges) {
    separator_keys_vec.push_back(edge->separatorKeys());
  }
  SharedClique root_clique = top_mrbt.roots().at(root_id);
  CliqueVector attached_cliques =
      top_mrbt.findCliquesByKeys(separator_keys_vec, root_clique);

  EdgeVector new_edges;
  // connect orphans
  for (size_t orphan_idx = 0; orphan_idx < orphan_edges.size(); orphan_idx++) {
    SharedEdge orphan_edge = orphan_edges[orphan_idx];
    SharedClique attached_clique = attached_cliques[orphan_idx];
    SharedClique orphan_clique = orphan_edge->childClique();

    bool dual_direction = orphan_edge->dualDirection();
    if (dual_direction) {
      orphan_clique->removeEdge(
          orphan_clique->childEdge(orphan_edge->parentClique()));
    }
    orphan_clique->removeEdge(orphan_edge);
    orphan_edge->setParent(attached_clique);
    orphan_clique->addEdge(orphan_edge);
    attached_clique->addEdge(orphan_edge);
    new_edges.push_back(orphan_edge);

    if (dual_direction) {
      // note: the marginals and conditionals of the edges remain to be computed
      SharedEdge orphan_edge_reverse =
          boost::make_shared<Edge>(orphan_clique, attached_clique);
      orphan_clique->addEdge(orphan_edge_reverse);
      attached_clique->addEdge(orphan_edge_reverse);
    }
  }
  return new_edges;
}

/* ************************************************************************* */
Key MRISAM2::findMostRecentStateKey(const RootID root_id,
                                    const SharedClique& root_clique) {
  int most_recent_index = -1;
  Key most_recent_key;
  for (const Key& key : root_clique->allKeys()) {
    //  LabeledSymbol symbol(key);
    //  if (symbol.chr()=='X' && symbol.label() == root_id + 97) {
    //    if ((int)symbol.index() > most_recent_index) {
    //      most_recent_index = symbol.index();
    //      most_recent_key = key;
    //    }
    //  }
    Symbol symbol(key);
    if (symbol.chr() == dyno::kPoseSymbolChar) {
      if ((int)symbol.index() > most_recent_index) {
        most_recent_index = symbol.index();
        most_recent_key = key;
      }
    }
  }
  if (most_recent_index == -1) {
    return *(root_clique->allKeys().rbegin());
  } else {
    return most_recent_key;
  }
}

/* ************************************************************************* */
MRISAM2::EdgeVector MRISAM2::recreateTop(
    const RootID root_id, CliqueSet& old_top_cliques,
    const FactorIndices& top_factor_indices, const EdgeVector& orphan_edges,
    const KeySet& new_factor_keys, MRISAM2Result& update_result) {
  // TODO: also update delta

  // check if any other roots is in old_top_cliques
  std::set<RootID> top_roots;
  RootKeySetMap other_top_root_keys_map;
  for (auto& it : roots_) {
    if (old_top_cliques.find(it.second) != old_top_cliques.end()) {
      RootID other_root_id = it.first;
      SharedClique other_root_clique = it.second;
      top_roots.insert(other_root_id);
      if (other_root_id != root_id) {
        KeySet root_keys;
        root_keys.insert(
            findMostRecentStateKey(other_root_id, other_root_clique));
        other_top_root_keys_map[other_root_id] = root_keys;
      }
    }
  }

  // any dual-directional edge cut should also correspond to a "root"
  int additional_root_id = *top_roots.rbegin();
  for (const SharedEdge& orphan_edge : orphan_edges) {
    if (orphan_edge->dualDirection()) {
      additional_root_id++;
      other_top_root_keys_map[additional_root_id] =
          orphan_edge->separatorKeys();
    }
  }

  // turn top_graph into a tree by elimination
  GaussianFactorGraph top_graph;
  for (FactorIndex i : top_factor_indices) {
    top_graph.push_back(linear_factors_.at(i));
  }
  for (const SharedEdge& orphan_edge : orphan_edges) {
    top_graph.push_back(orphan_edge->marginal());
  }
  VariableIndex top_var_index(top_graph);
  FastMap<Key, int> constraint_groups;
  int group = 1;
  for (Key key : new_factor_keys) {
    constraint_groups.insert(std::make_pair(key, group));
  }
  Ordering order =
      Ordering::ColamdConstrained(top_var_index, constraint_groups);
  // top_graph.print("top graph:");
  // order.print();

  MRBT top_tree(top_graph, order, root_id, other_top_root_keys_map, true);
  update_result.new_top_clique_size = top_tree.size();
  CliqueVector all_cliques = top_tree.allCliques();
  update_result.top_cliques = CliqueSet(all_cliques.begin(), all_cliques.end());
  // top_tree.print();

  // connect the orphan sub-trees, and update root cliques
  EdgeVector boundary_edges = connectOrphans(root_id, top_tree, orphan_edges);

  // re-assign the top roots
  for (RootID root_id : top_roots) {
    roots_[root_id] = top_tree.roots().at(root_id);
  }
  top_tree.disEngage();

  // remove the old top cliques to avoid momory leak with shared pointers
  for (auto& clique : old_top_cliques) {
    clique->disEngage();
  }

  return boundary_edges;
}

/* ************************************************************************* */
void MRISAM2::eliminateTop(const RootID root_id,
                           const EdgeVector& boundary_edges) {
  const SharedClique& root_clique = roots_.at(root_id);
  EdgeSet boundary_edges_out, boundary_edges_in;
  for (SharedEdge boundary_edge : boundary_edges) {
    boundary_edges_out.insert(boundary_edge);
    if (boundary_edge->dualDirection()) {
      SharedClique parent = boundary_edge->childClique();
      SharedClique child = boundary_edge->parentClique();
      boundary_edges_in.insert(parent->childEdge(child));
    }
  }

  eliminateNodeBottomUp(root_clique, root_clique, linear_factors_,
                        variable_index_, params_.eliminate_function,
                        boundary_edges_out);
  eliminateNodeTopDown(root_clique, root_clique, linear_factors_,
                       variable_index_, params_.eliminate_function,
                       boundary_edges_in);
  computeDelta(root_clique, root_clique, linear_factors_, variable_index_,
               delta_, -1, boundary_edges_out);
}

double MRISAM2::marginalChangeByGradient(
    const SharedFactor& old_marginal, const SharedFactor& new_marginal) const {
  VectorValues old_gradient;
  VectorValues new_gradient;
  KeyVector old_keys = old_marginal->keys();
  KeyVector new_keys = new_marginal->keys();

  // // TODO: check why there can be such a big factor size
  // if (old_keys.size() > 20 || new_keys.size() > 20) {
  //   PrintKeyVector(old_keys, "old_keys: ", MultiRobotKeyFormatter);
  //   PrintKeyVector(new_keys, "new_keys: ", MultiRobotKeyFormatter);
  // }

  for (Key key : old_keys) {
    old_gradient.insert(key, old_marginal->gradient(key, delta_));
  }
  for (Key key : new_keys) {
    new_gradient.insert(key, new_marginal->gradient(key, delta_));
  }

  for (Key old_key : old_keys) {
    if (!new_gradient.exists(old_key)) {
      new_gradient.insert(old_key, Vector::Zero(old_gradient[old_key].size()));
    }
  }
  for (Key new_key : new_keys) {
    if (!old_gradient.exists(new_key)) {
      old_gradient.insert(new_key, Vector::Zero(new_gradient[new_key].size()));
    }
  }

  VectorValues diff = new_gradient - old_gradient;
  double norm = diff.norm();

  if (params_.show_details) {
    std::cout << "\told gradient: " << old_gradient;
    std::cout << "\tnew gradient: " << new_gradient;
    std::cout << "\tdiff: " << norm << "\n";
  }

  if (norm > 1e10) {
    std::cout << "\told gradient: " << old_gradient;
    std::cout << "\tnew gradient: " << new_gradient;
    std::cout << "\tdiff: " << norm << "\n";
  }

  return norm;
}

double MRISAM2::marginalChangeByKLD(const SharedFactor& old_marginal,
                                    const SharedFactor& new_marginal) const {
  // TODO: check both factors should contain same keys

  // Matrix information_old = old_marginal->information();
  // Matrix information_new = new_marginal->information();
  // size_t dim = information_old.rows();

  // Vector mu_old;
  // Vector mu_new;
  // Vector mu_diff = mu_old - mu_new;

  // double tmp1 =
  // log(information_old.determinant()/information_new.determinant()); double
  // tmp2 = mu_diff.transpose() * information_new * mu_diff; double tmp3 =
  // (information_new * information_old.inverse()).trace();

  // return 0.5 * (tmp1 - dim + tmp2 + tmp3);
  return 0;
}

/* ************************************************************************* */
bool MRISAM2::marginalChanged(const SharedFactor& old_marginal,
                              const SharedFactor& new_marginal) const {
  if (!old_marginal) {
    return true;
  }

  double marginal_change =
      params_.marginal_change_use_gradient
          ? marginalChangeByGradient(old_marginal, new_marginal)
          : marginalChangeByKLD(old_marginal, new_marginal);

  // if (marginal_change >= params_.marginal_update_threshold) {
  //   std::cout << marginal_change << "\n";
  //   std::cout << params_.marginal_update_threshold << "\n";
  // }

  return marginal_change >= params_.marginal_update_threshold;
}

/* ************************************************************************* */
void MRISAM2::propagateMarginalsRecursive(const SharedEdge& edge,
                                          MRISAM2Result& update_result) {
  // set marginals for this edge

  GaussianFactorGraph gathered_factors;
  const SharedClique& child = edge->childClique();
  const SharedClique& clique = edge->parentClique();
  // gather information from all other branches
  for (const SharedEdge& gathered_edge : child->childEdges()) {
    if (gathered_edge->childClique() != clique) {
      gathered_factors.add(gathered_edge->marginal());
    }
  }
  gathered_factors.push_back(
      edge->elimFactors(linear_factors_, variable_index_));
  auto frontal_keys = edge->frontalKeys();
  Ordering ordering(frontal_keys.begin(), frontal_keys.end());
  auto elimination_result =
      params_.eliminate_function(gathered_factors, ordering);

  if (params_.show_details) {
    std::cout << "propagate marginals to edge " << edge->name() << "\n";
  }

  bool marginal_changed;
  if (params_.marginal_update_threshold > 0) {
    marginal_changed =
        marginalChanged(edge->marginal(), elimination_result.second);
  }
  update_result.propagated_marginal++;
  edge->setEliminationResult(elimination_result);

  // propagate to further edges TODO: check why it takes so long
  // if (params_.marginal_update_threshold <=0 || marginal_changed) {
  //   for (const SharedClique& parent : clique->parentCliques()) {
  //     if (parent != child) {
  //       propagateMarginalsRecursive(parent->childEdge(clique),
  //       update_result);
  //     }
  //   }
  // }
}

/* ************************************************************************* */
void MRISAM2::propagateDeltaRecursive(const SharedEdge& edge,
                                      MRISAM2Result& update_result) {
  bool values_changed = false;

  if (params_.show_details) {
    std::cout << "propagate delta on edge " << edge->name()
              << " set variables ";
    PrintKeySet(edge->frontalKeys(), "", MultiRobotKeyFormatter);
  }

  // set delta for frontal variables of the edge
  SharedClique clique = edge->childClique();
  SharedClique parent = edge->parentClique();
  if (params_.delta_update_threshold > 0) {
    KeySet frontal_keys_set = edge->frontalKeys();
    KeyVector frontal_keys(frontal_keys_set.begin(), frontal_keys_set.end());
    Vector original_values = delta_.vector(frontal_keys);
    VectorValues original_vector_values;
    if (params_.show_details) {
      for (Key key : frontal_keys) {
        original_vector_values.insert(key, delta_.at(key));
      }
    }
    delta_.update(edge->conditional()->solve(delta_));
    Vector diff = original_values - delta_.vector(frontal_keys);
    if (diff.lpNorm<Eigen::Infinity>() >= params_.delta_update_threshold) {
      values_changed = true;
    }
    if (params_.show_details) {
      for (Key key : frontal_keys) {
        std::cout << "\t" << MultiRobotKeyFormatter(key) << ": "
                  << original_vector_values.at(key) << " -> " << delta_.at(key)
                  << "\n";
      }
    }
  } else {
    delta_.update(edge->conditional()->solve(delta_));
  }
  update_result.propagated_delta++;

  // propagate to further edges
  if (params_.delta_update_threshold <= 0 || values_changed) {
    for (const SharedClique& child : clique->childCliques()) {
      if (child != parent) {
        propagateDeltaRecursive(clique->childEdge(child), update_result);
      }
    }
  }
}

/* ************************************************************************* */
void MRISAM2::propagateMarginals(const RootID root_id,
                                 const EdgeVector& boundary_edges,
                                 MRISAM2Result& update_result) {
  for (SharedEdge boundary_edge : boundary_edges) {
    if (boundary_edge->dualDirection()) {
      SharedClique clique = boundary_edge->childClique();
      SharedClique child = boundary_edge->parentClique();
      propagateMarginalsRecursive(clique->childEdge(child), update_result);
    }
  }
}

/* ************************************************************************* */
void MRISAM2::propagateDeltas(const RootID root_id,
                              const EdgeVector& boundary_edges,
                              MRISAM2Result& update_result) {
  for (const SharedEdge& boundary_edge : boundary_edges) {
    propagateDeltaRecursive(boundary_edge, update_result);
  }
}

/* ************************************************************************* */
/* ********************************* Nonlinear ***************************** */
/* ************************************************************************* */

/* ************************************************************************* */
void MRISAM2::checkRelinearizationRecursive(const SharedClique& parent,
                                            const SharedClique& clique,
                                            KeySet& relin_keys) const {
  KeySet conditionals;
  if (parent == clique) {
    conditionals = clique->allKeys();
  } else {
    conditionals = parent->childEdge(clique)->frontalKeys();
  }
  bool relinearize = false;
  for (Key var : conditionals) {
    double maxDelta = delta_[var].lpNorm<Eigen::Infinity>();
    // std::cout << "mrisam2 maxDelta: " << maxDelta << "\n";
    if (maxDelta >= params_.relinearization_threshold) {
      // std::cout << "mrisam2 relinearization check about threshold " <<
      // params_.relinearization_threshold << "\n";
      relin_keys.insert(var);
      relinearize = true;
    }
  }
  if (relinearize) {
    for (const SharedClique& child : clique->childCliques()) {
      if (child != parent) {
        checkRelinearizationRecursive(clique, child, relin_keys);
      }
    }
  }
}

/* ************************************************************************* */
KeySet MRISAM2::gatherRelinearizeKeys(const SharedClique& root_clique) const {
  KeySet relin_keys;
  checkRelinearizationRecursive(root_clique, root_clique, relin_keys);
  return relin_keys;
}

/* ************************************************************************* */
KeySet MRISAM2::gatherInvolvedKeys(
    const NonlinearFactorGraph& new_factors) const {
  KeySet new_factor_keys = new_factors.keys();
  KeySet involved_keys;
  for (Key key : new_factors.keys()) {
    if (theta_.exists(key)) {
      involved_keys.insert(key);
    }
  }
  return involved_keys;
}

/* ************************************************************************* */
void MRISAM2::updateRootCheck(const RootID root_id,
                              const NonlinearFactorGraph& new_factors,
                              const Values& new_theta) {
  if (roots_.find(root_id) == roots_.end()) {
    throw std::runtime_error("root does not exist at " +
                             std::to_string(root_id));
  }
  for (Key key : new_factors.keys()) {
    if (theta_.exists(key) && new_theta.exists(key)) {
      throw std::runtime_error("providing value that already exists for key " +
                               MultiRobotKeyFormatter(key));
    }
    if (!theta_.exists(key) && !new_theta.exists(key)) {
      throw std::runtime_error("new value not provided for key " +
                               MultiRobotKeyFormatter(key));
    }
  }
}

/* ************************************************************************* */
void MRISAM2::relinearizeTop(const KeySet& top_keys,
                             const FactorIndices& top_factor_indices) {
  for (Key key : top_keys) {
    theta_.update(key, *theta_.at(key).retract_(delta_.at(key)));
    delta_[key].setZero();
  }
  for (const FactorIndex& f_idx : top_factor_indices) {
    linear_factors_[f_idx] = nonlinear_factors_[f_idx]->linearize(theta_);
  }
}

/* ************************************************************************* */
void MRISAM2::addVariables(const Values& new_theta) {
  theta_.insert(new_theta);
  delta_.insert(new_theta.zeroVectors());
}

/* ************************************************************************* */
FactorIndices MRISAM2::addFactors(const NonlinearFactorGraph& new_factors) {
  nonlinear_factors_.add_factors(new_factors);
  variable_index_.augment(new_factors);
  return linear_factors_.add_factors(*new_factors.linearize(theta_));
}

/* ************************************************************************* */
MRISAM2Result MRISAM2::updateRoot(const RootID root_id,
                                  const NonlinearFactorGraph& new_factors,
                                  const Values& new_theta,
                                  const bool do_relinearization) {
  updateRootCheck(root_id, new_factors, new_theta);
  MRISAM2Result update_result;

  SharedClique root_clique = roots_.at(root_id);
  if (params_.show_details) {
    // std::cout << "\033[1;31mbold red text\033[0m\n";
    std::cout << "\033[1;31mupdate root at " << root_id << " with clique "
              << root_clique->name() << "\033[0m\n";
  }

  std::vector<clock_t> times;
  times.push_back(clock());

  // mark keys
  update_result.involved_keys = gatherInvolvedKeys(new_factors);
  update_result.relin_keys =
      do_relinearization ? gatherRelinearizeKeys(root_clique) : KeySet();

  if (params_.show_details) {
    PrintKeySet(update_result.involved_keys,
                "involved keys: ", MultiRobotKeyFormatter);
    PrintKeySet(update_result.relin_keys,
                "relin keys: ", MultiRobotKeyFormatter);
  }

  times.push_back(clock());

  // identify top (cliques invovled with theta above threshold + cliques with
  // separator variables invovled in new factors)
  CliqueSet top_cliques = markCliques(root_clique, update_result.involved_keys,
                                      update_result.relin_keys);
  update_result.old_top_clique_size = top_cliques.size();
  EdgeVector orphan_edges =
      extractTop(root_clique, top_cliques, update_result.top_factor_indices,
                 update_result.top_keys);

  update_result.variables_reeliminated = KeySet();
  for (const SharedClique& clique : top_cliques) {
    update_result.variables_reeliminated.merge(clique->allKeys());
  }
  for (Key key : new_theta.keys()) {
    update_result.variables_reeliminated.insert(key);
  }
  // PrintKeySet(update_result.variables_reeliminated, "mrisam2_relinkeys: ",
  // MultiRobotKeyFormatter);

  if (params_.show_details) {
    std::cout << "orphan edges: \n";
    for (const SharedEdge& edge : orphan_edges) {
      std::cout << "\t" << edge->name() << "\n";
    }
  }

  times.push_back(clock());

  // relinearize factors in top and update variable values
  if (do_relinearization) {
    relinearizeTop(update_result.top_keys, update_result.top_factor_indices);
    if (params_.show_details) {
      PrintKeySet(update_result.top_keys,
                  "relinearized variables: ", MultiRobotKeyFormatter);
    }
  }

  times.push_back(clock());

  // add new variables and factors
  addVariables(new_theta);
  FactorIndices new_factors_indices = addFactors(new_factors);
  update_result.top_factor_indices.insert(
      update_result.top_factor_indices.end(), new_factors_indices.begin(),
      new_factors_indices.end());

  // recreate top
  EdgeVector boundary_edges =
      recreateTop(root_id, top_cliques, update_result.top_factor_indices,
                  orphan_edges, new_factors.keys(), update_result);

  times.push_back(clock());

  eliminateTop(root_id, boundary_edges);
  if (params_.show_details) {
    std::cout << "new boundary edges: \n";
    for (const SharedEdge& edge : boundary_edges) {
      std::cout << "\t" << edge->name() << "\n";
    }
  }

  times.push_back(clock());

  // propagate marginals
  propagateMarginals(root_id, boundary_edges, update_result);

  times.push_back(clock());

  // propagate deltas, identify ones above threshold
  propagateDeltas(root_id, boundary_edges, update_result);

  times.push_back(clock());

  for (size_t i = 0; i < times.size() - 1; i++) {
    update_result.durations.push_back(times[i + 1] - times[i]);
  }

  return update_result;
}

/* ************************************************************************* */
Values MRISAM2::calculateBestEstimate() { return theta_.retract(delta_); }

}  // namespace gtsam
