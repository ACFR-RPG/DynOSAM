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

#include "MRBayesTree.h"

#include <gtsam/inference/LabeledSymbol.h>
#include <gtsam/linear/GaussianEliminationTree.h>
#include <gtsam/linear/GaussianJunctionTree.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "dynosam/backend/BackendDefinitions.hpp"

//  #include "Utils.h"

std::string Quoted(const std::string& str) { return "\"" + str + "\""; }

typedef std::pair<std::string, std::string> AttributeType;

/**
 * @brief combine key value pairs into a dict in json format
 * @param[in] items         key value paris
 * @param[in] num_indents   number of indents for each item, -1 for no
 * indent and no newline
 * @return                  a string representing dict in json format
 */
std::string JsonDict(const std::vector<AttributeType>& items,
                     const int num_indents = 0) {
  std::string indents;
  std::string s;
  if (num_indents < 0) {
    indents = "";
    s = "{";
  } else {
    indents = "\n" + std::string(num_indents, ' ');
    s = std::string(num_indents, ' ') + "{";
  }
  for (auto& item : items) {
    s += indents + item.first + ":" + item.second + ",";
  }
  s.pop_back();  // remove the last comma
  s += indents + "}";
  return s;
}

/**
 * @brief combine items into a list in json format
 * @param[in] items         vector of items
 * @param[in] num_indents   number of indents for each item, -1 for no
 * indent and no newline
 * @return                  a string representing list in json format
 */
std::string JsonList(const std::vector<std::string>& items,
                     const int num_indents = 0) {
  std::string indents;
  std::string s;
  if (num_indents < 0) {
    indents = "";
    s = "[";
  } else {
    indents = "\n" + std::string(num_indents, ' ');
    s = std::string(num_indents, ' ') + "[";
  }
  for (auto& item : items) {
    s += indents + item + ",";
  }
  s.pop_back();  // remove the last comma
  s += indents + "]";
  return s;
}

namespace gtsam {

/* ************************************************************************* */
/* ****************************** Cons/Destructors ************************* */
/* ************************************************************************* */

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
MRBayesTree<BayesTreeType, FactorGraphType>::MRBayesTree(
    const JunctionTreeType& junctionTree, RootID root_id) {
  auto jt_root = junctionTree.roots()[0];
  roots_[root_id] = CopyFromJTByNode(jt_root);
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
typename MRBayesTree<BayesTreeType, FactorGraphType>::SharedClique
MRBayesTree<BayesTreeType, FactorGraphType>::CopyFromJTByNode(
    const typename JunctionTreeType::sharedCluster jt_node) {
  // create a MRBT clique
  SharedClique mrbt_node = boost::make_shared<Clique>();
  KeySet clique_keys = jt_node->factors.keys();

  // create child nodes
  for (auto jt_child : jt_node->children) {
    SharedClique mrbt_child = CopyFromJTByNode(jt_child);
    SharedEdge edge = boost::make_shared<Edge>(mrbt_node, mrbt_child);
    mrbt_child->addEdge(edge);
    mrbt_node->addEdge(edge);

    const KeySet& child_keys = mrbt_child->allKeys();
    KeySet frontal_keys(jt_child->orderedFrontalKeys.begin(),
                        jt_child->orderedFrontalKeys.end());
    KeySet separator_keys;
    std::set_difference(child_keys.begin(), child_keys.end(),
                        frontal_keys.begin(), frontal_keys.end(),
                        std::inserter(separator_keys, separator_keys.end()));
    clique_keys.merge(separator_keys);
  }

  // set keys for current clique
  mrbt_node->setKeys(clique_keys);
  return mrbt_node;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
MRBayesTree<BayesTreeType, FactorGraphType>::MRBayesTree(
    const JunctionTreeType& junctionTree, const FactorGraphType& graph,
    RootID root_id, const RootKeySetMap& other_root_keys_map,
    const bool symbolic, const Eliminate& elimination_function)
    : MRBayesTree(junctionTree, root_id) {
  SharedClique jt_root_clique = roots_.at(root_id);
  const KeySet& jt_root_keys = jt_root_clique->allKeys();

  // bool found_root = false;
  // for (auto& it : other_root_keys_map) {
  //   bool is_this_root = true;
  //   for (Key key : it.second) {
  //     if (jt_root_keys.find(key) == jt_root_keys.end()) {
  //       is_this_root = false;
  //       break;
  //     }
  //   }
  //   if (is_this_root) {
  //     found_root = true;
  //     RootID root_id = it.first;
  //     roots_[root_id] = jt_root_clique;
  //     break;
  //   }
  // }
  // if (!found_root) {
  //   throw std::runtime_error(
  //       "the root in junction tree does not agree with any specified root");
  // }

  // find the cliques for the roots
  FastVector<KeySet> root_keys_vec;
  FastVector<RootID> root_id_vec;
  for (auto it : other_root_keys_map) {
    root_id_vec.push_back(it.first);
    root_keys_vec.push_back(it.second);
  }
  const SharedClique& root_clique = roots_.begin()->second;
  CliqueVector root_cliques = findCliquesByKeys(root_keys_vec, root_clique);

  // add each root
  for (size_t i = 0; i < other_root_keys_map.size(); i++) {
    if (roots_.find(root_id_vec[i]) == roots_.end()) {
      addRoot(root_cliques[i], root_id_vec[i]);
    }
  }

  // compute marginals and conditionals
  if (!symbolic) {
    VariableIndex vi(graph);
    eliminate(graph, vi, elimination_function);
  }
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
MRBayesTree<BayesTreeType, FactorGraphType>::MRBayesTree(
    const FactorGraphType& graph, const Ordering& order, const RootID root_id,
    const RootKeySetMap& other_root_keys_map, const bool symbolic,
    const Eliminate& elimination_function)
    : MRBayesTree(GaussianJunctionTree(GaussianEliminationTree(graph, order)),
                  graph, root_id, other_root_keys_map, symbolic,
                  elimination_function) {}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
MRBayesTree<BayesTreeType, FactorGraphType>::~MRBayesTree() {
  for (const SharedClique& clique : allCliques()) {
    clique->disEngage();
  }
}

/* ************************************************************************* */
/* *********************************** Edge ******************************** */
/* ************************************************************************* */

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::Edge::setEliminationResult(
    const EliminationResult& elimination_result) {
  conditional_ = elimination_result.first;
  marginal_ = elimination_result.second;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
bool MRBayesTree<BayesTreeType, FactorGraphType>::Edge::dualDirection() const {
  for (auto& edge : parent_clique_->parentEdges()) {
    if (edge->parentClique() == childClique()) {
      return true;
    }
  }
  return false;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
FactorIndices
MRBayesTree<BayesTreeType, FactorGraphType>::Edge::elimFactorIndices(
    const FactorGraphType& graph, const VariableIndex& vi) const {
  std::set<FactorIndex> factor_indices_set;
  // must have frontal keys
  for (auto frontal_key : frontalKeys()) {
    for (auto factor_idx : vi[frontal_key]) {
      bool to_add = true;
      for (Key key : graph.at(factor_idx)->keys()) {
        // all keys should be either frontal or separator
        if (childClique()->allKeys().find(key) ==
            childClique()->allKeys().end()) {
          to_add = false;
          break;
        }
      }
      if (to_add) {
        factor_indices_set.insert(factor_idx);
      }
    }
  }
  return FactorIndices(factor_indices_set.begin(), factor_indices_set.end());
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
FactorGraphType MRBayesTree<BayesTreeType, FactorGraphType>::Edge::elimFactors(
    const FactorGraphType& graph, const VariableIndex& vi) const {
  FactorGraphType elim_graph;
  for (size_t i : elimFactorIndices(graph, vi)) {
    elim_graph.push_back(graph.at(i));
  }
  return elim_graph;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
KeySet MRBayesTree<BayesTreeType, FactorGraphType>::Edge::frontalKeys() const {
  KeySet frontal_keys;
  std::set_difference(
      child_clique_->allKeys().begin(), child_clique_->allKeys().end(),
      parent_clique_->allKeys().begin(), parent_clique_->allKeys().end(),
      std::inserter(frontal_keys, frontal_keys.end()));
  return frontal_keys;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
KeySet MRBayesTree<BayesTreeType, FactorGraphType>::Edge::separatorKeys()
    const {
  KeySet separator_keys;
  std::set_intersection(
      parent_clique_->allKeys().begin(), parent_clique_->allKeys().end(),
      child_clique_->allKeys().begin(), child_clique_->allKeys().end(),
      std::inserter(separator_keys, separator_keys.end()));
  return separator_keys;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
std::string MRBayesTree<BayesTreeType, FactorGraphType>::Edge::name(
    const KeyFormatter& keyFormatter) const {
  return parentClique()->name(keyFormatter) + "->" +
         childClique()->name(keyFormatter);
}

/* ************************************************************************* */
/* ********************************* Clique ******************************** */
/* ************************************************************************* */

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::Clique::addEdge(
    SharedEdge edge) {
  if (edge->parentClique().get() == this) {
    child_edges_.push_back(edge);
  } else if (edge->childClique().get() == this) {
    parent_edges_.push_back(edge);
  } else {
    throw std::runtime_error("edge is not connected to node");
  }
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
const typename MRBayesTree<BayesTreeType, FactorGraphType>::SharedEdge&
MRBayesTree<BayesTreeType, FactorGraphType>::Clique::parentEdge(
    const SharedClique& parent) const {
  for (const SharedEdge& edge : parent_edges_) {
    if (edge->parentClique() == parent) {
      return edge;
    }
  }
  throw std::runtime_error("no such parent");
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
const typename MRBayesTree<BayesTreeType, FactorGraphType>::SharedEdge&
MRBayesTree<BayesTreeType, FactorGraphType>::Clique::childEdge(
    const SharedClique& child) const {
  for (const SharedEdge& edge : child_edges_) {
    if (edge->childClique() == child) {
      return edge;
    }
  }
  throw std::runtime_error("no such child");
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
typename MRBayesTree<BayesTreeType, FactorGraphType>::CliqueVector
MRBayesTree<BayesTreeType, FactorGraphType>::Clique::parentCliques() const {
  CliqueVector parent_cliques;
  for (const SharedEdge& edge : parentEdges()) {
    parent_cliques.push_back(edge->parentClique());
  }
  return parent_cliques;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
typename MRBayesTree<BayesTreeType, FactorGraphType>::CliqueVector
MRBayesTree<BayesTreeType, FactorGraphType>::Clique::childCliques() const {
  CliqueVector child_cliques;
  for (const SharedEdge& edge : childEdges()) {
    child_cliques.push_back(edge->childClique());
  }
  return child_cliques;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
typename MRBayesTree<BayesTreeType, FactorGraphType>::CliqueVector
MRBayesTree<BayesTreeType, FactorGraphType>::Clique::neighborCliques() const {
  CliqueVector cliques;
  for (SharedEdge edge : parentEdges()) {
    cliques.push_back(edge->parentClique());
  }
  for (SharedEdge edge : childEdges()) {
    if (std::find(cliques.begin(), cliques.end(), edge->childClique()) ==
        cliques.end()) {
      cliques.push_back(edge->childClique());
    }
  }
  return cliques;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
bool MRBayesTree<BayesTreeType, FactorGraphType>::Clique::isParentOf(
    const SharedClique& clique) const {
  for (const auto& edge : child_edges_) {
    if (edge->childClique() == clique) {
      return true;
    }
  }
  return false;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
bool MRBayesTree<BayesTreeType, FactorGraphType>::Clique::isChildOf(
    const SharedClique& clique) const {
  for (const auto& edge : parent_edges_) {
    if (edge->parentClique() == clique) {
      return true;
    }
  }
  return false;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
std::string MRBayesTree<BayesTreeType, FactorGraphType>::Clique::name(
    const KeyFormatter& keyFormatter) const {
  std::string name_str;
  bool is_first = true;
  for (Key key : allKeys()) {
    if (!is_first) {
      name_str += " ";
    }
    name_str += keyFormatter(key);
    is_first = false;
  }
  return name_str;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
FactorGraphType
MRBayesTree<BayesTreeType, FactorGraphType>::Clique::factorsInClique(
    const FactorGraphType& graph, const VariableIndex& vi) const {
  FactorGraphType factors_in_clique;
  for (size_t i : factorIndicesInClique(graph, vi)) {
    factors_in_clique.push_back(graph.at(i));
  }
  return factors_in_clique;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
FactorIndices
MRBayesTree<BayesTreeType, FactorGraphType>::Clique::factorIndicesInClique(
    const FactorGraphType& graph, const VariableIndex& vi) const {
  std::set<FactorIndex> factor_indices_set;

  // must have frontal keys
  const KeySet& all_keys = allKeys();
  for (auto key : all_keys) {
    for (auto factor_idx : vi[key]) {
      bool to_add = true;
      for (Key key : graph.at(factor_idx)->keys()) {
        // all keys should be either frontal or separator
        if (all_keys.find(key) == all_keys.end()) {
          to_add = false;
          break;
        }
      }
      if (to_add) {
        factor_indices_set.insert(factor_idx);
      }
    }
  }
  return FactorIndices(factor_indices_set.begin(), factor_indices_set.end());
}

template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::Clique::removeEdge(
    const SharedEdge& edge) {
  parent_edges_.erase(
      std::remove(parent_edges_.begin(), parent_edges_.end(), edge),
      parent_edges_.end());
  child_edges_.erase(
      std::remove(child_edges_.begin(), child_edges_.end(), edge),
      child_edges_.end());
}

template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::Clique::disEngage() {
  parent_edges_.clear();
  child_edges_.clear();
}

/* ************************************************************************* */
/* ***************************** Member functions ************************** */
/* ************************************************************************* */

template <class BayesTreeType, class FactorGraphType>
typename MRBayesTree<BayesTreeType, FactorGraphType>::CliqueVector
MRBayesTree<BayesTreeType, FactorGraphType>::findCliquesByKeys(
    const FastVector<KeySet>& keys_vec, const SharedClique& root_clique) {
  CliqueVector clique_vec(keys_vec.size(), nullptr);
  std::stack<std::pair<SharedClique, SharedClique>> dfs;
  dfs.push(std::make_pair(root_clique, root_clique));

  while (!dfs.empty()) {
    SharedClique current_clique, parent_clique;
    boost::tie(current_clique, parent_clique) = dfs.top();
    dfs.pop();
    bool all_found = true;
    const KeySet& current_clique_keys = current_clique->allKeys();
    for (size_t i = 0; i < keys_vec.size(); i++) {
      if (clique_vec[i] == nullptr) {
        bool found = true;
        for (Key key : keys_vec[i]) {
          if (current_clique_keys.find(key) == current_clique_keys.end()) {
            found = false;
            break;
          }
        }
        if (found) {
          clique_vec[i] = current_clique;
        } else {
          all_found = false;
        }
      }
    }
    if (all_found) {
      break;
    }

    for (auto neighbor_clique : current_clique->neighborCliques()) {
      if (neighbor_clique != parent_clique) {
        dfs.push(std::make_pair(neighbor_clique, current_clique));
      }
    }
  }

  for (size_t i = 0; i < clique_vec.size(); i++) {
    if (clique_vec[i] == nullptr) {
      throw std::runtime_error("unidentified root keys");
    }
  }

  return clique_vec;
}

template <class BayesTreeType, class FactorGraphType>
typename MRBayesTree<BayesTreeType, FactorGraphType>::KeyCliqueMap
MRBayesTree<BayesTreeType, FactorGraphType>::findCliquesByKeys(
    const KeySet& keys, const SharedClique& root_clique) {
  KeyCliqueMap key_c_map;
  std::stack<std::pair<SharedClique, SharedClique>> dfs;
  dfs.push(std::make_pair(root_clique, root_clique));

  KeySet target_keys = keys;
  while (!dfs.empty()) {
    SharedClique current_clique, parent_clique;
    boost::tie(current_clique, parent_clique) = dfs.top();
    dfs.pop();
    for (Key key : current_clique->allKeys()) {
      if (target_keys.find(key) != target_keys.end()) {
        key_c_map[key] = current_clique;
        target_keys.erase(key);
      }
    }
    if (target_keys.size() == 0) {
      break;
    }

    for (auto neighbor_clique : current_clique->neighborCliques()) {
      if (neighbor_clique != parent_clique) {
        dfs.push(std::make_pair(neighbor_clique, current_clique));
      }
    }
  }

  if (target_keys.size() > 0) {
    throw std::runtime_error("unidentified root keys");
  }
  return key_c_map;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::addRoot(
    const SharedClique& root_clique, RootID root_id) {
  roots_[root_id] = root_clique;
  // start from other root cliques, ensure downward option is available, add
  // downward direction to corresponding cliques
  std::stack<std::pair<SharedClique, SharedClique>> dfs;
  dfs.push(std::make_pair(root_clique, root_clique));

  while (!dfs.empty()) {
    SharedClique current_clique;
    SharedClique parent_clique;
    boost::tie(current_clique, parent_clique) = dfs.top();
    dfs.pop();
    CliqueVector neighbor_cliques = current_clique->neighborCliques();
    for (SharedClique neighbor_clique : neighbor_cliques) {
      if (neighbor_clique != parent_clique) {
        if (!current_clique->isParentOf(neighbor_clique)) {
          auto edge = boost::make_shared<Edge>(current_clique, neighbor_clique);
          current_clique->addEdge(edge);
          neighbor_clique->addEdge(edge);
          dfs.push(std::make_pair(neighbor_clique, current_clique));
        }
      }
    }
  }
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::eliminateNodeBottomUp(
    const SharedClique& clique, const SharedClique& parent,
    const FactorGraphType& graph, const VariableIndex& vi,
    const Eliminate& elimination_function,
    boost::optional<EdgeSet&> boundary_edges) {
  // std::cout << "processing clique " << clique->name() << " from parent " <<
  // parent->name() << "\n";
  FactorGraphType gathered_factors;

  // gather marginal information from chlid edges
  for (const SharedEdge& edge : clique->childEdges()) {
    if (boundary_edges &&
        (*boundary_edges).find(edge) != (*boundary_edges).end()) {
      gathered_factors.add(edge->marginal());
      continue;
    }
    if (edge->childClique() != parent) {
      eliminateNodeBottomUp(edge->childClique(), clique, graph, vi,
                            elimination_function, boundary_edges);
      gathered_factors.add(edge->marginal());
    }
  }

  if (parent == clique) {
    return;
  }

  // add the factors in the factor graph
  const SharedEdge& edge = clique->parentEdge(parent);
  gathered_factors.push_back(edge->elimFactors(graph, vi));

  // perform elimination, and set elimination result
  auto frontal_keys = edge->frontalKeys();
  Ordering ordering(frontal_keys.begin(), frontal_keys.end());
  auto elimination_result = elimination_function(gathered_factors, ordering);
  // std::cout << "set for edge bottomup " << edge->name() << "\n";
  edge->setEliminationResult(elimination_result);
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::eliminateNodeTopDown(
    const SharedClique& clique, const SharedClique& child,
    const FactorGraphType& graph, const VariableIndex& vi,
    const Eliminate& elimination_function,
    boost::optional<EdgeSet&> boundary_edges) {
  // set marginals for the edge clique->child
  if (clique != child) {
    SharedEdge edge = clique->childEdge(child);
    if (boundary_edges &&
        (*boundary_edges).find(edge) != (*boundary_edges).end()) {
      return;
    }

    FactorGraphType gathered_factors;
    // gather information from all other branches
    for (const SharedEdge& gathered_edge : child->childEdges()) {
      if (gathered_edge->childClique() != clique) {
        gathered_factors.add(gathered_edge->marginal());
      }
    }
    gathered_factors.push_back(edge->elimFactors(graph, vi));
    auto frontal_keys = edge->frontalKeys();
    Ordering ordering(frontal_keys.begin(), frontal_keys.end());
    auto elimination_result = elimination_function(gathered_factors, ordering);

    // std::cout << "set for edge topdown " << edge->name() << "\n";
    edge->setEliminationResult(elimination_result);
  }

  // propagate to further edges
  for (const SharedClique& parent : clique->parentCliques()) {
    if (parent != child) {
      eliminateNodeTopDown(parent, clique, graph, vi, elimination_function,
                           boundary_edges);
    }
  }
}

template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::computeDelta(
    const SharedClique& clique, const SharedClique& parent,
    const FactorGraphType& graph, const VariableIndex& vi, VectorValues& deltas,
    const double& threshold, boost::optional<EdgeSet&> boundary_edges) const {
  Vector original_values;
  bool values_changed = false;

  // set marginals for the edge clique->child
  if (clique == parent) {  // root clique
    GaussianFactorGraph local_graph;
    local_graph.push_back(clique->factorsInClique(graph, vi));
    for (SharedEdge edge : clique->childEdges()) {
      local_graph.push_back(edge->marginal());
    }
    if (threshold > 0) {
      KeyVector all_keys(clique->allKeys().begin(), clique->allKeys().end());
      original_values = deltas.vector(all_keys);
      deltas.update(local_graph.optimize());
      Vector diff = original_values - deltas.vector(all_keys);
      if (diff.lpNorm<Eigen::Infinity>() >= threshold) {
        values_changed = true;
      }
    } else {
      deltas.update(local_graph.optimize());
    }
  } else {
    SharedEdge edge = parent->childEdge(clique);
    if (boundary_edges &&
        (*boundary_edges).find(edge) != (*boundary_edges).end()) {
      return;
    }
    if (threshold > 0) {
      KeySet frontal_keys_set = edge->frontalKeys();
      KeyVector frontal_keys(frontal_keys_set.begin(), frontal_keys_set.end());
      original_values = deltas.vector(frontal_keys);
      deltas.update(edge->conditional()->solve(deltas));
      Vector diff = original_values - deltas.vector(frontal_keys);
      if (diff.lpNorm<Eigen::Infinity>() >= threshold) {
        values_changed = true;
      }
    } else {
      deltas.update(edge->conditional()->solve(deltas));
    }
  }

  // propagate to further edges
  if (threshold <= 0 || values_changed) {
    for (const SharedClique& child : clique->childCliques()) {
      if (child != parent) {
        computeDelta(child, clique, graph, vi, deltas, threshold,
                     boundary_edges);
      }
    }
  }
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::eliminate(
    const FactorGraphType& graph, const VariableIndex& vi,
    const Eliminate& elimination_function) {
  const SharedClique& root_clique = roots_.begin()->second;
  eliminateNodeBottomUp(root_clique, root_clique, graph, vi,
                        elimination_function);
  eliminateNodeTopDown(root_clique, root_clique, graph, vi,
                       elimination_function);
}

/* ************************************************************************* */
/* *************************** Test & Visualization ************************ */
/* ************************************************************************* */

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
typename MRBayesTree<BayesTreeType, FactorGraphType>::CliqueVector
MRBayesTree<BayesTreeType, FactorGraphType>::allCliques() const {
  CliqueVector all_cliques;
  if (roots().empty()) {
    return all_cliques;
  }

  std::stack<std::pair<SharedClique, SharedClique>> dfs;
  const auto& root_clique = roots_.begin()->second;
  dfs.push(std::make_pair(root_clique, root_clique));
  while (!dfs.empty()) {
    SharedClique current_clique, parent_clique;
    boost::tie(current_clique, parent_clique) = dfs.top();
    dfs.pop();
    all_cliques.push_back(current_clique);

    for (SharedClique neighbor_clique : current_clique->neighborCliques()) {
      if (neighbor_clique != parent_clique)
        dfs.push(std::make_pair(neighbor_clique, current_clique));
    }
  }
  return all_cliques;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
typename MRBayesTree<BayesTreeType, FactorGraphType>::EdgeVector
MRBayesTree<BayesTreeType, FactorGraphType>::allEdges() const {
  EdgeVector all_edges;
  if (roots().empty()) {
    return all_edges;
  }

  std::stack<std::pair<SharedClique, SharedClique>> dfs;
  const auto& root_clique = roots_.begin()->second;
  dfs.push(std::make_pair(root_clique, root_clique));
  while (!dfs.empty()) {
    SharedClique current_clique, parent_clique;
    boost::tie(current_clique, parent_clique) = dfs.top();
    dfs.pop();
    if (current_clique != parent_clique) {
      if (current_clique->isParentOf(parent_clique)) {
        all_edges.push_back(current_clique->childEdge(parent_clique));
      }
      if (parent_clique->isParentOf(current_clique)) {
        all_edges.push_back(parent_clique->childEdge(current_clique));
      }
    }

    for (SharedClique neighbor_clique : current_clique->neighborCliques()) {
      if (neighbor_clique != parent_clique)
        dfs.push(std::make_pair(neighbor_clique, current_clique));
    }
  }
  return all_edges;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
typename MRBayesTree<BayesTreeType, FactorGraphType>::NameCliqueMap
MRBayesTree<BayesTreeType, FactorGraphType>::getNameCliqueMap() const {
  NameCliqueMap name_clique_map;
  for (auto& clique : allCliques()) {
    name_clique_map[clique->name()] = clique;
  }
  return name_clique_map;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
typename MRBayesTree<BayesTreeType, FactorGraphType>::NameEdgeMap
MRBayesTree<BayesTreeType, FactorGraphType>::getNameEdgeMap() const {
  NameEdgeMap name_edge_map;
  for (auto& edge : allEdges()) {
    name_edge_map[edge->name()] = edge;
  }
  return name_edge_map;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
bool MRBayesTree<BayesTreeType, FactorGraphType>::checkMarginals() const {
  SharedClique root_clique = roots_.begin()->second;

  std::stack<std::pair<SharedClique, SharedClique>> dfs;
  dfs.push(std::make_pair(root_clique, root_clique));

  while (!dfs.empty()) {
    SharedClique current_clique, parent_clique;
    boost::tie(current_clique, parent_clique) = dfs.top();
    dfs.pop();
    if (parent_clique->isParentOf(current_clique)) {
      SharedEdge edge = parent_clique->childEdge(current_clique);
      std::string edge_name =
          parent_clique->name() + "->" + current_clique->name();
      if (!edge->marginal()) {
        std::cout << edge_name << " marginal uncomputed\n";
        // return false;
      }
      if (!edge->conditional()) {
        std::cout << edge_name << " conditional uncomputed\n";
        // return false;
      }
    }
    if (parent_clique->isChildOf(current_clique)) {
      SharedEdge edge = current_clique->childEdge(parent_clique);
      std::string edge_name =
          current_clique->name() + "->" + parent_clique->name();
      if (!edge->marginal()) {
        std::cout << edge_name << " marginal uncomputed\n";
        // return false;
      }
      if (!edge->conditional()) {
        std::cout << edge_name << " conditional uncomputed\n";
        // return false;
      }
    }
    for (auto neighbor_clique : current_clique->neighborCliques()) {
      if (neighbor_clique != parent_clique) {
        dfs.push(std::make_pair(neighbor_clique, current_clique));
      }
    }
  }
  return true;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
bool MRBayesTree<BayesTreeType, FactorGraphType>::Clique::equals(
    const SharedClique& other) const {
  if (name() != other->name()) {
    return false;
  }
  if (parentEdges().size() != other->parentEdges().size()) {
    return false;
  }
  if (childEdges().size() != other->childEdges().size()) {
    return false;
  }

  // check parent edges
  std::map<std::string, SharedEdge> this_name_p_edge_map;
  std::map<std::string, SharedEdge> other_name_p_edge_map;
  for (const SharedEdge& edge : parentEdges()) {
    this_name_p_edge_map[edge->parentClique()->name()] = edge;
  }
  for (const SharedEdge& edge : other->parentEdges()) {
    other_name_p_edge_map[edge->parentClique()->name()] = edge;
  }
  for (auto it : this_name_p_edge_map) {
    std::string name = it.first;
    if (other_name_p_edge_map.find(name) == other_name_p_edge_map.end()) {
      return false;
    }
    const SharedEdge& this_edge = it.second;
    const SharedEdge& other_edge = other_name_p_edge_map.at(name);
    if (!this_edge->equals(other_edge)) {
      return false;
    }
  }

  // check child edges
  std::map<std::string, SharedEdge> this_name_c_edge_map;
  std::map<std::string, SharedEdge> other_name_c_edge_map;
  for (const SharedEdge& edge : childEdges()) {
    this_name_c_edge_map[edge->childClique()->name()] = edge;
  }
  for (const SharedEdge& edge : other->childEdges()) {
    other_name_c_edge_map[edge->childClique()->name()] = edge;
  }
  for (auto it : this_name_c_edge_map) {
    std::string name = it.first;
    if (other_name_c_edge_map.find(name) == other_name_c_edge_map.end()) {
      return false;
    }
    const SharedEdge& this_edge = it.second;
    const SharedEdge& other_edge = other_name_c_edge_map.at(name);
    if (!this_edge->equals(other_edge)) {
      return false;
    }
  }
  return true;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
bool MRBayesTree<BayesTreeType, FactorGraphType>::Edge::equals(
    const SharedEdge& other) const {
  if (parentClique()->name() != other->parentClique()->name()) {
    return false;
  }
  if (childClique()->name() != other->childClique()->name()) {
    return false;
  }

  // check marginal and conditional
  if (!marginal()) {
    std::cout << "this marginal not computed\n";
    return false;
  }
  if (!other->marginal()) {
    std::cout << "other marginal not computed\n";
    return false;
  }
  if (!conditional()) {
    std::cout << "this conditional not computed\n";
    return false;
  }
  if (!other->conditional()) {
    std::cout << "other conditional not computed\n";
    return false;
  }
  if (!marginal_->equals(*(other->marginal()))) {
    std::cout << "on edge " << name() << "\n";
    std::cout << "marginal do not match!\n";
    // marginal()->print();
    // other->marginal()->print();
    // return false;
  }
  if (!conditional_->equals(*(other->conditional()))) {
    std::cout << "on edge " << name() << "\n";
    std::cout << "conditional do not match!\n";
    // conditional()->print();
    // other->conditional()->print();
    // return false;
  }

  return true;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
bool MRBayesTree<BayesTreeType, FactorGraphType>::equals(
    const This& other) const {
  if (roots_.size() != other.roots().size()) {
    return false;
  }

  CliqueVector this_cliques = allCliques();
  CliqueVector other_cliques = other.allCliques();
  if (this_cliques.size() != other_cliques.size()) {
    return false;
  }

  std::map<std::string, SharedClique> this_name_clique_map;
  std::map<std::string, SharedClique> other_name_clique_map;
  for (const SharedClique& clique : this_cliques) {
    this_name_clique_map[clique->name()] = clique;
  }
  for (const SharedClique& clique : other_cliques) {
    other_name_clique_map[clique->name()] = clique;
  }

  for (auto it : this_name_clique_map) {
    std::string name = it.first;
    const SharedClique& this_clique = it.second;
    if (other_name_clique_map.find(name) == other_name_clique_map.end()) {
      return false;
    }
    const SharedClique& other_clique = other_name_clique_map.at(name);
    if (!this_clique->equals(other_clique)) {
      return false;
    }
  }

  return true;
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::Clique::print(
    const KeyFormatter& keyFormatter) const {
  PrintKeySet(keys_, "", MultiRobotKeyFormatter);
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::Edge::print(
    const KeyFormatter& keyFormatter) const {
  std::cout << "parent: " << parentClique()->name() << "\n";
  std::cout << "child: " << childClique()->name() << "\n";
  std::cout << "marginal:\n";
  if (!marginal()) {
    std::cout << "NULL\n";
  } else {
    marginal()->print();
  }
  std::cout << "conditional:\n";
  if (!conditional()) {
    std::cout << "NULL\n";
  } else {
    conditional()->print();
  }
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::print() const {
  struct DfsPrintEntry {
    size_t depth;
    SharedClique clique;
    SharedClique parent;

    DfsPrintEntry(size_t _depth, SharedClique _clique, SharedClique _parent)
        : depth(_depth), clique(_clique), parent(_parent) {}
  };

  std::stack<DfsPrintEntry> dfs;
  const auto& root = roots_.begin()->second;
  dfs.push(DfsPrintEntry(0, root, root));
  while (!dfs.empty()) {
    DfsPrintEntry current = dfs.top();
    for (size_t i = 0; i < current.depth; i++) {
      std::cout << "|";
    }
    if (current.clique->isParentOf(current.parent) &&
        current.parent->isParentOf(current.clique)) {
      std::cout << "<-> ";
    } else if (current.clique->isParentOf(current.parent)) {
      std::cout << "<- ";
    } else if (current.parent->isParentOf(current.clique)) {
      std::cout << "-> ";
    }
    std::cout << current.clique->name() << "\n";

    dfs.pop();
    for (SharedClique child : current.clique->neighborCliques()) {
      if (child != current.parent)
        dfs.push(DfsPrintEntry(current.depth + 1, child, current.clique));
    }
  }
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
typename MRBayesTree<BayesTreeType, FactorGraphType>::LocationType
MRBayesTree<BayesTreeType, FactorGraphType>::defaultLayout() const {
  LocationType locations;
  SharedClique root_clique = roots().begin()->second;
  std::vector<std::pair<SharedClique, SharedClique>> bfs_layer;
  bfs_layer.push_back(std::make_pair(root_clique, root_clique));
  int layer = 0;
  while (bfs_layer.size() > 0) {
    std::vector<std::pair<SharedClique, SharedClique>> next_layer;
    for (size_t j = 0; j < bfs_layer.size(); j++) {
      const SharedClique& current_clique = bfs_layer[j].first;
      const SharedClique& parent_clique = bfs_layer[j].second;

      // assign locations for this layer

      double x =
          bfs_layer.size() == 1
              ? 0
              : (double)j / double(bfs_layer.size() - 1) * (2 * layer) - layer;
      double y = layer;
      locations[current_clique] = (Vector(2) << x, y).finished();

      // add nodes to next layer
      CliqueVector neighbor_cliques = current_clique->neighborCliques();
      for (SharedClique neighbor_clique : neighbor_cliques) {
        if (neighbor_clique != parent_clique) {
          next_layer.push_back(std::make_pair(neighbor_clique, current_clique));
        }
      }
    }
    bfs_layer = next_layer;
    layer += 1;
  }
  return locations;
}

template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::setEdgeWeightBottomUp(
    std::map<SharedEdge, size_t>& edge_weights, const SharedClique& clique,
    const SharedClique& parent) const {
  // compute weight for all child edges
  size_t max_child_edge_weight = 0;
  for (const SharedEdge& child_edge : clique->childEdges()) {
    if (child_edge->dualDirection() && child_edge->childClique() != parent) {
      setEdgeWeightBottomUp(edge_weights, child_edge->childClique(), clique);
      max_child_edge_weight =
          std::max(max_child_edge_weight, edge_weights.at(child_edge));
    }
  }
  // set weight for this edge
  if (clique != parent) {
    edge_weights[parent->childEdge(clique)] = max_child_edge_weight + 1;
  }
}

template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::setEdgeWeightTopDown(
    std::map<SharedEdge, size_t>& edge_weights, const SharedClique& clique,
    const SharedClique& child) const {
  // compute weight for this edge
  if (clique != child) {
    size_t max_child_edge_weight = 0;
    for (const SharedEdge& child_edge : child->childEdges()) {
      if (child_edge->dualDirection() && child_edge->childClique() != clique) {
        max_child_edge_weight =
            std::max(max_child_edge_weight, edge_weights.at(child_edge));
      }
    }
    edge_weights[clique->childEdge(child)] = max_child_edge_weight + 1;
  }

  // compute weight for parent edges
  for (const SharedEdge& parent_edge : clique->parentEdges()) {
    if (parent_edge->dualDirection() && parent_edge->parentClique() != child) {
      setEdgeWeightTopDown(edge_weights, parent_edge->parentClique(), clique);
    }
  }
}

template <class BayesTreeType, class FactorGraphType>
size_t MRBayesTree<BayesTreeType, FactorGraphType>::getWeight(
    const std::map<SharedEdge, size_t>& edge_weights,
    const SharedClique& clique) const {
  size_t max_child_edge_weight = 0;
  for (const SharedEdge& child_edge : clique->childEdges()) {
    if (child_edge->dualDirection()) {
      max_child_edge_weight =
          std::max(max_child_edge_weight, edge_weights.at(child_edge));
    }
  }
  return max_child_edge_weight;
}

template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::findBalanceNodeRecursive(
    const std::map<SharedEdge, size_t>& edge_weights,
    const SharedClique& clique, const SharedClique& parent,
    SharedClique& balance_node, size_t& balance_dist) const {
  // compare this node and balance node
  size_t dist = getWeight(edge_weights, clique);
  bool has_single_direction_edge = true;
  // for (const SharedEdge& child_edge : clique->childEdges()) {
  //   if (!child_edge->dualDirection()) {
  //     has_single_direction_edge = true;
  //     break;
  //   }
  // }

  if (has_single_direction_edge && dist < balance_dist) {
    balance_dist = dist;
    balance_node = clique;
  }

  // check all following nodes
  for (const SharedEdge& child_edge : clique->childEdges()) {
    if (child_edge->dualDirection() && child_edge->childClique() != parent) {
      findBalanceNodeRecursive(edge_weights, child_edge->childClique(), clique,
                               balance_node, balance_dist);
    }
  }
}

std::string keysToString(const KeySet& keys,
                         const KeyFormatter& key_formatter) {
  std::string str;
  bool is_first = true;
  for (Key key : keys) {
    if (!is_first) {
      str += ",";
    }
    std::string formatted_key = dyno::DynoLikeKeyFormatter(key);
    //  LabeledSymbol symbol(key);
    //  if (symbol.chr() == 'L') {
    //    formatted_key = "L" + std::to_string(symbol.index());
    //  }
    //  else {
    //    char robot_chr = symbol.label() - 1;
    //    formatted_key = robot_chr + std::to_string(symbol.index());
    //  }
    str += formatted_key;
    is_first = false;
  }
  return str;
}

template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::saveNodeRecursive(
    std::ofstream& o_file, const std::map<SharedClique, size_t>& clique_id_map,
    const SharedClique& clique, const SharedClique& parent,
    const CliqueSet& root_cliques, const CliqueSet& top_cliques,
    const EdgeSet& prop_edges, const KeyFormatter& key_formatter) const {
  // save the node
  bool among_roots = false;
  for (const SharedEdge& child_edge : clique->childEdges()) {
    if (child_edge->dualDirection()) {
      among_roots = true;
      break;
    }
  }
  std::string name;
  if (among_roots || clique->parentEdges().size() == 0) {
    name = keysToString(clique->allKeys(), key_formatter);
  } else {
    SharedEdge parent_edge = clique->parentEdges()[0];
    name = keysToString(parent_edge->frontalKeys(), key_formatter) + ":" +
           keysToString(parent_edge->separatorKeys(), key_formatter);
  }
  if (root_cliques.find(clique) != root_cliques.end()) {
    o_file << clique_id_map.at(clique) << "[label=" << Quoted(name)
           << "penwidth=5 color=red fontsize=12];" << std::endl;
  } else if (top_cliques.find(clique) != top_cliques.end()) {
    o_file << clique_id_map.at(clique) << "[label=" << Quoted(name)
           << "penwidth=5 color=green fontsize=12];" << std::endl;
  } else {
    o_file << clique_id_map.at(clique) << "[label=" << Quoted(name)
           << "penwidth=3 fontsize=12];" << std::endl;
  }

  // save the edges
  for (const SharedEdge& child_edge : clique->childEdges()) {
    if (child_edge->childClique() != parent) {
      if (child_edge->dualDirection()) {
        o_file << clique_id_map.at(child_edge->childClique()) << "->"
               << clique_id_map.at(clique)
               << "[penwidth=6 dir=both color=" + Quoted("blue:blue") + "]"
               << std::endl;
      } else {
        o_file << clique_id_map.at(clique) << "->"
               << clique_id_map.at(child_edge->childClique()) << "[penwidth=3]"
               << std::endl;
      }
      saveNodeRecursive(o_file, clique_id_map, child_edge->childClique(),
                        clique, root_cliques, top_cliques, prop_edges,
                        key_formatter);
    }
  }
}

/* ************************************************************************* */
template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::saveGraph(
    const std::string& file_name, const CliqueSet& top_cliques,
    const EdgeSet& prop_edges, const KeyFormatter& key_formatter) const {
  // calculate edge weight (distance to furthest root) for al dual-direction
  // edges
  SharedClique root_clique = roots_.begin()->second;
  std::map<SharedEdge, size_t> edge_weights;
  setEdgeWeightBottomUp(edge_weights, root_clique, root_clique);
  setEdgeWeightTopDown(edge_weights, root_clique, root_clique);

  // decide the balance node
  SharedClique balance_node = root_clique;
  size_t balance_dist = getWeight(edge_weights, balance_node);
  findBalanceNodeRecursive(edge_weights, root_clique, root_clique, balance_node,
                           balance_dist);

  // output nodes and edges
  std::map<SharedClique, size_t> clique_id_map;
  size_t clique_id = 0;
  for (const SharedClique& clique : allCliques()) {
    clique_id++;
    clique_id_map[clique] = clique_id;
  }

  std::ofstream o_file(file_name, std::ofstream::out);
  o_file << "digraph G{" << std::endl;
  CliqueSet root_cliques;
  for (const auto& it : roots_) {
    root_cliques.insert(it.second);
  }
  saveNodeRecursive(o_file, clique_id_map, balance_node, balance_node,
                    root_cliques, top_cliques, prop_edges, key_formatter);
  o_file << "}" << std::endl;
  o_file.close();
}

template <class BayesTreeType, class FactorGraphType>
void MRBayesTree<BayesTreeType, FactorGraphType>::exportTree(
    const std::string fileName, const LocationType& locations) const {
  CliqueVector cliques = allCliques();

  std::vector<std::string> clique_list;
  std::vector<std::string> edge_list;

  for (SharedClique& clique : cliques) {
    std::vector<AttributeType> attributes;
    std::vector<std::string> keys;
    for (Key key : clique->allKeys()) {
      keys.push_back(Quoted(MultiRobotKeyFormatter(key)));
    }
    bool is_root = false;
    std::vector<RootID> root_ids;
    for (const auto& it : roots_) {
      if (clique == it.second) {
        is_root = true;
        root_ids.push_back(it.first);
      }
    }
    attributes.push_back(
        std::make_pair(Quoted("name"), Quoted(clique->name())));
    attributes.push_back(std::make_pair(Quoted("keys"), JsonList(keys, -1)));
    attributes.push_back(
        std::make_pair(Quoted("root"), is_root ? "true" : "false"));
    if (locations.find(clique) != locations.end()) {
      std::string loc_string;
      auto location = locations.at(clique);
      loc_string = "[" + std::to_string(location(0)) + ", " +
                   std::to_string(location(1)) + "]";
      attributes.push_back(std::make_pair(Quoted("location"), loc_string));
    }

    if (is_root) {
      std::string root_id_str;
      root_id_str += std::to_string(root_ids[0]);
      for (size_t i = 1; i < root_ids.size(); i++) {
        root_id_str += "," + std::to_string(root_ids[i]);
      }
      attributes.push_back(
          std::make_pair(Quoted("root_id"), Quoted(root_id_str)));
    }
    clique_list.push_back(JsonDict(attributes));
  }

  CliqueSet visited_cliques;
  for (const SharedClique& clique : cliques) {
    visited_cliques.insert(clique);
    std::string clique_name = clique->name();
    for (SharedClique neighbor_clique : clique->neighborCliques()) {
      if (visited_cliques.find(neighbor_clique) != visited_cliques.end()) {
        continue;
      }
      std::string neighbor_name = neighbor_clique->name();
      std::vector<AttributeType> attributes;
      std::string parent_name, child_name;
      bool dual_direction;

      if (clique->isParentOf(neighbor_clique)) {
        parent_name = clique_name;
        child_name = neighbor_name;
        dual_direction = clique->childEdge(neighbor_clique)->dualDirection();
      } else {
        parent_name = neighbor_name;
        child_name = clique_name;
        dual_direction = clique->parentEdge(neighbor_clique)->dualDirection();
      }

      attributes.push_back(
          std::make_pair(Quoted("name"), Quoted(parent_name + child_name)));
      // attributes.push_back(std::make_pair(Quoted("variables"), "[" +
      // std::to_string(parentId) + ", " + std::to_string(childId) + "]"));
      attributes.push_back(
          std::make_pair(Quoted("parent"), Quoted(parent_name)));
      attributes.push_back(std::make_pair(Quoted("child"), Quoted(child_name)));
      attributes.push_back(std::make_pair(Quoted("dualDirection"),
                                          dual_direction ? "true" : "false"));
      edge_list.push_back(JsonDict(attributes));
    }
  }

  std::string clique_list_str = JsonList(clique_list);
  std::string edge_list_str = JsonList(edge_list);
  std::string all_str =
      JsonList(std::vector<std::string>{clique_list_str, edge_list_str});

  std::ofstream stm;
  stm.open(fileName);
  stm << all_str;
  stm.close();
}

template class MRBayesTree<GaussianBayesTree, GaussianFactorGraph>;

}  // namespace gtsam
