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

#include <gtsam/inference/JunctionTree.h>
#include <gtsam/linear/GaussianBayesTree.h>

namespace gtsam {

template <class BayesTreeType, class FactorGraphType>
class MRBayesTree {
 public:
  typedef MRBayesTree<BayesTreeType, FactorGraphType> This;
  typedef JunctionTree<BayesTreeType, FactorGraphType> JunctionTreeType;
  typedef typename FactorGraphType::Eliminate Eliminate;
  typedef typename FactorGraphType::EliminationResult EliminationResult;
  typedef typename FactorGraphType::FactorType FactorType;
  typedef EliminationTraits<FactorGraphType> EliminationTraitsType;
  typedef typename EliminationTraitsType::BayesNetType BayesNetType;
  typedef typename BayesNetType::ConditionalType ConditionalType;
  typedef boost::shared_ptr<ConditionalType> SharedConditional;
  typedef typename FactorType::shared_ptr SharedFactor;

  class Edge;
  class Clique;
  typedef boost::shared_ptr<Clique> SharedClique;
  typedef boost::shared_ptr<Edge> SharedEdge;
  typedef FastVector<SharedEdge> EdgeVector;
  typedef std::set<SharedEdge> EdgeSet;
  typedef FastVector<SharedClique> CliqueVector;
  typedef std::map<Key, SharedClique> KeyCliqueMap;
  typedef std::set<SharedClique> CliqueSet;
  typedef std::map<std::string, SharedClique> NameCliqueMap;
  typedef std::map<std::string, SharedEdge> NameEdgeMap;

  typedef int RootID;
  typedef std::map<RootID, Key> RootKeyMap;
  typedef std::map<RootID, KeySet> RootKeySetMap;
  typedef std::map<RootID, SharedClique> RootCliqueMap;
  typedef std::map<RootID, FactorGraphType> RootGraphMap;

  /** Default Constructor of MRBayesTree */
  MRBayesTree() {}

  /** constructor that directly initiates from roots, for testing */
  MRBayesTree(const RootCliqueMap& roots) : roots_(roots) {}

  /** Constructor from junction tree. Will create a MRBayesTree with only the
   * root in the junction tree. (note: this constructor only construct the
   * structure of MRBT, but do not perform elimination to compute marginals) */
  MRBayesTree(const JunctionTreeType& junction_tree, RootID root_id = 0);

  /** Constructor from junction tree
   * @param junction_tree   junction tree
   * @param graph           original factor graph
   * @param root_id         id of this root
   * @param other_root_keys_map   keys to identify other selected roots
   * @param symbolic        if specified, will not perform actual elimination
   * @param elimination_function
   */
  MRBayesTree(const JunctionTreeType& junction_tree,
              const FactorGraphType& graph, const RootID root_id,
              const RootKeySetMap& other_root_keys_map,
              const bool symbolic = false,
              const Eliminate& elimination_function = EliminateCholesky);

  /** Constructor from factor graph with elimination ordering
   * @param graph           original factor graph
   * @param order           variable elimination order
   * @param root_id         id of this root
   * @param other_root_keys_map   keys to identify selected roots
   * @param symbolic        if specified, will not perform actual elimination
   * @param elimination_function
   */
  MRBayesTree(const FactorGraphType& graph, const Ordering& order,
              const RootID root_id, const RootKeySetMap& other_root_keys_map,
              const bool symbolic = false,
              const Eliminate& elimination_function = EliminateCholesky);

  /** destructor to avoid memory leak with shared pointers */
  ~MRBayesTree();

  /** directional edge between cliques */
  class Edge {
   public:
    Edge(const SharedClique& parent_clique, const SharedClique& child_clique)
        : parent_clique_(parent_clique), child_clique_(child_clique) {}

    void setParent(const SharedClique& clique) { parent_clique_ = clique; }

    void setChild(const SharedClique& clique) { child_clique_ = clique; }

    void setEliminationResult(const EliminationResult& elimination_result);

    bool dualDirection() const;

    /** find factors to eliminate for this edge */
    FactorGraphType elimFactors(const FactorGraphType& graph,
                                const VariableIndex& vi) const;

    FactorIndices elimFactorIndices(const FactorGraphType& graph,
                                    const VariableIndex& vi) const;

    KeySet frontalKeys() const;

    KeySet separatorKeys() const;

    const SharedClique& parentClique() const { return parent_clique_; }

    const SharedClique& childClique() const { return child_clique_; }

    const SharedFactor& marginal() const { return marginal_; }

    const SharedConditional& conditional() const { return conditional_; }

    std::string name(
        const KeyFormatter& keyFormatter = MultiRobotKeyFormatter) const;

    void print(const KeyFormatter& keyFormatter = MultiRobotKeyFormatter) const;

    bool equals(const SharedEdge& other) const;

   protected:
    SharedClique parent_clique_;
    SharedClique child_clique_;
    SharedFactor marginal_ = nullptr;
    SharedConditional conditional_ = nullptr;
  };

  /** Clique of MRBayesTree */
  class Clique {
   public:
    Clique() {}

    Clique(KeySet keys) : keys_(keys) {}

    void setKeys(const KeySet& keys) { keys_ = keys; }

    /** add edge to clique, should detect wheter it's child or parent edge */
    void addEdge(SharedEdge edge);

    const KeySet& allKeys() const { return keys_; }

    size_t size() const { return keys_.size(); }

    /** edges pointing to this clique */
    const EdgeVector& parentEdges() const { return parent_edges_; }

    /** edges pointing away from this clique */
    const EdgeVector& childEdges() const { return child_edges_; }

    CliqueVector parentCliques() const;

    CliqueVector childCliques() const;

    const SharedEdge& parentEdge(const SharedClique& parent) const;

    const SharedEdge& childEdge(const SharedClique& child) const;

    CliqueVector neighborCliques() const;

    bool isParentOf(const SharedClique& clique) const;

    bool isChildOf(const SharedClique& clique) const;

    FactorGraphType factorsInClique(const FactorGraphType& graph,
                                    const VariableIndex& vi) const;

    FactorIndices factorIndicesInClique(const FactorGraphType& graph,
                                        const VariableIndex& vi) const;

    std::string name(
        const KeyFormatter& keyFormatter = MultiRobotKeyFormatter) const;

    void print(const KeyFormatter& keyFormatter = MultiRobotKeyFormatter) const;

    bool equals(const SharedClique& other) const;

    void removeEdge(const SharedEdge& edge);

    void disEngage();

   private:
    KeySet keys_;
    EdgeVector parent_edges_;
    EdgeVector child_edges_;
  };

  /** add a root to the tree, will also add the edges in the root direction
   * @param root_clique     root clique for the new root
   * @param root_id         id of new root
   */
  void addRoot(const SharedClique& root_clique, RootID root_id);

  void print() const;

  void setEdgeWeightBottomUp(std::map<SharedEdge, size_t>& edge_weights,
                             const SharedClique& clique,
                             const SharedClique& parent) const;

  void setEdgeWeightTopDown(std::map<SharedEdge, size_t>& edge_weights,
                            const SharedClique& clique,
                            const SharedClique& child) const;

  size_t getWeight(const std::map<SharedEdge, size_t>& edge_weights,
                   const SharedClique& clique) const;

  void findBalanceNodeRecursive(
      const std::map<SharedEdge, size_t>& edge_weights,
      const SharedClique& clique, const SharedClique& parent,
      SharedClique& balance_node, size_t& balance_dist) const;

  void saveNodeRecursive(std::ofstream& o_file,
                         const std::map<SharedClique, size_t>& clique_id_map,
                         const SharedClique& clique, const SharedClique& parent,
                         const CliqueSet& root_cliques,
                         const CliqueSet& top_cliques,
                         const EdgeSet& prop_edges,
                         const KeyFormatter& key_formatter) const;

  void saveGraph(
      const std::string& file_name, const CliqueSet& top_cliques = CliqueSet(),
      const EdgeSet& prop_edges = EdgeSet(),
      const KeyFormatter& keyFormatter = MultiRobotKeyFormatter) const;

  /** perform dfs from the first root, and return all cliques */
  CliqueVector allCliques() const;

  NameCliqueMap getNameCliqueMap() const;

  EdgeVector allEdges() const;

  NameEdgeMap getNameEdgeMap() const;

  typedef std::map<SharedClique, Vector2> LocationType;

  /** visualization layout of the tree */
  LocationType defaultLayout() const;

  /** export the tree into a json file for visualization */
  void exportTree(const std::string fileName,
                  const LocationType& locations = LocationType()) const;

  bool checkMarginals() const;

  /** check equality with another tree */
  bool equals(const This& other) const;

  const RootCliqueMap& roots() const { return roots_; }

  /** starting from the root, dfs to find the first cliques containing the keys
   * @param keys      keys to search for
   * @param root      root to start the dfs search
   * @return map from key to clique
   */
  KeyCliqueMap findCliquesByKeys(const KeySet& keys, const SharedClique& root);

  CliqueVector findCliquesByKeys(const FastVector<KeySet>& keys_vec,
                                 const SharedClique& root);

  /** disengage roots of the tree, so that the cliques won't be deleted on
   * calling destructor */
  void disEngage() { roots_.clear(); }

  /** recursively compute the delta of frontal variables, up to the boundary
   * edges (not including boundary edges). In this function call, only the
   * specified edge (parent -> clique) is updated, and all descendent edges are
   * called recursively. If threshold is specified >0, the change in delta will
   * be checked to implement wildfire spread
   * @param clique    parent clique of the edge
   * @param parent    parent clique of the edge
   * @param graph     original factor graph
   * @param vi        variable index for the graph
   * @param deltas    (return) computed delta values
   * @param threshold threshold in wildfire spread of delta update
   * @param boundary_edges          boundary edges to stop the process
   */
  void computeDelta(
      const SharedClique& clique, const SharedClique& parent,
      const FactorGraphType& graph, const VariableIndex& vi,
      VectorValues& deltas, const double& threshold = 0,
      boost::optional<EdgeSet&> boundary_edges = boost::none) const;

  void printRoots() const {
    std::cout << "roots:\n";
    for (const auto& it : roots_) {
      std::cout << it.first << ":\t" << it.second->name() << "\n";
    }
  }

  size_t size() const { return allCliques().size(); }

  size_t largestCliqueSize() const {
    size_t largest_clique_size = 0;
    for (const SharedClique& clique : allCliques()) {
      if (clique->size() > largest_clique_size) {
        largest_clique_size = clique->size();
      }
    }
    return largest_clique_size;
  }

 protected:
  RootCliqueMap roots_;

  SharedClique CopyFromJTByNode(
      const typename JunctionTreeType::sharedCluster jt_node);

  /** recursively compute the marginal and conditional of the edges in the
   * direction that points away from the root, up to the boundary edges (not
   * including boundary edges). Use a bottom-up approach such that the edges
   * further from the root are processed first. In this function call, we first
   * recursively update the descendent edges, then update the specified edge
   * (parent -> clique).
   * @param clique    child clique of the edge
   * @param parent    parent clique of the edge
   * @param graph     original factor graph
   * @param vi        variable index for the graph
   * @param elimination_function    function to perform variable elimination
   * @param boundary_edges          boundary edges to stop the process
   */
  void eliminateNodeBottomUp(
      const SharedClique& clique, const SharedClique& parent,
      const FactorGraphType& graph, const VariableIndex& vi,
      const Eliminate& elimination_function,
      boost::optional<EdgeSet&> boundary_edges = boost::none);

  /** recursively compute the marginal and conditional of the edges in the
   * direction that points to the root, up to the boundary edges (not including
   * boundary edges). Use a top-down approach such that the edges closer to the
   * root are processed first. In this function call, only the specified edge
   * (clique -> child) is updated, and all ancester edges are called
   * recursively.
   * @param clique    parent clique of the edge
   * @param child     child clique of the edge
   * @param graph     original factor graph
   * @param vi        variable index for the graph
   * @param elimination_function    function to perform variable elimination
   * @param boundary_edges          boundary edges to stop the process
   */
  void eliminateNodeTopDown(
      const SharedClique& clique, const SharedClique& child,
      const FactorGraphType& graph, const VariableIndex& vi,
      const Eliminate& elimination_function,
      boost::optional<EdgeSet&> boundary_edges = boost::none);

  /** do elimination across cliques to compute marginals and conditionals in all
   * edges of the tree
   * @param graph     original factor graph
   * @param vi        variable index of the graph
   * @param elimination_function  function to do variable elimination
   */
  void eliminate(const FactorGraphType& graph, const VariableIndex& vi,
                 const Eliminate& elimination_function);
};

typedef MRBayesTree<GaussianBayesTree, GaussianFactorGraph> GaussianMRBayesTree;

}  // namespace gtsam
