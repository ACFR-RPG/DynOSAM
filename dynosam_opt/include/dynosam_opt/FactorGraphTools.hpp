/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <glog/logging.h>
#include <gtsam/base/treeTraversal-inst.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <deque>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/viz/types.hpp>
#include <unordered_map>
#include <unordered_set>

#include "dynosam_common/Types.hpp"
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/Numerical.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

namespace factor_graph_tools {

/**
 * @brief Wrapper on the creation of a robust Huber noise model from a Shared
 * noise model
 *
 * @param k double k paramter for the Huber lss function
 * @param model  gtsam::SharedNoiseModel model to make robust
 * @return gtsam::SharedNoiseModel
 */
gtsam::SharedNoiseModel robustifyHuber(double k, gtsam::SharedNoiseModel model);

void addBetweenFactor(FrameId from_frame, FrameId to_frame,
                      const gtsam::Pose3 from_pose_to,
                      gtsam::SharedNoiseModel noise_model,
                      gtsam::NonlinearFactorGraph& graph);

// expects DERIVEDFACTOR to be a NoiseModelFactor so that we can use
// whitenedError
template <typename DERIVEDFACTOR>
gtsam::FactorIndices determineFactorOutliers(
    const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values,
    double confidence = 0.99) {
  gtsam::FactorIndices outlier_indicies;
  for (size_t k = 0; k < graph.size(); k++) {
    if (!graph[k]) continue;

    auto derived_factor = boost::dynamic_pointer_cast<DERIVEDFACTOR>(graph[k]);
    if (derived_factor) {
      double weighted_threshold =
          0.5 *
          chi_squared_quantile(
              derived_factor->dim(),
              confidence);  // 0.5 derives from the error definition in gtsam
      // NOTE: double casting as we expect DERIVEDFACTOR to be a
      // NoiseModelFactor
      gtsam::NoiseModelFactor::shared_ptr nm_factor =
          boost::dynamic_pointer_cast<gtsam::NoiseModelFactor>(derived_factor);
      CHECK(nm_factor);
      // all noise models must be gaussian otherwise whittening will be
      // reweighted
      auto robust = boost::dynamic_pointer_cast<gtsam::noiseModel::Robust>(
          nm_factor->noiseModel());
      auto gaussian_nm_factor =
          robust ? nm_factor->cloneWithNewNoiseModel(robust->noise())
                 : nm_factor;
      CHECK(gaussian_nm_factor);

      double error = gaussian_nm_factor->error(values);

      if (error > weighted_threshold) {
        outlier_indicies.push_back(k);
      }
    }
  }
  return outlier_indicies;
}

template <typename GRAPH>
size_t getAssociatedFactors(
    std::vector<typename GRAPH::const_iterator>& associated_factors,
    const GRAPH& graph, gtsam::Key query_key) {
  associated_factors.clear();

  for (auto it = graph.begin(); it != graph.end(); it++) {
    auto factor = *it;
    if (factor->find(query_key) != factor->end()) {
      associated_factors.push_back(it);
    }
  }
  return associated_factors.size();
}

template <typename GRAPH>
size_t getAssociatedFactors(gtsam::FactorIndices& associated_factors,
                            const GRAPH& graph, gtsam::Key query_key) {
  associated_factors.clear();
  std::vector<typename GRAPH::const_iterator> associated_factor_iters;
  size_t result =
      getAssociatedFactors(associated_factor_iters, graph, query_key);

  for (const typename GRAPH::const_iterator& iter : associated_factor_iters) {
    // index in graph
    associated_factors.push_back(std::distance(graph.begin(), iter));
  }
  return result;
}

template <typename GRAPH>
size_t getAssociatedFactors(
    std::vector<typename GRAPH::sharedFactor>& associated_factors,
    const GRAPH& graph, gtsam::Key query_key) {
  associated_factors.clear();
  std::vector<typename GRAPH::const_iterator> associated_factor_iters;
  size_t result =
      getAssociatedFactors(associated_factor_iters, graph, query_key);

  for (const typename GRAPH::const_iterator& iter : associated_factor_iters) {
    associated_factors.push_back(*iter);
  }
  return result;
}

// wrapper on the gtsam tree traversal functions to make life easier
struct travsersal {
 public:
  template <typename CLQUE>
  using CliqueVisitor = std::function<void(const boost::shared_ptr<CLQUE>&)>;

  // Do nothing
  template <typename CLQUE>
  inline static void NoopCliqueVisitor(const boost::shared_ptr<CLQUE>&) {}

 private:
  // helper data structures
  struct Data {};

  /**
   * @brief Definition of VISITOR_PRE struct as per gtsam::treeTraversal
   *
   * Simply contains a user defined function which takes a shared_ptr to the
   * CLIQUE as an argument. Simplies the interface through which a tree can be
   * traversed
   *
   * @tparam CLQUE
   */
  template <typename CLQUE>
  struct NodeVisitor {
    using Visitor = CliqueVisitor<CLQUE>;

    NodeVisitor(const Visitor& visitor) : visitor_(visitor) {}

    // This operator can be used for both pre and post order traversal since the
    // templated search does not care about the (existance of) return type in
    // the pre-vis case
    Data operator()(const boost::shared_ptr<CLQUE>& clique, Data&) {
      visitor_(clique);
      return Data{};
    }

    const Visitor visitor_;
  };

 public:
  /**
   * @brief Traverses a tree and calls the user defined visitor function at
   * every pre-visit.
   *
   *
   * @tparam CLIQUE
   * @param bayes_tree
   * @param pre_op_visitor
   */
  template <typename CLIQUE>
  static void depthFirstTraversal(const gtsam::BayesTree<CLIQUE>& bayes_tree,
                                  const CliqueVisitor<CLIQUE>& post_op_visitor,
                                  const CliqueVisitor<CLIQUE>& pre_op_visitor) {
    Data rootdata;
    NodeVisitor<CLIQUE> post_visitor(post_op_visitor);
    NodeVisitor<CLIQUE> pre_visitor(pre_op_visitor);
    // TODO: see gtsam/inference/inference-inst.h where the elimate function is
    // actually called they have both post and pre order traversal and i'm not
    // sure which one is used... the bayes tree paper talkes about solving where
    // a parse up from the leaves is computed and then a pass down where the
    // optimal assignment is retrieved I believe that the pass up (which must be
    // post-order) is also equivalent to the elimination ordering

    // NOTE: not using threaded version
    // TbbOpenMPMixedScope threadLimiter; // Limits OpenMP threads since we're
    // mixing TBB and OpenMP treeTraversal::DepthFirstForestParallel(bayesTree,
    // rootData, preVisitor, postVisitor);
    gtsam::treeTraversal::DepthFirstForest(bayes_tree, rootdata, pre_visitor,
                                           post_visitor);
  }

  template <typename CLIQUE>
  static void depthFirstTraversalEliminiationOrder(
      const gtsam::BayesTree<CLIQUE>& bayes_tree,
      const CliqueVisitor<CLIQUE>& visitor) {
    depthFirstTraversal(bayes_tree, visitor, NoopCliqueVisitor<CLIQUE>);
  }

  /// Gets keys in elimination order by traversing the tree in post-order
  template <typename CLIQUE>
  static gtsam::KeySet getEliminatonOrder(
      const gtsam::BayesTree<CLIQUE>& bayes_tree) {
    gtsam::KeySet keys;
    CliqueVisitor<CLIQUE> visitor =
        [&keys](const boost::shared_ptr<CLIQUE>& clique) -> void {
      auto conditional = clique->conditional();
      // it is FACTOR::const_iterator
      for (auto it = conditional->beginFrontals();
           it != conditional->endFrontals(); it++) {
        keys.insert(*it);
      }
    };

    depthFirstTraversalEliminiationOrder(bayes_tree, visitor);

    return keys;
  }

  template <typename CLIQUE>
  static gtsam::KeySet getLeafKeys(const gtsam::BayesTree<CLIQUE>& bayes_tree) {
    gtsam::KeySet keys;
    CliqueVisitor<CLIQUE> visitor =
        [&keys](const boost::shared_ptr<CLIQUE>& clique) -> void {
      if (clique->children.size() == 0u) {
        auto conditional = clique->conditional();
        // it is FACTOR::const_iterator
        for (auto it = conditional->beginFrontals();
             it != conditional->endFrontals(); it++) {
          keys.insert(*it);
        }
      }
    };

    depthFirstTraversalEliminiationOrder(bayes_tree, visitor);

    return keys;
  }
};

/**
 * @brief Calculate the max clique size and the average clique size given a
 * BayesTree
 *
 * @tparam CLIQUE
 * @param tree
 * @return std::pair<size_t, double>
 */
template <typename CLIQUE>
std::pair<size_t, double> getCliqueSize(const gtsam::BayesTree<CLIQUE>& isam2) {
  std::set<typename gtsam::BayesTree<CLIQUE>::sharedClique> isam2_cliques;
  auto nodes = isam2.nodes();
  for (const auto& it : nodes) {
    isam2_cliques.insert(it.second);
  }
  size_t max_clique_size = 0;
  size_t total_clique_size = 0;
  for (auto& isam2_clique : isam2_cliques) {
    size_t clique_size = isam2_clique->conditional()->frontals().size() +
                         isam2_clique->conditional()->parents().size();
    total_clique_size += clique_size;
    max_clique_size = std::max(max_clique_size, clique_size);
  }
  double avg_clique_size =
      (double)total_clique_size / (double)isam2_cliques.size();
  return std::make_pair(max_clique_size, avg_clique_size);
}

struct SparsityStats {
  gtsam::Matrix matrix;
  //! Number of zero elements
  size_t nr_zero_elements{0};
  //! Number total elements in the matrix (rows * cols)
  size_t nr_elements{0};
  //! Number of non-zero elements
  size_t nnz_elements{0};

  SparsityStats(){};

  /**
   * @brief Construct details from a matrix.
   *
   * Stat values are set upon construction
   *
   * @tparam Derived
   * @param M
   */
  template <typename Derived>
  SparsityStats(const Eigen::MatrixBase<Derived>& M)
      : matrix(M.template cast<double>()), nr_zero_elements(0), nr_elements(0) {
    nr_elements = M.rows() * M.cols();
    for (int i = 0; i < M.rows(); ++i) {
      for (int j = 0; j < M.cols(); ++j) {
        if (std::fabs(M(i, j)) < 1e-15) {
          nr_zero_elements++;
        } else {
          nnz_elements++;
        }
      }
    }
  }

  void saveOnOutputPath(const std::string& file_name) {
    save(getOutputFilePath(file_name));
  }

  void save(const std::string& file_path) {
    VLOG(20) << "Writing matrix matrix to file_path " << file_path;
    gtsam::save(matrix, "sparse_matrix", file_path);
  }
};

SparsityStats computeCholeskySparsityStats(
    gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
    const std::optional<gtsam::Ordering>& ordering = {});

SparsityStats computeHessianSparsityStats(
    gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
    const std::optional<gtsam::Ordering>& ordering = {});
SparsityStats computeJacobianSparsityStats(
    gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
    const std::optional<gtsam::Ordering>& ordering = {});

// TODO: we can make all these printing functions really nice in the future
// for now, just make for printing affected keys or regular printing
template <class CLIQUE>
void dotBayesTree(
    std::ostream& s, boost::shared_ptr<CLIQUE> clique,
    const gtsam::KeyFormatter& keyFormatter,
    const gtsam::FastMap<gtsam::Key, std::string>& colour_map = {},
    int parentnum = 0) {
  static int num = 0;
  bool first = true;
  std::stringstream out;
  out << num;
  std::string parent = out.str();
  parent += "[label=\"";

  // frontal key since this is what is used to add to nodes [j] (ie.e j is a
  // frontal key)
  bool keyInColourMap = false;
  std::string colour;

  for (auto key : clique->conditional_->frontals()) {
    if (!first) parent += ", ";
    first = false;
    parent += keyFormatter(key);

    keyInColourMap = colour_map.exists(key);
    // only for the last key? Assumes all keys have the same colour?
    if (keyInColourMap) colour = colour_map.at(key);
  }

  if (clique->parent()) {
    parent += " : ";
    s << parentnum << "->" << num << "\n";
  }

  first = true;
  for (auto parentKey : clique->conditional_->parents()) {
    if (!first) parent += ", ";
    first = false;
    parent += keyFormatter(parentKey);
  }
  //  parent += "\"];\n";

  if (keyInColourMap) {
    parent += "\",color=" + colour;
    parent += "];\n";
  } else {
    parent += "\"];\n";
  }

  s << parent;
  parentnum = num;

  for (auto c : clique->children) {
    num++;
    dotBayesTree(s, c, keyFormatter, colour_map, parentnum);
  }
}

template <class CLIQUE>
void saveBayesTree(
    const gtsam::BayesTree<CLIQUE>& tree, const std::string& filename,
    const gtsam::KeyFormatter& keyFormatter,
    const gtsam::FastMap<gtsam::Key, std::string> colour_map = {}) {
  std::ofstream of(filename.c_str());

  if (tree.roots().empty())
    throw std::invalid_argument(
        "the root of Bayes tree has not been initialized!");
  of << "digraph G{\n";
  for (const auto& root : tree.roots())
    dotBayesTree(of, root, keyFormatter, colour_map);
  of << "}";
  std::flush(of);
}

template <class CLIQUE>
std::pair<SparsityStats, cv::Mat> computeRFactor(
    const gtsam::BayesTree<CLIQUE>& bayes_tree) {
  size_t dim = 0;
  for (const auto& clique : bayes_tree) {
    dim += clique->conditional()->rows();
  }

  gtsam::Matrix full_R = gtsam::Matrix::Zero(dim, dim);

  // Step 2: Extract R blocks from the Bayes Tree and assemble them
  size_t rowStart = 0;
  for (const auto& clique : bayes_tree) {
    auto conditional = clique->conditional();
    if (conditional) {
      gtsam::Matrix R_block = conditional->R();  // Extract R block
      size_t blockSize = R_block.rows();

      // Insert R_block into the full R matrix
      full_R.block(rowStart, rowStart, blockSize, blockSize) = R_block;

      rowStart += blockSize;  // Move row index
    }
  }

  size_t nnz = 0u;
  cv::Mat R_img(cv::Size(full_R.cols(), full_R.rows()), CV_8UC3,
                cv::viz::Color::white());
  for (int i = 0; i < full_R.rows(); ++i) {
    for (int j = 0; j < full_R.cols(); ++j) {
      // only draw if non zero
      if (std::fabs(full_R(i, j)) > 1e-15) {
        R_img.at<cv::Vec3b>(i, j) = (cv::Vec3b)cv::viz::Color::black();
        nnz++;
      }
    }
  }
  return std::make_pair(SparsityStats{}, R_img);
}

std::pair<SparsityStats, cv::Mat> computeRFactor(
    gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
    const gtsam::Ordering ordering);

struct DrawBlockJacobiansOptions {
  cv::Size desired_size = cv::Size(480, 480);

  bool draw_label = true;    // if true, the labels for each vertical block (ie,
                             // the variable names) will be draw atop each block
  int text_box_height = 50;  // text box for label
  gtsam::KeyFormatter label_formatter =
      gtsam::DefaultKeyFormatter;  // function used to generate the label, if
                                   // draw_label== true

  //! Alias to a function that takes a key and returns a colour. Used to colour
  //! the non-zero elements of the Jacobian depending on the variable (block)
  using ColourSelector = std::function<cv::Scalar(gtsam::Key)>;

  inline static cv::Scalar DefaultColourSelector(gtsam::Key) {
    return cv::viz::Color::black();
  }

  ColourSelector colour_selector = DefaultColourSelector;

  /**
   * @brief Constructs a set of options which has the label formatter and colour
   * selector set to the Dyno sam versions.
   *
   * Specifcially the label formatter uses DynosamKeyFormatter and the colour
   * selector uses knowledge of the keys in our graph structructure to pick
   * colours for our variables
   *
   * @param options
   * @return DrawBlockJacobiansOptions
   */
  static DrawBlockJacobiansOptions makeDynoSamOptions(
      std::optional<DrawBlockJacobiansOptions> options = {}) {
    DrawBlockJacobiansOptions dyno_sam_options;
    if (options) dyno_sam_options = *options;

    // override label and colour formatter fucntion for dynosam
    dyno_sam_options.label_formatter = DynosamKeyFormatter;

    auto dyno_sam_colour_selector = [](gtsam::Key key) {
      return Color::uniqueId((size_t)gtsam::Symbol(key).chr()).bgra();
    };

    dyno_sam_options.colour_selector = dyno_sam_colour_selector;
    return dyno_sam_options;
  }
};

cv::Mat drawBlockJacobians(gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
                           const gtsam::Ordering& ordering,
                           const DrawBlockJacobiansOptions& options);

// template <typename T>
// void toGraphFileFormat(std::ostream& os, const T& t);

// template <typename T>
// std::ostream& seralizeFactorToGraphFileFormat(std::ostream& os, const T& t,
//                                               const std::string& tag) {
//   static_assert(is_gtsam_factor_v<T>);

//   // TAG <keys> VALUE
//   // where the value is provided by toGraphFileFormat and should be in the
//   form
//   // <measurement> <covaraince>

//   os << tag << " ";
//   const gtsam::Factor& factor = static_cast<const gtsam::Factor&>(t);
//   for (const gtsam::Key key : factor) {
//     os << key << " ";
//   }
//   toGraphFileFormat<T>(os, t);
//   os << '\n';
//   return os;
// }

// template <typename T>
// std::ostream& seralizeValueToGraphFileFormat(std::ostream& os, const T& t,
//                                              gtsam::Key key,
//                                              const std::string& tag) {
//   os << tag << " " << key << " ";
//   // TAG <key> VALUE
//   // where the value is provided by toGraphFileFormat and should be the value
//   // itself
//   toGraphFileFormat<T>(os, t);
//   os << '\n';
//   return os;
// }

}  // namespace factor_graph_tools

template <class GRAPH>
class FactorGraphManager {
 public:
  using FactorGraphType = GRAPH;
  using This = FactorGraphManager<FactorGraphType>;

  using FactorType = typename GRAPH::FactorType;
  using sharedFactor = typename GRAPH::sharedFactor;
  using value_type = sharedFactor;

  using iterator = typename GRAPH::iterator;
  using const_iterator = typename GRAPH::const_iterator;

  FactorGraphManager(const FactorGraphType graph) : graph_(graph) {}
  virtual ~FactorGraphManager() = default;

  FactorGraphType getAssociatedFactors(gtsam::Key key) const {
    std::vector<sharedFactor> factors;

    // while the manager is NOT a grpah, it does complete MOST of the concept
    // requirements of a gtsam::FactorGraph but providing iterator, shared
    // factor etc... type alias and std container definitions (begin and end) so
    // that we can use this with templated GRAPH functions
    const size_t num_connected =
        factor_graph_tools::getAssociatedFactors(factors, *this, key);

    if (num_connected < 1) {
      return FactorGraphType{};
    }

    FactorGraphType connected_factors;
    for (const auto& f : factors) {
      connected_factors += f;
    }
    return connected_factors;
  }

  virtual gtsam::GaussianFactorGraph::shared_ptr linearize() const = 0;

  const FactorGraphType& getGraph() const { return graph_; }

  gtsam::JacobianFactor constructJacobian(
      const gtsam::Ordering::OrderingType& ordering_type) {
    return constructJacobian(gtsam::Ordering::Create(ordering_type, graph_));
  }

  gtsam::JacobianFactor constructJacobian(const gtsam::Ordering& ordering) {
    auto gfg = this->linearize();
    return gtsam::JacobianFactor(*gfg, ordering);
  }

  cv::Mat drawBlockJacobian(
      const gtsam::Ordering& ordering,
      const factor_graph_tools::DrawBlockJacobiansOptions& options) const {
    return factor_graph_tools::drawBlockJacobians(this->linearize(), ordering,
                                                  options);
  }

  cv::Mat drawBlockJacobian(
      const gtsam::Ordering::OrderingType& ordering_type,
      const factor_graph_tools::DrawBlockJacobiansOptions& options) const {
    return drawBlockJacobian(gtsam::Ordering::Create(ordering_type, graph_),
                             options);
  }

  factor_graph_tools::SparsityStats computeJacobianSparsityStats(
      const gtsam::Ordering& ordering, bool save = false,
      const std::string& file = "jacobian.txt") const {
    factor_graph_tools::SparsityStats stats =
        factor_graph_tools::computeJacobianSparsityStats(this->linearize(),
                                                         ordering);

    if (save) stats.saveOnOutputPath(file);

    return stats;
  }

  factor_graph_tools::SparsityStats computeHessianSparsityStats(
      const gtsam::Ordering& ordering, bool save = false,
      const std::string& file = "hessian.txt") const {
    factor_graph_tools::SparsityStats stats =
        factor_graph_tools::computeHessianSparsityStats(this->linearize(),
                                                        ordering);

    if (save) stats.saveOnOutputPath(file);

    return stats;
  }

  // alows stl access
  const_iterator begin() const { return graph_.begin(); }
  const_iterator end() const { return graph_.end(); }

 protected:
  const FactorGraphType graph_;
};

class NonlinearFactorGraphManager
    : public FactorGraphManager<gtsam::NonlinearFactorGraph> {
 public:
  using Base = FactorGraphManager<gtsam::NonlinearFactorGraph>;

  NonlinearFactorGraphManager(const gtsam::NonlinearFactorGraph& graph,
                              const gtsam::Values& values)
      : Base(graph), values_(values) {}

  gtsam::GaussianFactorGraph::shared_ptr linearize() const override {
    return graph_.linearize(values_);
  }

  // strong dependancy on a factor graph constructed from THIS system
  // e.g. using the type of symbols defined in dynosam and the type of factors
  // we use the type of symbol and the characters here to determine the type of
  // factor and the USE of this factor (e.g. a between factor could be a
  // smoothing factor or an odom factor)
  // void writeDynosamGraphFile(const std::string& filename) const;

  // /**
  //  * @brief Constructs a factor_graph_tools::DrawBlockJacobiansOptions where
  //  the colour_selector function uses the type of value
  //  * (since the NonlinearFactorGraphManager knows the values) to colour the
  //  non-zero elements.
  //  *
  //  * @param options
  //  * @return factor_graph_tools::DrawBlockJacobiansOptions
  //  */
  // factor_graph_tools::DrawBlockJacobiansOptions
  // optionsByValue(std::optional<factor_graph_tools::DrawBlockJacobiansOptions>
  // options);

 protected:
  const gtsam::Values values_;
};

using namespace gtsam;

template <typename BayesTree>
class BayesTreeMarginalizationHelper {
 public:
  using Clique = typename BayesTree::Clique;
  using sharedClique = typename BayesTree::sharedClique;

  /**
   * This function identifies variables that need to be re-eliminated before
   * performing marginalization.
   *
   * Re-elimination is necessary for a clique containing marginalizable
   * variables if:
   *
   * 1. Some non-marginalizable variables appear before marginalizable ones
   *    in that clique;
   * 2. Or it has a child node depending on a marginalizable variable AND the
   *    subtree rooted at that child contains non-marginalizables.
   *
   * In addition, for any descendant node depending on a marginalizable
   * variable, if the subtree rooted at that descendant contains
   * non-marginalizable variables (i.e., it lies on a path from one of the
   * aforementioned cliques that require re-elimination to a node containing
   * non-marginalizable variables at the leaf side), then it also needs to
   * be re-eliminated.
   *
   * @param[in] bayesTree The Bayes tree
   * @param[in] marginalizableKeys Keys to be marginalized
   * @return Set of additional keys that need to be re-eliminated
   */
  static std::unordered_set<Key> gatherAdditionalKeysToReEliminate(
      const BayesTree& bayesTree, const KeyVector& marginalizableKeys) {
    const bool debug = true;  // ISDEBUG("BayesTreeMarginalizationHelper");

    std::unordered_set<const Clique*> additionalCliques =
        gatherAdditionalCliquesToReEliminate(bayesTree, marginalizableKeys);

    std::unordered_set<Key> additionalKeys;
    for (const Clique* clique : additionalCliques) {
      addCliqueToKeySet(clique, &additionalKeys);
    }

    if (debug) {
      std::cout << "BayesTreeMarginalizationHelper: Additional keys to "
                   "re-eliminate: ";
      for (const Key& key : additionalKeys) {
        std::cout << DynosamKeyFormatter(key) << " ";
      }
      std::cout << std::endl;
    }

    return additionalKeys;
  }

 protected:
  /**
   * This function identifies cliques that need to be re-eliminated before
   * performing marginalization.
   * See the docstring of @ref gatherAdditionalKeysToReEliminate().
   */
  static std::unordered_set<const Clique*> gatherAdditionalCliquesToReEliminate(
      const BayesTree& bayesTree, const KeyVector& marginalizableKeys) {
    std::unordered_set<const Clique*> additionalCliques;
    std::unordered_set<Key> marginalizableKeySet(marginalizableKeys.begin(),
                                                 marginalizableKeys.end());
    CachedSearch cachedSearch;

    // Check each clique that contains a marginalizable key
    for (const Clique* clique :
         getCliquesContainingKeys(bayesTree, marginalizableKeySet)) {
      // clique->print("Clique: ", DynosamKeyFormatter);
      if (additionalCliques.count(clique)) {
        // The clique has already been visited. This can happen when an
        // ancestor of the current clique also contain some marginalizable
        // varaibles and it's processed beore the current.
        continue;
      }

      if (needsReelimination(clique, marginalizableKeySet, &cachedSearch)) {
        // Add the current clique
        additionalCliques.insert(clique);

        // Then add the dependent cliques
        gatherDependentCliques(clique, marginalizableKeySet, &additionalCliques,
                               &cachedSearch);
      }
    }
    return additionalCliques;
  }

  /**
   * Gather the cliques containing any of the given keys.
   *
   * @param[in] bayesTree The Bayes tree
   * @param[in] keysOfInterest Set of keys of interest
   * @return Set of cliques that contain any of the given keys
   */
  static std::unordered_set<const Clique*> getCliquesContainingKeys(
      const BayesTree& bayesTree,
      const std::unordered_set<Key>& keysOfInterest) {
    std::unordered_set<const Clique*> cliques;
    for (const Key& key : keysOfInterest) {
      CHECK(bayesTree.nodes().exists(key))
          << "Key does not exist in bayesTree " << DynosamKeyFormatter(key);
      cliques.insert(bayesTree[key].get());
    }
    return cliques;
  }

  /**
   * A struct to cache the results of the below two functions.
   */
  struct CachedSearch {
    std::unordered_map<const Clique*, bool> wholeMarginalizableCliques;
    std::unordered_map<const Clique*, bool> wholeMarginalizableSubtrees;
  };

  /**
   * Check if all variables in the clique are marginalizable.
   *
   * Note we use a cache map to avoid repeated searches.
   */
  static bool isWholeCliqueMarginalizable(
      const Clique* clique, const std::unordered_set<Key>& marginalizableKeys,
      CachedSearch* cache) {
    auto it = cache->wholeMarginalizableCliques.find(clique);
    if (it != cache->wholeMarginalizableCliques.end()) {
      return it->second;
    } else {
      bool ret = true;
      for (Key key : clique->conditional()->frontals()) {
        if (!marginalizableKeys.count(key)) {
          ret = false;
          break;
        }
      }
      cache->wholeMarginalizableCliques.insert({clique, ret});
      return ret;
    }
  }

  /**
   * Check if all variables in the subtree are marginalizable.
   *
   * Note we use a cache map to avoid repeated searches.
   */
  static bool isWholeSubtreeMarginalizable(
      const Clique* subtree, const std::unordered_set<Key>& marginalizableKeys,
      CachedSearch* cache) {
    auto it = cache->wholeMarginalizableSubtrees.find(subtree);
    if (it != cache->wholeMarginalizableSubtrees.end()) {
      return it->second;
    } else {
      bool ret = true;
      if (isWholeCliqueMarginalizable(subtree, marginalizableKeys, cache)) {
        for (const sharedClique& child : subtree->children) {
          if (!isWholeSubtreeMarginalizable(child.get(), marginalizableKeys,
                                            cache)) {
            ret = false;
            break;
          }
        }
      } else {
        ret = false;
      }
      cache->wholeMarginalizableSubtrees.insert({subtree, ret});
      return ret;
    }
  }

  /**
   * Check if a clique contains variables that need reelimination due to
   * elimination ordering conflicts.
   *
   * @param[in] clique The clique to check
   * @param[in] marginalizableKeys Set of keys to be marginalized
   * @return true if any variables in the clique need re-elimination
   */
  static bool needsReelimination(
      const Clique* clique, const std::unordered_set<Key>& marginalizableKeys,
      CachedSearch* cache) {
    bool hasNonMarginalizableAhead = false;

    // Check each frontal variable in order
    for (Key key : clique->conditional()->frontals()) {
      if (marginalizableKeys.count(key)) {
        // If we've seen non-marginalizable variables before this one,
        // we need to reeliminate
        if (hasNonMarginalizableAhead) {
          return true;
        }

        // Check if any child depends on this marginalizable key and the
        // subtree rooted at that child contains non-marginalizables.
        for (const sharedClique& child : clique->children) {
          if (hasDependency(child.get(), key) &&
              !isWholeSubtreeMarginalizable(child.get(), marginalizableKeys,
                                            cache)) {
            return true;
          }
        }
      } else {
        hasNonMarginalizableAhead = true;
      }
    }
    return false;
  }

  /**
   * Gather all dependent nodes that lie on a path from the root clique
   * to a clique containing a non-marginalizable variable at the leaf side.
   *
   * @param[in] rootClique The root clique
   * @param[in] marginalizableKeys Set of keys to be marginalized
   */
  static void gatherDependentCliques(
      const Clique* rootClique,
      const std::unordered_set<Key>& marginalizableKeys,
      std::unordered_set<const Clique*>* additionalCliques,
      CachedSearch* cache) {
    std::vector<const Clique*> dependentChildren;
    dependentChildren.reserve(rootClique->children.size());
    for (const sharedClique& child : rootClique->children) {
      if (additionalCliques->count(child.get())) {
        // This child has already been visited. This can happen if the
        // child itself contains a marginalizable variable and it's
        // processed before the current rootClique.
        continue;
      }
      if (hasDependency(child.get(), marginalizableKeys)) {
        dependentChildren.push_back(child.get());
      }
    }
    gatherDependentCliquesFromChildren(dependentChildren, marginalizableKeys,
                                       additionalCliques, cache);
  }

  /**
   * A helper function for the above gatherDependentCliques().
   */
  static void gatherDependentCliquesFromChildren(
      const std::vector<const Clique*>& dependentChildren,
      const std::unordered_set<Key>& marginalizableKeys,
      std::unordered_set<const Clique*>* additionalCliques,
      CachedSearch* cache) {
    std::deque<const Clique*> descendants(dependentChildren.begin(),
                                          dependentChildren.end());
    while (!descendants.empty()) {
      const Clique* descendant = descendants.front();
      descendants.pop_front();

      // If the subtree rooted at this descendant contains non-marginalizables,
      // it must lie on a path from the root clique to a clique containing
      // non-marginalizables at the leaf side.
      if (!isWholeSubtreeMarginalizable(descendant, marginalizableKeys,
                                        cache)) {
        additionalCliques->insert(descendant);

        // Add children of the current descendant to the set descendants.
        for (const sharedClique& child : descendant->children) {
          if (additionalCliques->count(child.get())) {
            // This child has already been visited.
            continue;
          } else {
            descendants.push_back(child.get());
          }
        }
      }
    }
  }

  /**
   * Add all frontal variables from a clique to a key set.
   *
   * @param[in] clique Clique to add keys from
   * @param[out] additionalKeys Pointer to the output key set
   */
  static void addCliqueToKeySet(const Clique* clique,
                                std::unordered_set<Key>* additionalKeys) {
    for (Key key : clique->conditional()->frontals()) {
      additionalKeys->insert(key);
    }
  }

  /**
   * Check if the clique depends on the given key.
   *
   * @param[in] clique Clique to check
   * @param[in] key Key to check for dependencies
   * @return true if clique depends on the key
   */
  static bool hasDependency(const Clique* clique, Key key) {
    auto& conditional = clique->conditional();
    if (std::find(conditional->beginParents(), conditional->endParents(),
                  key) != conditional->endParents()) {
      return true;
    } else {
      return false;
    }
  }

  /**
   * Check if the clique depends on any of the given keys.
   */
  static bool hasDependency(const Clique* clique,
                            const std::unordered_set<Key>& keys) {
    auto& conditional = clique->conditional();
    for (auto it = conditional->beginParents(); it != conditional->endParents();
         ++it) {
      if (keys.count(*it)) {
        return true;
      }
    }

    return false;
  }
};
// BayesTreeMarginalizationHelper

class ISAM2Helper : public BayesTreeMarginalizationHelper<gtsam::ISAM2::Base> {
 public:
  using Base = BayesTreeMarginalizationHelper<gtsam::ISAM2::Base>;

  static bool marginalizeKeys(gtsam::ISAM2Result* result,
                              gtsam::ISAM2* smoother,
                              const gtsam::KeyVector& marginalize_keys);
  static void createOrderingConstraints(
      const gtsam::KeyVector& marginalizableKeys, const ISAM2& smoother,
      boost::optional<gtsam::FastMap<gtsam::Key, int> >& constrainedKeys);
};

}  // namespace dyno
