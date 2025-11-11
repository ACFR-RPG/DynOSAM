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

#include <gtsam/base/debug.h>
#include <gtsam/inference/JunctionTree-inst.h>  // We need the inst file because we'll make a special JT templated on ISAM2
#include <gtsam/inference/Symbol.h>
#include <gtsam/inference/VariableIndex.h>
#include <gtsam/linear/GaussianBayesTree.h>
#include <gtsam/linear/GaussianEliminationTree.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <string>
#include <utility>
#include <variant>

#include "dynosam_opt/ISAM2.hpp"
#include "dynosam_opt/ISAM2Result.hpp"

namespace dyno {

/* ************************************************************************* */
// Special BayesTree class that uses ISAM2 cliques - this is the result of
// reeliminating ISAM2 subtrees.
class ISAM2BayesTree : public ISAM2::Base {
 public:
  typedef ISAM2::Base Base;
  typedef ISAM2BayesTree This;
  typedef boost::shared_ptr<This> shared_ptr;

  ISAM2BayesTree() {}
};

/* ************************************************************************* */
// Special JunctionTree class that produces ISAM2 BayesTree cliques, used for
// reeliminating ISAM2 subtrees.
class ISAM2JunctionTree
    : public gtsam::JunctionTree<ISAM2BayesTree, gtsam::GaussianFactorGraph> {
 public:
  typedef JunctionTree<ISAM2BayesTree, gtsam::GaussianFactorGraph> Base;
  typedef ISAM2JunctionTree This;
  typedef boost::shared_ptr<This> shared_ptr;

  explicit ISAM2JunctionTree(
      const gtsam::GaussianEliminationTree& eliminationTree)
      : Base(eliminationTree) {}
};

struct DeltaImpl {
  struct PartialSolveResult {
    ISAM2::sharedClique bayesTree;
  };

  struct ReorderingMode {
    size_t nFullSystemVars;
    enum { /*AS_ADDED,*/ COLAMD } algorithm;
    enum { NO_CONSTRAINT, CONSTRAIN_LAST } constrain;
    std::optional<gtsam::FastMap<gtsam::Key, int>> constrainedKeys;
  };

  /**
   * Update the Newton's method step point, using wildfire
   */
  static size_t UpdateGaussNewtonDelta(const ISAM2::Roots& roots,
                                       const gtsam::KeySet& replacedKeys,
                                       double wildfireThreshold,
                                       gtsam::VectorValues* delta);

  /**
   * Update the RgProd (R*g) incrementally taking into account which variables
   * have been recalculated in \c replacedKeys.  Only used in Dogleg.
   */
  static size_t UpdateRgProd(const ISAM2::Roots& roots,
                             const gtsam::KeySet& replacedKeys,
                             const gtsam::VectorValues& gradAtZero,
                             gtsam::VectorValues* RgProd);

  /**
   * Compute the gradient-search point.  Only used in Dogleg.
   */
  static gtsam::VectorValues ComputeGradientSearch(
      const gtsam::VectorValues& gradAtZero, const gtsam::VectorValues& RgProd);
};

/* ************************************************************************* */
/**
 * Implementation functions for update method
 * All of the methods below have clear inputs and outputs, even if not
 * functional: iSAM2 is inherintly imperative.
 */
struct UpdateImpl {
  const ISAM2Params& params_;
  const ISAM2UpdateParams& updateParams_;
  UpdateImpl(const ISAM2Params& params, const ISAM2UpdateParams& updateParams)
      : params_(params), updateParams_(updateParams) {}

  // Provide some debugging information at the start of update
  static void LogStartingUpdate(const gtsam::NonlinearFactorGraph& newFactors,
                                const ISAM2& isam2) {
    gttic(pushBackFactors);
    const bool debug = ISDEBUG("ISAM2 update");
    const bool verbose = ISDEBUG("ISAM2 update verbose");

    if (verbose) {
      std::cout << "ISAM2::update\n";
      isam2.print("ISAM2: ");
    }

    if (debug || verbose) {
      newFactors.print("The new factors are: ");
    }
  }

  // Check relinearization if we're at the nth step, or we are using a looser
  // loop relinerization threshold.
  bool relinarizationNeeded(size_t update_count) const {
    return updateParams_.force_relinearize ||
           (params_.enableRelinearization &&
            update_count % params_.relinearizeSkip == 0);
  }

  // Add any new factors \Factors:=\Factors\cup\Factors'.
  void pushBackFactors(const gtsam::NonlinearFactorGraph& newFactors,
                       gtsam::NonlinearFactorGraph* nonlinearFactors,
                       gtsam::GaussianFactorGraph* linearFactors,
                       gtsam::VariableIndex* variableIndex,
                       gtsam::FactorIndices* newFactorsIndices,
                       gtsam::KeySet* keysWithRemovedFactors) const {
    gttic(pushBackFactors);

    // Perform the first part of the bookkeeping updates for adding new factors.
    // Adds them to the complete list of nonlinear factors, and populates the
    // list of new factor indices, both optionally finding and reusing empty
    // factor slots.
    *newFactorsIndices = nonlinearFactors->add_factors(
        newFactors, params_.findUnusedFactorSlots);

    // Remove the removed factors
    gtsam::NonlinearFactorGraph removedFactors;
    removedFactors.reserve(updateParams_.removeFactorIndices.size());
    for (const auto index : updateParams_.removeFactorIndices) {
      removedFactors.push_back(nonlinearFactors->at(index));
      nonlinearFactors->remove(index);
      if (params_.cacheLinearizedFactors) linearFactors->remove(index);
    }

    // Remove removed factors from the variable index so we do not attempt to
    // relinearize them
    variableIndex->remove(updateParams_.removeFactorIndices.begin(),
                          updateParams_.removeFactorIndices.end(),
                          removedFactors);
    *keysWithRemovedFactors = removedFactors.keys();
  }

  // Get keys from removed factors and new factors, and compute unused keys,
  // i.e., keys that are empty now and do not appear in the new factors.
  void computeUnusedKeys(const gtsam::NonlinearFactorGraph& newFactors,
                         const gtsam::VariableIndex& variableIndex,
                         const gtsam::KeySet& keysWithRemovedFactors,
                         gtsam::KeySet* unusedKeys) const {
    gttic(computeUnusedKeys);
    gtsam::KeySet removedAndEmpty;
    for (gtsam::Key key : keysWithRemovedFactors) {
      if (variableIndex.empty(key))
        removedAndEmpty.insert(removedAndEmpty.end(), key);
    }
    gtsam::KeySet newFactorSymbKeys = newFactors.keys();
    std::set_difference(removedAndEmpty.begin(), removedAndEmpty.end(),
                        newFactorSymbKeys.begin(), newFactorSymbKeys.end(),
                        std::inserter(*unusedKeys, unusedKeys->end()));
  }

  // Calculate nonlinear error
  void error(const gtsam::NonlinearFactorGraph& nonlinearFactors,
             const gtsam::Values& estimate,
             std::optional<double>* result) const {
    gttic(error);
    *result = nonlinearFactors.error(estimate);
  }

  // Mark linear update
  void gatherInvolvedKeys(const gtsam::NonlinearFactorGraph& newFactors,
                          const gtsam::NonlinearFactorGraph& nonlinearFactors,
                          const gtsam::KeySet& keysWithRemovedFactors,
                          gtsam::KeySet* markedKeys) const {
    gttic(gatherInvolvedKeys);
    *markedKeys = newFactors.keys();  // Get keys from new factors
    // Also mark keys involved in removed factors
    markedKeys->insert(keysWithRemovedFactors.begin(),
                       keysWithRemovedFactors.end());

    // Also mark any provided extra re-eliminate keys
    if (updateParams_.extraReelimKeys) {
      for (gtsam::Key key : *updateParams_.extraReelimKeys) {
        markedKeys->insert(key);
      }
    }

    // Also, keys that were not observed in existing factors, but whose affected
    // keys have been extended now (e.g. smart factors)
    if (updateParams_.newAffectedKeys) {
      for (const auto& factorAddedKeys : *updateParams_.newAffectedKeys) {
        const auto factorIdx = factorAddedKeys.first;
        const auto& affectedKeys = nonlinearFactors.at(factorIdx)->keys();
        markedKeys->insert(affectedKeys.begin(), affectedKeys.end());
      }
    }
  }

  // Update detail, unused, and observed keys from markedKeys
  void updateKeys(const gtsam::KeySet& markedKeys, ISAM2Result* result) const {
    gttic(updateKeys);
    // Observed keys for detailed results
    if (result->detail && params_.enableDetailedResults) {
      for (gtsam::Key key : markedKeys) {
        result->detail->variableStatus[key].isObserved = true;
      }
    }

    for (gtsam::Key index : markedKeys) {
      // Only add if not unused
      if (result->unusedKeys.find(index) == result->unusedKeys.end())
        // Make a copy of these, as we'll soon add to them
        result->observedKeys.push_back(index);
    }
  }

  static void CheckRelinearizationRecursiveMap(
      const gtsam::FastMap<char, gtsam::Vector>& thresholds,
      const gtsam::VectorValues& delta, const ISAM2::sharedClique& clique,
      gtsam::KeySet* relinKeys) {
    // Check the current clique for relinearization
    bool relinearize = false;
    for (gtsam::Key var : *clique->conditional()) {
      // Find the threshold for this variable type
      const gtsam::Vector& threshold =
          thresholds.find(gtsam::Symbol(var).chr())->second;

      const gtsam::Vector& deltaVar = delta[var];

      // Verify the threshold vector matches the actual variable size
      if (threshold.rows() != deltaVar.rows())
        throw std::invalid_argument(
            "Relinearization threshold vector dimensionality for '" +
            std::string(1, gtsam::Symbol(var).chr()) +
            "' passed into iSAM2 parameters does not match actual variable "
            "dimensionality.");

      // Check for relinearization
      if ((deltaVar.array().abs() > threshold.array()).any()) {
        relinKeys->insert(var);
        relinearize = true;
      }
    }

    // If this node was relinearized, also check its children
    if (relinearize) {
      for (const ISAM2::sharedClique& child : clique->children) {
        CheckRelinearizationRecursiveMap(thresholds, delta, child, relinKeys);
      }
    }
  }

  static void CheckRelinearizationRecursiveDouble(
      double threshold, const gtsam::VectorValues& delta,
      const ISAM2::sharedClique& clique, gtsam::KeySet* relinKeys) {
    // Check the current clique for relinearization
    bool relinearize = false;
    for (gtsam::Key var : *clique->conditional()) {
      double maxDelta = delta[var].lpNorm<Eigen::Infinity>();
      if (maxDelta >= threshold) {
        relinKeys->insert(var);
        relinearize = true;
      }
    }

    // If this node was relinearized, also check its children
    if (relinearize) {
      for (const ISAM2::sharedClique& child : clique->children) {
        CheckRelinearizationRecursiveDouble(threshold, delta, child, relinKeys);
      }
    }
  }

  /**
   * Find the set of variables to be relinearized according to
   * relinearizeThreshold. This check is performed recursively, starting at
   * the top of the tree. Once a variable in the tree does not need to be
   * relinearized, no further checks in that branch are performed. This is an
   * approximation of the Full version, designed to save time at the expense
   * of accuracy.
   * @param delta The linear delta to check against the threshold
   * @param keyFormatter Formatter for printing nonlinear keys during
   * debugging
   * @return The set of variable indices in delta whose magnitude is greater
   * than or equal to relinearizeThreshold
   */
  static gtsam::KeySet CheckRelinearizationPartial(
      const ISAM2::Roots& roots, const gtsam::VectorValues& delta,
      const ISAM2Params::RelinearizationThreshold& relinearizeThreshold) {
    gtsam::KeySet relinKeys;
    for (const ISAM2::sharedClique& root : roots) {
      if (std::holds_alternative<double>(relinearizeThreshold)) {
        CheckRelinearizationRecursiveDouble(
            std::get<double>(relinearizeThreshold), delta, root, &relinKeys);
      } else if (std::holds_alternative<gtsam::FastMap<char, gtsam::Vector>>(
                     relinearizeThreshold)) {
        CheckRelinearizationRecursiveMap(
            std::get<gtsam::FastMap<char, gtsam::Vector>>(relinearizeThreshold),
            delta, root, &relinKeys);
      }
    }
    return relinKeys;
  }

  /**
   * Find the set of variables to be relinearized according to
   * relinearizeThreshold. Any variables in the VectorValues delta whose
   * vector magnitude is greater than or equal to relinearizeThreshold are
   * returned.
   * @param delta The linear delta to check against the threshold
   * @param keyFormatter Formatter for printing nonlinear keys during
   * debugging
   * @return The set of variable indices in delta whose magnitude is greater
   * than or equal to relinearizeThreshold
   */
  static gtsam::KeySet CheckRelinearizationFull(
      const gtsam::VectorValues& delta,
      const ISAM2Params::RelinearizationThreshold& relinearizeThreshold) {
    gtsam::KeySet relinKeys;

    if (const double* threshold = std::get_if<double>(&relinearizeThreshold)) {
      for (const gtsam::VectorValues::KeyValuePair& key_delta : delta) {
        double maxDelta = key_delta.second.lpNorm<Eigen::Infinity>();
        if (maxDelta >= *threshold) relinKeys.insert(key_delta.first);
      }
    } else if (const gtsam::FastMap<char, gtsam::Vector>* thresholds =
                   std::get_if<gtsam::FastMap<char, gtsam::Vector>>(
                       &relinearizeThreshold)) {
      for (const gtsam::VectorValues::KeyValuePair& key_delta : delta) {
        const gtsam::Vector& threshold =
            thresholds->find(gtsam::Symbol(key_delta.first).chr())->second;
        if (threshold.rows() != key_delta.second.rows())
          throw std::invalid_argument(
              "Relinearization threshold vector dimensionality for '" +
              std::string(1, gtsam::Symbol(key_delta.first).chr()) +
              "' passed into iSAM2 parameters does not match actual variable "
              "dimensionality.");
        if ((key_delta.second.array().abs() > threshold.array()).any())
          relinKeys.insert(key_delta.first);
      }
    }

    return relinKeys;
  }

  // Mark keys in \Delta above threshold \beta:
  gtsam::KeySet gatherRelinearizeKeys(const ISAM2::Roots& roots,
                                      const gtsam::VectorValues& delta,
                                      const gtsam::KeySet& fixedVariables,
                                      gtsam::KeySet* markedKeys) const {
    gttic(gatherRelinearizeKeys);
    // J=\{\Delta_{j}\in\Delta|\Delta_{j}\geq\beta\}.
    gtsam::KeySet relinKeys =
        params_.enablePartialRelinearizationCheck
            ? CheckRelinearizationPartial(roots, delta,
                                          params_.relinearizeThreshold)
            : CheckRelinearizationFull(delta, params_.relinearizeThreshold);
    if (updateParams_.forceFullSolve)
      relinKeys = CheckRelinearizationFull(delta, 0.0);  // for debugging

    // Remove from relinKeys any keys whose linearization points are fixed
    for (gtsam::Key key : fixedVariables) {
      relinKeys.erase(key);
    }
    if (updateParams_.noRelinKeys) {
      for (gtsam::Key key : *updateParams_.noRelinKeys) {
        relinKeys.erase(key);
      }
    }

    // Add the variables being relinearized to the marked keys
    markedKeys->insert(relinKeys.begin(), relinKeys.end());
    return relinKeys;
  }

  // Record relinerization threshold keys in detailed results
  void recordRelinearizeDetail(const gtsam::KeySet& relinKeys,
                               ISAM2Result::DetailedResults* detail) const {
    if (detail && params_.enableDetailedResults) {
      for (gtsam::Key key : relinKeys) {
        detail->variableStatus[key].isAboveRelinThreshold = true;
        detail->variableStatus[key].isRelinearized = true;
      }
    }
  }

  // Mark all cliques that involve marked variables \Theta_{J} and all
  // their ancestors.
  void findFluid(const ISAM2::Roots& roots, const gtsam::KeySet& relinKeys,
                 gtsam::KeySet* markedKeys,
                 ISAM2Result::DetailedResults* detail) const {
    gttic(findFluid);
    for (const auto& root : roots)
      // add other cliques that have the marked ones in the separator
      root->findAll(relinKeys, markedKeys);

    // Relinearization-involved keys for detailed results
    if (detail && params_.enableDetailedResults) {
      gtsam::KeySet involvedRelinKeys;
      for (const auto& root : roots)
        root->findAll(relinKeys, &involvedRelinKeys);
      for (gtsam::Key key : involvedRelinKeys) {
        if (!detail->variableStatus[key].isAboveRelinThreshold) {
          detail->variableStatus[key].isRelinearizeInvolved = true;
          detail->variableStatus[key].isRelinearized = true;
        }
      }
    }
  }

  // Linearize new factors
  void linearizeNewFactors(const gtsam::NonlinearFactorGraph& newFactors,
                           const gtsam::Values& theta,
                           size_t numNonlinearFactors,
                           const gtsam::FactorIndices& newFactorsIndices,
                           gtsam::GaussianFactorGraph* linearFactors) const {
    gttic(linearizeNewFactors);
    auto linearized = newFactors.linearize(theta);
    if (params_.findUnusedFactorSlots) {
      linearFactors->resize(numNonlinearFactors);
      for (size_t i = 0; i < newFactors.size(); ++i)
        (*linearFactors)[newFactorsIndices[i]] = (*linearized)[i];
    } else {
      linearFactors->push_back(*linearized);
    }
    assert(linearFactors->size() == numNonlinearFactors);
  }

  void augmentVariableIndex(const gtsam::NonlinearFactorGraph& newFactors,
                            const gtsam::FactorIndices& newFactorsIndices,
                            gtsam::VariableIndex* variableIndex) const {
    gttic(augmentVariableIndex);
    // Augment the variable index with the new factors
    if (params_.findUnusedFactorSlots)
      variableIndex->augment(newFactors, newFactorsIndices);
    else
      variableIndex->augment(newFactors);

    // Augment it with existing factors which now affect to more variables:
    if (updateParams_.newAffectedKeys) {
      for (const auto& factorAddedKeys : *updateParams_.newAffectedKeys) {
        const auto factorIdx = factorAddedKeys.first;
        variableIndex->augmentExistingFactor(factorIdx, factorAddedKeys.second);
      }
    }
  }

  static void LogRecalculateKeys(const ISAM2Result& result) {
    const bool debug = ISDEBUG("ISAM2 recalculate");

    if (debug) {
      std::cout << "markedKeys: ";
      for (const gtsam::Key key : result.markedKeys) {
        std::cout << key << " ";
      }
      std::cout << std::endl;
      std::cout << "observedKeys: ";
      for (const gtsam::Key key : result.observedKeys) {
        std::cout << key << " ";
      }
      std::cout << std::endl;
    }
  }

  static gtsam::FactorIndexSet GetAffectedFactors(
      const gtsam::KeyList& keys, const gtsam::VariableIndex& variableIndex) {
    gttic(GetAffectedFactors);
    gtsam::FactorIndexSet indices;
    for (const gtsam::Key key : keys) {
      const gtsam::FactorIndices& factors(variableIndex[key]);
      indices.insert(factors.begin(), factors.end());
    }
    return indices;
  }

  // find intermediate (linearized) factors from cache that are passed into
  // the affected area
  static gtsam::GaussianFactorGraph GetCachedBoundaryFactors(
      const ISAM2::Cliques& orphans) {
    gtsam::GaussianFactorGraph cachedBoundary;

    for (const auto& orphan : orphans) {
      // retrieve the cached factor and add to boundary
      cachedBoundary.push_back(orphan->cachedFactor());
    }

    return cachedBoundary;
  }
};

}  // namespace dyno
