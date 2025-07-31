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

#include <gtsam/linear/GaussianBayesTree.h>
#include <gtsam/nonlinear/DoglegOptimizerImpl.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <optional>
#include <string>
#include <vector>

namespace dyno {

struct ISAM2Result {
  /** The nonlinear error of all of the factors, \a including new factors and
   * variables added during the current call to ISAM2::update().  This error is
   * calculated using the following variable values:
   * \li Pre-existing variables will be evaluated by combining their
   * linearization point before this call to update, with their partial linear
   * delta, as computed by ISAM2::calculateEstimate().
   * \li New variables will be evaluated at their initialization points passed
   * into the current call to update.
   * \par Note: This will only be computed if
   * ISAM2Params::evaluateNonlinearError is set to \c true, because there is
   * some cost to this computation.
   */
  std::optional<double> errorBefore;

  /** The nonlinear error of all of the factors computed after the current
   * update, meaning that variables above the relinearization threshold
   * (ISAM2Params::relinearizeThreshold) have been relinearized and new
   * variables have undergone one linear update.  Variable values are
   * again computed by combining their linearization points with their
   * partial linear deltas, by ISAM2::calculateEstimate().
   * \par Note: This will only be computed if
   * ISAM2Params::evaluateNonlinearError is set to \c true, because there is
   * some cost to this computation.
   */
  std::optional<double> errorAfter;

  /** The number of variables that were relinearized because their linear
   * deltas exceeded the reslinearization threshold
   * (ISAM2Params::relinearizeThreshold), combined with any additional
   * variables that had to be relinearized because they were involved in
   * the same factor as a variable above the relinearization threshold.
   * On steps where no relinearization is considered
   * (see ISAM2Params::relinearizeSkip), this count will be zero.
   */
  size_t variablesRelinearized;

  /** The number of variables that were reeliminated as parts of the Bayes'
   * Tree were recalculated, due to new factors.  When loop closures occur,
   * this count will be large as the new loop-closing factors will tend to
   * involve variables far away from the root, and everything up to the root
   * will be reeliminated.
   */
  size_t variablesReeliminated;

  /** The number of factors that were included in reelimination of the Bayes'
   * tree. */
  size_t factorsRecalculated;

  /** The number of cliques in the Bayes' Tree */
  size_t cliques;

  /** The indices of the newly-added factors, in 1-to-1 correspondence with the
   * factors passed as \c newFactors to ISAM2::update().  These indices may be
   * used later to refer to the factors in order to remove them.
   */
  gtsam::FactorIndices newFactorsIndices;

  /** Unused keys, and indices for unused keys,
   * i.e., keys that are empty now and do not appear in the new factors.
   */
  gtsam::KeySet unusedKeys;

  /** keys for variables that were observed, i.e., not unused. */
  gtsam::KeyVector observedKeys;

  /** Keys of variables that had factors removed. */
  gtsam::KeySet keysWithRemovedFactors;

  /** All keys that were marked during the update process. */
  gtsam::KeySet markedKeys;

  /** If recalculateBatch was used to solve the system */
  bool isBatch;

  size_t newVariables;
  size_t involvedVariables;
  size_t onlyRelinearizedVariables;
  size_t fluidVariables;

  /**
   * A struct holding detailed results, which must be enabled with
   * ISAM2Params::enableDetailedResults.
   */
  struct DetailedResults {
    /** The status of a single variable, this struct is stored in
     * DetailedResults::variableStatus */
    struct VariableStatus {
      /** Whether the variable was just reeliminated, due to being relinearized,
       * observed, new, or on the path up to the root clique from another
       * reeliminated variable. */
      bool isReeliminated;
      bool isAboveRelinThreshold;  ///< Whether the variable was just
                                   ///< relinearized due to being above the
                                   ///< relinearization threshold
      bool isRelinearizeInvolved;  ///< Whether the variable was below the
                                   ///< relinearization threshold but was
                                   ///< relinearized by being involved in a
                                   ///< factor with a variable above the
                                   ///< relinearization threshold
      bool isRelinearized;  /// Whether the variable was relinearized, either by
                            /// being above the relinearization threshold or by
                            /// involvement.
      bool isObserved;      ///< Whether the variable was just involved in new
                            ///< factors
      bool isNew;           ///< Whether the variable itself was just added
      bool inRootClique;    ///< Whether the variable is in the root clique
      VariableStatus()
          : isReeliminated(false),
            isAboveRelinThreshold(false),
            isRelinearizeInvolved(false),
            isRelinearized(false),
            isObserved(false),
            isNew(false),
            inRootClique(false) {}
    };

    using StatusMap = gtsam::FastMap<gtsam::Key, VariableStatus>;

    /// The status of each variable during this update, see VariableStatus.
    StatusMap variableStatus;
  };

  /** Detailed results, if enabled by ISAM2Params::enableDetailedResults.  See
   * Detail for information about the results data stored here. */
  std::optional<DetailedResults> detail;

  explicit ISAM2Result(bool enableDetailedResults = false) {
    if (enableDetailedResults) detail = DetailedResults();
  }

  /// Return pointer to detail, 0 if no detail requested
  DetailedResults* details() {
    if (detail.has_value()) {
      return &(*detail);
    } else {
      return nullptr;
    }
  }

  /// Print results
  void print(const std::string str = "") const {
    using std::cout;
    cout << str << "  Reelimintated: " << variablesReeliminated
         << "  Relinearized: " << variablesRelinearized
         << "  Cliques: " << cliques << std::endl;
  }

  /** Getters and Setters */
  size_t getVariablesRelinearized() const { return variablesRelinearized; }
  size_t getVariablesReeliminated() const { return variablesReeliminated; }
  gtsam::FactorIndices getNewFactorsIndices() const {
    return newFactorsIndices;
  }
  size_t getCliques() const { return cliques; }
  double getErrorBefore() const {
    return errorBefore ? *errorBefore : std::nan("");
  }
  double getErrorAfter() const {
    return errorAfter ? *errorAfter : std::nan("");
  }
};

}  // namespace dyno

#include <gtsam/nonlinear/ISAM2Result.h>

template <>
inline bool dyno::convert(const dyno::ISAM2Result& input,
                          gtsam::ISAM2Result& output) {
  // TODO: for now just need the new factor indices...
  output.newFactorsIndices = input.newFactorsIndices;
  return true;
}
