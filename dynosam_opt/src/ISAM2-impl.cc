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

#include "dynosam_opt/ISAM2-impl.hpp"

#include <gtsam/base/debug.h>
#include <gtsam/config.h>            // for GTSAM_USE_TBB
#include <gtsam/inference/Symbol.h>  // for selective linearization thresholds

#include <cassert>
#include <functional>
#include <limits>
#include <string>

using namespace std;
using namespace gtsam;

namespace dyno {

/* ************************************************************************* */
namespace internal {
inline static void optimizeInPlace(const ISAM2::sharedClique& clique,
                                   VectorValues* result) {
  // parents are assumed to already be solved and available in result
  result->update(clique->conditional()->solve(*result));

  // starting from the root, call optimize on each conditional
  for (const ISAM2::sharedClique& child : clique->children)
    optimizeInPlace(child, result);
}
}  // namespace internal

size_t DeltaImpl::UpdateGaussNewtonDelta(const ISAM2::Roots& roots,
                                         const gtsam::KeySet& replacedKeys,
                                         double wildfireThreshold,
                                         gtsam::VectorValues* delta) {
  size_t lastBacksubVariableCount;

  if (wildfireThreshold <= 0.0) {
    // Threshold is zero or less, so do a full recalculation
    for (const ISAM2::sharedClique& root : roots)
      internal::optimizeInPlace(root, delta);
    lastBacksubVariableCount = delta->size();

  } else {
    // Optimize with wildfire
    lastBacksubVariableCount = 0;
    for (const ISAM2::sharedClique& root : roots)
      lastBacksubVariableCount += optimizeWildfireNonRecursive(
          root, wildfireThreshold, replacedKeys, delta);  // modifies delta

#if !defined(NDEBUG) && defined(GTSAM_EXTRA_CONSISTENCY_CHECKS)
    for (gtsam::VectorValues::const_iterator key_delta = delta->begin();
         key_delta != delta->end(); ++key_delta) {
      assert((*delta)[key_delta->first].allFinite());
    }
#endif
  }

  return lastBacksubVariableCount;
}

/* ************************************************************************* */
namespace internal {
void updateRgProd(const ISAM2::sharedClique& clique,
                  const gtsam::KeySet& replacedKeys,
                  const gtsam::VectorValues& grad, gtsam::VectorValues* RgProd,
                  size_t* varsUpdated) {
  // Check if any frontal or separator keys were recalculated, if so, we need
  // update deltas and recurse to children, but if not, we do not need to
  // recurse further because of the running separator property.
  bool anyReplaced = false;
  for (gtsam::Key j : *clique->conditional()) {
    if (replacedKeys.exists(j)) {
      anyReplaced = true;
      break;
    }
  }

  if (anyReplaced) {
    // Update the current variable
    // Get VectorValues slice corresponding to current variables
    gtsam::Vector gR =
        grad.vector(KeyVector(clique->conditional()->beginFrontals(),
                              clique->conditional()->endFrontals()));
    gtsam::Vector gS =
        grad.vector(KeyVector(clique->conditional()->beginParents(),
                              clique->conditional()->endParents()));

    // Compute R*g and S*g for this clique
    gtsam::Vector RSgProd =
        clique->conditional()->R() * gR + clique->conditional()->S() * gS;

    // Write into RgProd vector
    gtsam::DenseIndex vectorPosition = 0;
    for (gtsam::Key frontal : clique->conditional()->frontals()) {
      gtsam::Vector& RgProdValue = (*RgProd)[frontal];
      RgProdValue = RSgProd.segment(vectorPosition, RgProdValue.size());
      vectorPosition += RgProdValue.size();
    }

    // Now solve the part of the Newton's method point for this clique
    // (back-substitution)
    // (*clique)->solveInPlace(deltaNewton);

    *varsUpdated += clique->conditional()->nrFrontals();

    // Recurse to children
    for (const ISAM2::sharedClique& child : clique->children) {
      updateRgProd(child, replacedKeys, grad, RgProd, varsUpdated);
    }
  }
}
}  // namespace internal

/* ************************************************************************* */
size_t DeltaImpl::UpdateRgProd(const ISAM2::Roots& roots,
                               const gtsam::KeySet& replacedKeys,
                               const gtsam::VectorValues& gradAtZero,
                               gtsam::VectorValues* RgProd) {
  // Update variables
  size_t varsUpdated = 0;
  for (const ISAM2::sharedClique& root : roots) {
    internal::updateRgProd(root, replacedKeys, gradAtZero, RgProd,
                           &varsUpdated);
  }

  return varsUpdated;
}

/* ************************************************************************* */
gtsam::VectorValues DeltaImpl::ComputeGradientSearch(
    const gtsam::VectorValues& gradAtZero, const gtsam::VectorValues& RgProd) {
  // Compute gradient squared-magnitude
  const double gradientSqNorm = gradAtZero.dot(gradAtZero);

  // Compute minimizing step size
  double RgNormSq = RgProd.vector().squaredNorm();
  double step = -gradientSqNorm / RgNormSq;

  // Compute steepest descent point
  return step * gradAtZero;
}

}  // namespace dyno
