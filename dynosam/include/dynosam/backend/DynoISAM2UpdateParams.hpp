/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    DynoISAM2UpdateParams.h
 * @brief   Class that stores extra params for ISAM2::update()
 * @author  Michael Kaess, Richard Roberts, Frank Dellaert, Jose Luis Blanco
 */

// \callgraph

#pragma once

#include "dynosam/backend/DynoISAM2Result.hpp" //FactorIndices
#include <functional>


#include <gtsam/base/FastList.h>
#include <gtsam/dllexport.h>              // GTSAM_EXPORT
#include <gtsam/inference/Key.h>          // Key, KeySet
#include <gtsam/inference/Ordering.h>
#include <optional>

namespace dyno {

using namespace gtsam;

class DynoISAM2UpdateParams;

using ConstructOrdering = std::function<gtsam::Ordering(
    const DynoISAM2UpdateParams&,
    const DynoISAM2Result&,
    const gtsam::KeySet& /*affectedKeysSet*/,
    const gtsam::VariableIndex&)>;

/**
 * @ingroup isam2
 * This struct is used by ISAM2::update() to pass additional parameters to
 * give the user a fine-grained control on how factors and relinearized, etc.
 */
struct DynoISAM2UpdateParams {
  DynoISAM2UpdateParams() = default;

  /** Indices of factors to remove from system (default: empty) */
  FactorIndices removeFactorIndices;

  /** An optional map of keys to group labels, such that a variable can be
   * constrained to a particular grouping in the BayesTree */
  std::optional<FastMap<Key, int>> constrainedKeys;

  /** An optional set of nonlinear keys that iSAM2 will hold at a constant
   * linearization point, regardless of the size of the linear delta */
  std::optional<FastList<Key>> noRelinKeys;

  /** An optional set of nonlinear keys that iSAM2 will re-eliminate, regardless
   * of the size of the linear delta. This allows the provided keys to be
   * reordered. */
  std::optional<FastList<Key>> extraReelimKeys;

  /** Relinearize any variables whose delta magnitude is sufficiently large
   * (Params::relinearizeThreshold), regardless of the relinearization
   * interval (Params::relinearizeSkip). */
  bool force_relinearize{false};

  /** An optional set of new Keys that are now affected by factors,
   * indexed by factor indices (as returned by ISAM2::update()).
   * Use when working with smart factors. For example:
   *  - Timestamp `i`: ISAM2::update() called with a new smart factor depending
   *    on Keys `X(0)` and `X(1)`. It returns that the factor index for the new
   *    smart factor (inside ISAM2) is `13`.
   *  - Timestamp `i+1`: The same smart factor has been augmented to now also
   *    depend on Keys `X(2)`, `X(3)`. Next call to ISAM2::update() must include
   *    its `newAffectedKeys` field with the map `13 -> {X(2), X(3)}`.
   */
  std::optional<FastMap<FactorIndex, KeySet>> newAffectedKeys;

  /** By default, iSAM2 uses a wildfire update scheme that stops updating when
   * the deltas become too small down in the tree. This flagg forces a full
   * solve instead. */
  bool forceFullSolve{false};

  std::optional<ConstructOrdering> incrementalOrdering;
  std::optional<ConstructOrdering> batchOrdering;

  inline void setIncrementalOrderingFunction(const ConstructOrdering& orderingFunction) { incrementalOrdering = orderingFunction; }
  inline void setBatchOrderingFunction(const ConstructOrdering& orderingFunction) { batchOrdering = orderingFunction; }

  void setOrderingFunctions(const ConstructOrdering& orderingFunction) {
    incrementalOrdering = orderingFunction;
    batchOrdering = orderingFunction;
  }


};

}  // namespace dyno
