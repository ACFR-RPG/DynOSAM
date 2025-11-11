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

#include <gtsam/base/FastList.h>
#include <gtsam/dllexport.h>  // GTSAM_EXPORT
#include <gtsam/inference/FactorGraph.h>
#include <gtsam/inference/Key.h>  // Key, KeySet
#include <gtsam/nonlinear/ISAM2UpdateParams.h>
// #include <gtsam/nonlinear/ISAM2Result.h>  //FactorIndices
#include <optional>

namespace dyno {

struct ISAM2UpdateParams {
  ISAM2UpdateParams() = default;

  /** Indices of factors to remove from system (default: empty) */
  gtsam::FactorIndices removeFactorIndices;

  /** An optional map of keys to group labels, such that a variable can be
   * constrained to a particular grouping in the BayesTree */
  std::optional<gtsam::FastMap<gtsam::Key, int>> constrainedKeys;

  /** An optional set of nonlinear keys that iSAM2 will hold at a constant
   * linearization point, regardless of the size of the linear delta */
  std::optional<gtsam::FastList<gtsam::Key>> noRelinKeys;

  /** An optional set of nonlinear keys that iSAM2 will re-eliminate, regardless
   * of the size of the linear delta. This allows the provided keys to be
   * reordered. */
  std::optional<gtsam::FastList<gtsam::Key>> extraReelimKeys;

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
  std::optional<gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet>>
      newAffectedKeys;

  /** By default, iSAM2 uses a wildfire update scheme that stops updating when
   * the deltas become too small down in the tree. This flagg forces a full
   * solve instead. */
  bool forceFullSolve{false};
};

namespace {

// TODO: gross and only needed because gtsam version we're on still uses boost!!
template <typename T>
std::optional<T> boostOptToStd(const boost::optional<T>& opt) {
  if (opt.has_value()) {
    return std::optional<T>(*opt);
  } else {
    return std::nullopt;
  }
}
}  // namespace

}  // namespace dyno

template <>
inline bool dyno::convert(const gtsam::ISAM2UpdateParams& input,
                          dyno::ISAM2UpdateParams& output) {
  output.removeFactorIndices = input.removeFactorIndices;
  output.constrainedKeys = boostOptToStd(input.constrainedKeys);
  output.noRelinKeys = boostOptToStd(input.noRelinKeys);
  output.extraReelimKeys = boostOptToStd(input.extraReelimKeys);
  output.force_relinearize = input.force_relinearize;
  output.newAffectedKeys = boostOptToStd(input.newAffectedKeys);
  output.forceFullSolve = input.forceFullSolve;
  return true;
}
