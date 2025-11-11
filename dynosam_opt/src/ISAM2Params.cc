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

#include "dynosam_opt/ISAM2Params.hpp"

using namespace std;
using namespace gtsam;

namespace dyno {

/* ************************************************************************* */
string ISAM2DoglegParams::adaptationModeTranslator(
    const DoglegOptimizerImpl::TrustRegionAdaptationMode& adaptationMode)
    const {
  string s;
  switch (adaptationMode) {
    case DoglegOptimizerImpl::SEARCH_EACH_ITERATION:
      s = "SEARCH_EACH_ITERATION";
      break;
    case DoglegOptimizerImpl::ONE_STEP_PER_ITERATION:
      s = "ONE_STEP_PER_ITERATION";
      break;
    default:
      s = "UNDEFINED";
      break;
  }
  return s;
}

/* ************************************************************************* */
DoglegOptimizerImpl::TrustRegionAdaptationMode
ISAM2DoglegParams::adaptationModeTranslator(
    const string& adaptationMode) const {
  string s = adaptationMode;
  // convert to upper case
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  if (s == "SEARCH_EACH_ITERATION")
    return DoglegOptimizerImpl::SEARCH_EACH_ITERATION;
  if (s == "ONE_STEP_PER_ITERATION")
    return DoglegOptimizerImpl::ONE_STEP_PER_ITERATION;

  /* default is SEARCH_EACH_ITERATION */
  return DoglegOptimizerImpl::SEARCH_EACH_ITERATION;
}

/* ************************************************************************* */
ISAM2Params::Factorization ISAM2Params::factorizationTranslator(
    const string& str) {
  string s = str;
  // convert to upper case
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  if (s == "QR") return ISAM2Params::QR;
  if (s == "CHOLESKY") return ISAM2Params::CHOLESKY;

  /* default is CHOLESKY */
  return ISAM2Params::CHOLESKY;
}

/* ************************************************************************* */
string ISAM2Params::factorizationTranslator(
    const ISAM2Params::Factorization& value) {
  string s;
  switch (value) {
    case ISAM2Params::QR:
      s = "QR";
      break;
    case ISAM2Params::CHOLESKY:
      s = "CHOLESKY";
      break;
    default:
      s = "UNDEFINED";
      break;
  }
  return s;
}

}  // namespace dyno
