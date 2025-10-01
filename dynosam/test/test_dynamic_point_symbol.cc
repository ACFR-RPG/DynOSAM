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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <exception>

#include "dynosam_common/OptimizationDefinitions.hpp"

using namespace dyno;

TEST(CantorPairingFunction, basicPair) {
  // checking that examples provided on wikipedia are correct (ie
  // imeplementation is good)
  // https://en.wikipedia.org/wiki/Pairing_function#Examples
  //  EXPECT_EQ(CantorPairingFunction::pair({47, 32}), 3192);
}

TEST(CantorPairingFunction, basicZip) {
  // checking that examples provided on wikipedia are correct (ie
  // imeplementation is good)
  // https://en.wikipedia.org/wiki/Pairing_function#Examples

  // const auto z = CantorPairingFunction::unzip(1432);
  // EXPECT_EQ(z.first, 52);
  // EXPECT_EQ(z.second, 1);
}

TEST(CantorPairingFunction, testAccess) {
  const auto x = 15;
  const auto y = 79;

  const auto z = CantorPairingFunction::pair({x, y});
  const auto result = CantorPairingFunction::depair(z);
  EXPECT_EQ(result.first, x);
  EXPECT_EQ(result.second, y);
}

TEST(CantorPairingFunction, testReconstructionSpecialCase) {
  const auto x = 46528;  // tracklet fails at runtime. Special test for this
                         // case

  const auto z = CantorPairingFunction::pair({x, 1});
  const auto result = CantorPairingFunction::depair(z);
  EXPECT_EQ(result.first, x);
  EXPECT_EQ(result.second, 1);
}

TEST(DynamicPointSymbol, testReconstruction) {
  const auto x = 15;
  const auto y = 79;

  DynamicPointSymbol dps('m', x, y);
  gtsam::Symbol sym(dps);

  gtsam::Key key(sym);
  DynamicPointSymbol reconstructed_dps(key);
  EXPECT_EQ(dps, reconstructed_dps);
  EXPECT_EQ(x, reconstructed_dps.trackletId());
  EXPECT_EQ(y, reconstructed_dps.frameId());
}

TEST(DynamicPointSymbol, testReconstructionSpecialCase) {
  const TrackletId bad_id =
      46528;  // tracklet fails at runtime. Special test for this case
  // const TrackletId bad_id = 46000; //tracklet fails at runtime. Special test
  // for this case

  DynamicPointSymbol dps('m', bad_id, 0);
  gtsam::Symbol sym(dps);

  gtsam::Key key(sym);
  DynamicPointSymbol reconstructed_dps(key);
  EXPECT_EQ(dps, reconstructed_dps);
  EXPECT_EQ(bad_id, reconstructed_dps.trackletId());
}
