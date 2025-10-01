/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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
#include <limits>

#include "dynosam_common/utils/Histogram.hpp"

using namespace dyno;

TEST(Histogram, testConstructors) {
  Histogram hist(bh::make_histogram(bh::axis::regular<>(6, -1.0, 2.0)));
  Histogram hist1(bh::make_histogram(bh::axis::regular<>(6, -1.0, 2.0),
                                     bh::axis::variable<>({0, 1, 3, 5, 7})));
  Histogram hist2(bh::make_histogram(bh::axis::variable<>(
      {0.0, 1.0, 3.0, 5.0, 7.0, std::numeric_limits<double>::max()})));

  Histogram hist3(bh::make_histogram(bh::axis::variable<>(
      {0.0, 1.0, 3.0, 5.0, 7.0, std::numeric_limits<double>::max()})));

  EXPECT_EQ(hist.histogram_.rank(), 1);
  EXPECT_EQ(hist1.histogram_.rank(), 2);

  auto data = {-0.5, 1.1, 0.3, 1.7, 3.2, 4.0, 4.6, 4.0, 7.0, 9.0, 8.1};
  std::for_each(data.begin(), data.end(), std::ref(hist2.histogram_));

  auto data1 = {1.2, 3.3, 4.0};
  std::for_each(data1.begin(), data1.end(), std::ref(hist3.histogram_));

  LOG(INFO) << hist3.toString();

  hist3.histogram_ += hist2.histogram_;
  LOG(INFO) << hist3.toString();
  LOG(INFO) << hist2.toString();

  using json = nlohmann::json;
  json hist_json;
  to_json(hist_json, hist2);
  LOG(INFO) << hist_json;
}
