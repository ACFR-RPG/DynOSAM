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

/********************************************************************************
 Copyright 2017 Autonomous Systems Lab, ETH Zurich, Switzerland

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*********************************************************************************/

/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   Accumulator.h
 * @brief  For accumulating statistics.
 * @author Antoni Rosinol
 */

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace dyno {

namespace utils {

static constexpr int kInfiniteWindowSize = std::numeric_limits<int>::max();

// If the window size is set to -1, the vector will grow infinitely, otherwise,
// the vector has a fixed size.
template <typename SampleType, typename SumType, int WindowSize>
class Accumulator {
 public:
  using This = Accumulator<SampleType, SumType, WindowSize>;
  Accumulator()
      : sample_index_(0),
        total_samples_(0),
        sum_(0),
        window_sum_(0),
        min_(std::numeric_limits<SampleType>::max()),
        max_(std::numeric_limits<SampleType>::lowest()),
        most_recent_(0) {
    CHECK_GT(WindowSize, 0);
    if (WindowSize < kInfiniteWindowSize) {
      samples_.reserve(WindowSize);
    }
  }

  /* ------------------------------------------------------------------------ */
  void Add(SampleType sample) {
    most_recent_ = sample;
    if (sample_index_ < WindowSize) {
      samples_.push_back(sample);
      window_sum_ += sample;
      ++sample_index_;
    } else {
      SampleType& oldest = samples_.at(sample_index_++ % WindowSize);
      window_sum_ += sample - oldest;
      oldest = sample;
    }
    sum_ += sample;
    ++total_samples_;
    if (sample > max_) {
      max_ = sample;
    }
    if (sample < min_) {
      min_ = sample;
    }
  }

  /* ------------------------------------------------------------------------ */
  int total_samples() const { return total_samples_; }

  /* ------------------------------------------------------------------------ */
  SumType sum() const { return sum_; }

  /* ------------------------------------------------------------------------ */
  SumType Mean() const {
    return (total_samples_ < 1) ? 0.0
                                : sum_ / static_cast<SumType>(total_samples_);
  }

  /* ------------------------------------------------------------------------ */
  // Rolling mean is only used for fixed sized data for now. We don't need this
  // function for our infinite accumulator at this point.
  SumType RollingMean() const {
    if (WindowSize < kInfiniteWindowSize) {
      return window_sum_ / std::min(sample_index_, WindowSize);
    } else {
      return Mean();
    }
  }

  /* ------------------------------------------------------------------------ */
  SampleType GetMostRecent() const { return most_recent_; }

  /* ------------------------------------------------------------------------ */
  SumType max() const { return max_; }

  /* ------------------------------------------------------------------------ */
  SumType min() const { return min_; }

  /* ------------------------------------------------------------------------ */
  inline SumType median() const {
    CHECK_GT(samples_.size(), 0);

    return samples_.at(std::ceil(samples_.size() / 2) - 1);
  }

  /* ------------------------------------------------------------------------ */
  // First quartile.
  inline SumType q1() const {
    CHECK_GT(samples_.size(), 0);
    return samples_.at(std::ceil(samples_.size() / 4) - 1);
  }

  /* ------------------------------------------------------------------------ */
  // Third quartile.
  inline SumType q3() const {
    CHECK_GT(samples_.size(), 0);
    return samples_.at(std::ceil(samples_.size() * 3 / 4) - 1);
  }

  /* ------------------------------------------------------------------------ */
  SumType LazyVariance() const {
    if (samples_.size() < 2) {
      return 0.0;
    }

    SumType var = static_cast<SumType>(0.0);
    SumType mean = RollingMean();

    for (unsigned int i = 0; i < samples_.size(); ++i) {
      var += (samples_[i] - mean) * (samples_[i] - mean);
    }

    var /= samples_.size() - 1;
    return var;
  }

  /* ------------------------------------------------------------------------ */
  SumType StandardDeviation() const { return std::sqrt(LazyVariance()); }

  // TODO: needs tests
  //  only includes values that are within mean +/- threshold * std
  This OutlierRejectionStd(double threshold) {
    const auto mean = Mean();
    const auto std = StandardDeviation();

    const auto min_value = mean - threshold * std;
    const auto max_value = mean + threshold * std;

    This filtered;
    for (auto value : *this) {
      if (value > min_value && value < max_value) {
        filtered.Add(value);
      }
    }

    return filtered;
  }

  /* ------------------------------------------------------------------------ */
  const std::vector<SampleType>& GetAllSamples() const { return samples_; }

  typename std::vector<SampleType>::iterator begin() {
    return samples_.begin();
  }
  const typename std::vector<SampleType>::iterator begin() const {
    return samples_.begin();
  }

  typename std::vector<SampleType>::iterator end() { return samples_.end(); }
  const typename std::vector<SampleType>::iterator end() const {
    return samples_.end();
  }

  const typename std::vector<SampleType>::const_iterator cbegin() const {
    return samples_.cbegin();
  }
  const typename std::vector<SampleType>::const_iterator cend() const {
    return samples_.cend();
  }

 private:
  std::vector<SampleType> samples_;
  int sample_index_;
  int total_samples_;
  SumType sum_;
  SumType window_sum_;
  SampleType min_;
  SampleType max_;
  SampleType most_recent_;
};

typedef Accumulator<double, double, kInfiniteWindowSize> Accumulatord;

}  // namespace utils

}  // namespace dyno
