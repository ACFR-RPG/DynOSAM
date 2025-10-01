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

#include "dynosam_common/Types.hpp"
#include "dynosam_vision_common/Feature.hpp"
namespace dyno {
/// We divide the image into a grid of cells and try to find maximally one
/// feature per cell. This is to ensure good distribution of features in the
/// image.
/// https://github.com/uzh-rpg/rpg_svo_pro_open/blob/master/svo_common/include/svo/common/occupancy_grid_2d.h
class OccupandyGrid2D {
 public:
  using Grid = std::vector<bool>;
  using FeatureGrid = Keypoints;

  OccupandyGrid2D(int cell_size, int n_cols, int n_rows)
      : cell_size(cell_size),
        n_cols(n_cols),
        n_rows(n_rows),
        occupancy_(n_cols * n_rows, false),
        feature_occupancy_(n_cols * n_rows, Keypoint(0, 0)) {}

  OccupandyGrid2D(const OccupandyGrid2D& rhs)
      : cell_size(rhs.cell_size),
        n_cols(rhs.n_cols),
        n_rows(rhs.n_rows),
        occupancy_(rhs.occupancy_),
        feature_occupancy_(rhs.feature_occupancy_) {}

  ~OccupandyGrid2D() = default;

  inline static int getNCell(const int n_pixels, const int size) {
    return std::ceil(static_cast<double>(n_pixels) / static_cast<double>(size));
  }

  const int cell_size;
  const int n_cols;
  const int n_rows;
  Grid occupancy_;
  FeatureGrid feature_occupancy_;

  inline void reset() {
    std::fill(occupancy_.begin(), occupancy_.end(), false);
  }

  inline size_t size() { return occupancy_.size(); }

  inline bool empty() { return occupancy_.empty(); }

  inline bool isOccupied(const size_t cell_index) {
    CHECK_LT(cell_index, occupancy_.size());
    return occupancy_[cell_index];
  }

  inline void setOccupied(const size_t cell_index) {
    CHECK_LT(cell_index, occupancy_.size());
    occupancy_[cell_index] = true;
  }

  inline int numOccupied() const {
    return std::count(occupancy_.begin(), occupancy_.end(), true);
  }

  template <typename Derived>
  size_t getCellIndex(const Eigen::MatrixBase<Derived>& px) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    return std::floor(px(1) / cell_size) * n_cols +
           std::floor(px(0) / cell_size);
    //    return static_cast<size_t>((px(1)) / cell_size * n_cols +
    //                               (px(0)) / cell_size);
  }

  inline size_t getCellIndex(int x, int y, int scale = 1) const {
    return getCellIndex(Eigen::Vector2d(scale * x, scale * y));
    //    return static_cast<size_t>((scale * y) / cell_size * n_cols +
    //                               (scale * x) / cell_size);
  }

  inline void fillWithKeypoints(const Keypoints& keypoints) {
    // TODO(cfo): could be implemented using block operations.
    for (size_t i = 0; i < keypoints.size(); ++i) {
      const Keypoint& keypoint = keypoints.at(i);
      const int int_x = static_cast<int>(keypoint(0));
      const int int_y = static_cast<int>(keypoint(1));
      const size_t idx = getCellIndex(int_x, int_y, 1);
      occupancy_.at(idx) = true;
      feature_occupancy_.at(idx) = Keypoint(keypoint(0), keypoint(1));
    }
  }
};

using OccGrid2DPtr = std::shared_ptr<OccupandyGrid2D>;

}  // namespace dyno
