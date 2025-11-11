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

#pragma once

#include "dynosam/dataprovider/DatasetProvider.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {

// depth, motion masks, gt
using OMDDatasetProvider =
    DynoDatasetProvider<cv::Mat, cv::Mat, GroundTruthInputPacket>;

/**
 * @brief Data loader for the oxford multi motion (omd) datasets as provided at:
 * https://robotic-esp.com/datasets/omd/#tools-id
 *
 */
class OMDDataLoader : public OMDDatasetProvider {
 public:
  /**
   * @brief Construct a new Cluster Slam Data Loader object
   *
   * Datasset is ordered as per the
   * https://robotic-esp.com/datasets/omd/#tools-id and the dataset path is
   * /path/to/dataset/
   *  /rgbd
   *  /raw_depth
   *  /vicon.csv
   *  ....
   *  /optical_flow
   *  /instance_masks
   *
   * The last two folders are not provided in the original dataset and are
   * pre-processed using DynoSAM-Preprocessing
   *
   * @param dataset_path Top left path to the dataset eg /path/to/omd, where
   */
  OMDDataLoader(const fs::path& dataset_path);

  // we can get the camera params from this dataset, so overload the function!
  // returns camera params from camera1
  CameraParams::Optional getCameraParams() const override {
    return left_camera_params_;
  }

 private:
  CameraParams left_camera_params_;
};

}  // namespace dyno
