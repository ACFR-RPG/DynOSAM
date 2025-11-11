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

#include "dynosam/dataprovider/DatasetProvider.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {

// depth, motion mask, semantic mask and gt
using VirtualKittiDatasetProvider =
    DynoDatasetProvider<cv::Mat, cv::Mat, cv::Mat, GroundTruthInputPacket>;

/**
 * A note on ground truth track ID indexing - VirtualKitti starts their
 * trackID's from 0 which is annoying when handling the trackID's within a
 * semantic/motion image mask since we normally expect the id's to be the exact
 * pixel values (where 0 is background).
 *
 * Instead we increase the actual trackID index to be the ground truth one (eg
 * the one provided in the poses.txt file) + 1 so we can index the trackId's
 * directly in the instance mask which will line up with the ones stored in the
 * ground truth packets (after we have +1 to their index)
 *
 */

class VirtualKittiDataLoader : public VirtualKittiDatasetProvider {
 public:
  struct Params {
    std::string scene;       // eg Scene01
    std::string scene_type;  // eg clone, fog...
    MaskType mask_type = MaskType::MOTION;

    static Params fromYaml(const std::string& params_folder);
  };

  CameraParams::Optional getCameraParams() const override {
    return camera_params_;
  }

  // expect to be the top level where the folders undearneath are in the form
  // vkitti_2.0.3_depth... (or as in the download...)
  VirtualKittiDataLoader(const fs::path& dataset_path, const Params& params);

 private:
  void setCameraParams();

 private:
  const Params params_;

  const std::string v_depth_folder = "vkitti_2.0.3_depth";
  const std::string v_forward_flow_folder = "vkitti_2.0.3_forwardFlow";
  const std::string v_backward_flow_folder = "vkitti_2.0.3_backwardFlow";
  const std::string v_forward_scene_flow_folder =
      "vkitti_2.0.3_forwardSceneFlow";
  const std::string v_class_semantics_folder = "vkitti_2.0.3_classSegmentation";
  const std::string v_instance_segmentation_folder =
      "vkitti_2.0.3_instanceSegmentation";
  const std::string v_rgb_folder = "vkitti_2.0.3_rgb";
  const std::string v_text_gt_folder = "vkitti_2.0.3_textgt";

  CameraParams camera_params_;
};

}  // namespace dyno
