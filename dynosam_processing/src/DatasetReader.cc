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

#include "dynosam_processing/DatasetReader.hpp"

#include <glog/logging.h>

#include "dynosam_common/Transforms.hpp"
#include "dynosam_processing/DataReaderUtils.hpp"

namespace dyno {

std::string RGBDataFolder::getFolderName() const { return "image_0"; }

cv::Mat RGBDataFolder::getItem(size_t idx) {
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(6) << idx;
  const std::string file_path =
      (std::string)getAbsolutePath() + "/" + ss.str() + ".png";

  cv::Mat rgb;
  utils::loadRGB(file_path, rgb);

  CHECK(!rgb.empty());
  return rgb;
}

std::string OpticalFlowDataFolder::getFolderName() const { return "flow"; }

cv::Mat OpticalFlowDataFolder::getItem(size_t idx) {
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(6) << idx;
  const std::string file_path =
      (std::string)getAbsolutePath() + "/" + ss.str() + ".flo";

  cv::Mat flow;
  utils::loadFlow(file_path, flow);

  CHECK(!flow.empty());
  return flow;
}

cv::Mat DepthDataFolder::getItem(size_t idx) {
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(6) << idx;
  const std::string file_path =
      (std::string)getAbsolutePath() + "/" + ss.str() + ".png";

  cv::Mat depth;
  utils::loadDepth(file_path, depth);

  CHECK(!depth.empty());
  return depth;
}

cv::Mat SegMaskFolder::getItem(size_t idx) {
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(6) << idx;
  const std::string file_path =
      (std::string)getAbsolutePath() + "/" + ss.str() + ".txt";

  const cv::Size expected_mask_size = rgb_data_folder_->getItem(idx).size();

  cv::Mat mask;
  utils::loadSemanticMask(file_path, expected_mask_size, mask);

  CHECK(!mask.empty());
  return mask;
}

std::string TimestampFile::getFolderName() const { return "times.txt"; }

double TimestampFile::getItem(size_t idx) { return times.at(idx); }

void TimestampFile::onPathInit() {
  std::ifstream times_stream((std::string)getAbsolutePath(), std::ios::in);

  while (!times_stream.eof()) {
    std::string s;
    getline(times_stream, s);
    if (!s.empty()) {
      std::stringstream ss;
      ss << s;
      double t;
      ss >> t;
      times.push_back(t);
    }
  }
  times_stream.close();
}

BasicDynoSensorRig::BasicDynoSensorRig(const CameraParams& canonical_params,
                                       const DepthRigType depth_rig_type,
                                       const ImuCalibration& imu_calibration,
                                       const ReferenceFrames& reference_frames)
    : canonical_params_(canonical_params),
      depth_rig_type_(depth_rig_type),
      imu_calibration_(imu_calibration),
      reference_frames_(reference_frames) {
  // by default updata extrinsics to rotate optical frame to robotic frame
  canonical_params_.setExtrinsics(openCVToRobotTransform());

  // should not have to apply any rotation to the IMU calibration
  // becuase the canonical params extrinsics are ONLY used for visualisation
  // output and we assume the the IMU calibration has extrinscis that will
  // transform the IMU data into the canonical params frame prior to extrinsics
  // application
}

}  // namespace dyno
