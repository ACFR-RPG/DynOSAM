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

#include <opencv4/opencv2/core.hpp>

#include "dynosam/common/ImageContainer.hpp"
#include "dynosam/common/Types.hpp"

namespace dyno {

static constexpr char kRgbdFrontendOutputJsonFile[] =
    "rgbd_frontend_output.bson";

/**
 * @brief Determines which frontend module to load.
 *
 *
 *
 */
enum FrontendType { kRGBD = 0, kMono = 1 };

enum class TrackingStatus {
  VALID,
  LOW_DISPARITY,
  FEW_MATCHES,
  INVALID,
  DISABLED
};

template <>
inline std::string to_string(const TrackingStatus& status) {
  std::string status_str = "";
  switch (status) {
    case TrackingStatus::VALID: {
      status_str = "VALID";
      break;
    }
    case TrackingStatus::INVALID: {
      status_str = "INVALID";
      break;
    }
    case TrackingStatus::DISABLED: {
      status_str = "DISABLED";
      break;
    }
    case TrackingStatus::FEW_MATCHES: {
      status_str = "FEW_MATCHES";
      break;
    }
    case TrackingStatus::LOW_DISPARITY: {
      status_str = "LOW_DISPARITY";
      break;
    }
  }
  return status_str;
}

/**
 * @brief Struct containing debug imagery from the frontend that (optionally) is
 * included in the frontend output
 *
 */
struct DebugImagery {
  DYNO_POINTER_TYPEDEFS(DebugImagery)

  // TODO: make const!!
  cv::Mat tracking_image;
  // TODO: for now!
  cv::Mat rgb_viz;
  cv::Mat flow_viz;
  cv::Mat depth_viz;
  cv::Mat mask_viz;
};

}  // namespace dyno
