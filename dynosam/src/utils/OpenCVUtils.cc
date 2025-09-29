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

#include "dynosam/utils/OpenCVUtils.hpp"

#include <config_utilities/config_utilities.h>
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
// #include <opencv4/opencv2/core/parallel/backend/

#include "dynosam/common/Types.hpp"  //for template to_string
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/visualizer/ColourMap.hpp"

namespace dyno {

template <>
std::string to_string<cv::Size>(const cv::Size& t) {
  return "[h=" + std::to_string(t.height) + " w=" + std::to_string(t.width) +
         "]";
}

template <>
std::string to_string<cv::Rect>(const cv::Rect& t) {
  return "[x=" + std::to_string(t.x) + " y=" + std::to_string(t.y) +
         " h=" + std::to_string(t.height) + " w=" + std::to_string(t.width) +
         "]";
}

namespace utils {

double calculateIoU(const cv::Rect& a, const cv::Rect& b) {
  cv::Rect intersection = a & b;
  double intersection_area = intersection.area();

  // Calculate union area
  double union_area = a.area() + b.area() - intersection_area;

  // Handle case where unionArea is zero (e.g., no overlap and one or both rects
  // have zero area)
  if (union_area == 0) {
    return 0.0;
  }

  // Calculate IoU
  return intersection_area / union_area;
}

bool cvSizeEqual(const cv::Size& a, const cv::Size& b) {
  return a.height == b.height && a.width == b.width;
}

bool cvSizeEqual(const cv::Mat& a, const cv::Mat& b) {
  return cvSizeEqual(a.size(), b.size());
}

void drawCircleInPlace(cv::Mat& img, const cv::Point2d& point,
                       const cv::Scalar& colour, const int radius,
                       const int thickness) {
  if (!matContains(img, point)) return;
  cv::circle(img, point, radius, colour, thickness);
}

std::string cvTypeToString(int type) {
  std::string r;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }
  r += "C";
  r += (chans + '0');
  return r;
}

std::string cvTypeToString(const cv::Mat& mat) {
  return cvTypeToString(mat.type());
}

cv::Mat concatenateImagesHorizontally(const cv::Mat& left_img,
                                      const cv::Mat& right_img) {
  cv::Mat left_img_tmp = left_img.clone();
  if (left_img_tmp.channels() == 1) {
    cv::cvtColor(left_img_tmp, left_img_tmp, cv::COLOR_GRAY2BGR);
  }
  cv::Mat right_img_tmp = right_img.clone();
  if (right_img_tmp.channels() == 1) {
    cv::cvtColor(right_img_tmp, right_img_tmp, cv::COLOR_GRAY2BGR);
  }

  cv::Size left_img_size = left_img_tmp.size();
  cv::Size right_img_size = right_img_tmp.size();

  CHECK_EQ(left_img_size.height, left_img_size.height)
      << "Cannot concat horizontally if images are not the same "
         "height";

  cv::Mat dual_img(left_img_size.height,
                   left_img_size.width + right_img_size.width, CV_8UC3);

  cv::Mat left(dual_img,
               cv::Rect(0, 0, left_img_size.width, left_img_size.height));
  left_img_tmp.copyTo(left);

  cv::Mat right(dual_img, cv::Rect(left_img_size.width, 0, right_img_size.width,
                                   right_img_size.height));

  right_img_tmp.copyTo(right);
  return dual_img;
}

cv::Mat concatenateImagesVertically(const cv::Mat& top_img,
                                    const cv::Mat& bottom_img) {
  cv::Mat top_img_tmp = top_img.clone();
  if (top_img_tmp.channels() == 1) {
    cv::cvtColor(top_img_tmp, top_img_tmp, cv::COLOR_GRAY2BGR);
  }
  cv::Mat bottom_img_tmp = bottom_img.clone();
  if (bottom_img_tmp.channels() == 1) {
    cv::cvtColor(bottom_img_tmp, bottom_img_tmp, cv::COLOR_GRAY2BGR);
  }

  cv::Size top_img_size = bottom_img_tmp.size();
  cv::Size bottom_img_size = bottom_img_tmp.size();

  CHECK_EQ(top_img_size.width, bottom_img_size.width)
      << "Cannot concat vertically if images are not the same width";

  cv::Mat dual_img(top_img_size.height + bottom_img_size.height,
                   top_img_size.width, CV_8UC3);

  cv::Mat top(dual_img,
              cv::Rect(0, 0, top_img_size.width, top_img_size.height));
  top_img_tmp.copyTo(top);

  cv::Mat bottom(dual_img,
                 cv::Rect(0, top_img_size.height, bottom_img_size.width,
                          bottom_img_size.height));

  bottom_img_tmp.copyTo(bottom);
  return dual_img;
}

void flowToRgb(const cv::Mat& flow, cv::Mat& rgb) {
  CHECK(flow.channels() == 2) << "Expecting flow in frame to have 2 channels";

  // Visualization part
  cv::Mat flow_parts[2];
  cv::split(flow, flow_parts);

  // Convert the algorithm's output into Polar coordinates
  cv::Mat magnitude, angle, magn_norm;
  cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
  cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
  angle *= ((1.f / 360.f) * (180.f / 255.f));

  // Build hsv image
  cv::Mat _hsv[3], hsv, hsv8, bgr;
  _hsv[0] = angle;
  _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
  _hsv[2] = magn_norm;
  cv::merge(_hsv, 3, hsv);
  hsv.convertTo(hsv8, CV_8U, 255.0);

  // Display the results
  cv::cvtColor(hsv8, rgb, cv::COLOR_HSV2BGR);
}

void labelMaskToRGB(const cv::Mat& mask, const cv::Mat& rgb_input,
                    cv::Mat& output, float alpha, int background_label) {
  CHECK_EQ(mask.channels(), 1u) << "Expecting mask input to have channels 1";
  CHECK_EQ(rgb_input.channels(), 3) << "Expecting rgb input to have channels 3";
  CHECK(cvSizeEqual(mask, rgb_input));

  rgb_input.copyTo(output);

  FunctionalParallelOpenCVMat process(output, [&](cv::Mat& viz, int i, int j) {
    const ObjectId* mask_ptr = mask.ptr<ObjectId>(i);
    const ObjectId object_label = mask_ptr[j];
    if (object_label != background_label) {
      cv::Scalar color = Color::uniqueId(object_label).bgra();

      cv::Vec3b* viz_row = viz.ptr<cv::Vec3b>(i);

      // blend colour with existing values in viz_row size we have already
      // copied the rgb input values into the output (referenced as viz)
      for (int c = 0; c < 3; ++c) {
        viz_row[j][c] = static_cast<uchar>(alpha * color[c] +
                                           (1.0 - alpha) * viz_row[j][c]);
      }
    }
  });
  process.run();
}

cv::Mat labelMaskToRGB(const cv::Mat& mask, int background_label,
                       const cv::Mat& rgb) {
  cv::Mat output;
  labelMaskToRGB(mask, rgb, output, 0.7, background_label);

  // for (int i = 0; i < mask.rows; i++) {
  //   for (int j = 0; j < mask.cols; j++) {
  //     // background is zero
  //     if (mask.at<int>(i, j) != background_label) {
  //       cv::Scalar color = Color::uniqueId(mask.at<int>(i, j));
  //       // rgb or bgr?
  //       mask_viz.at<cv::Vec3b>(i, j)[0] = color[0];
  //       mask_viz.at<cv::Vec3b>(i, j)[1] = color[1];
  //       mask_viz.at<cv::Vec3b>(i, j)[2] = color[2];
  //     }
  //   }
  // }

  return output;
}

cv::Mat labelMaskToRGB(const cv::Mat& mask, int background_label) {
  cv::Mat rgb = cv::Mat::zeros(mask.size(), CV_8UC3);
  return labelMaskToRGB(mask, background_label, rgb);
}

void getDisparityVis(cv::InputArray src, cv::OutputArray dst,
                     int unknown_disparity) {
  CHECK(!src.empty() && (src.depth() == CV_16S || src.depth() == CV_32F) &&
        (src.channels() == 1));
  // cv::Mat srcMat = src.getMat();
  cv::Mat srcMat = src.getMat();
  dst.create(srcMat.rows, srcMat.cols, CV_8UC1);
  cv::Mat& dstMat = dst.getMatRef();

  // Check its extreme values.
  double min_val;
  double max_val;
  cv::minMaxLoc(src, &min_val, &max_val);

  // Multiply by 1.25 just to saturate a bit the extremums.
  double scale = 2.0 * 255.0 / (max_val - min_val);
  srcMat.convertTo(dstMat, CV_8UC1, scale / 16.0);
  dstMat &= (srcMat != unknown_disparity);
}

void drawLabeledBoundingBox(cv::Mat& image, const std::string& label,
                            const cv::Scalar& colour,
                            const cv::Rect& bounding_box,
                            const int& bb_thickness) {
  constexpr static double kFontScale = 0.6;
  constexpr static int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
  // text thickness
  constexpr static int kThickness = 1;

  // Top left corner.
  const cv::Point& tlc = bounding_box.tl();

  // Display the label at the top of the bounding box.
  int base_line;
  cv::Size label_size =
      cv::getTextSize(label, kFontFace, kFontScale, kThickness, &base_line);

  // draw on filled black rectangle that the object label will then be drawn
  // over to make it easier to see
  constexpr static int pixel_buffer = 2;  // pixel buffer around the rectangle
  const cv::Point tlc_black_rectangle(tlc.x,
                                      tlc.y - label_size.height - pixel_buffer);
  const cv::Point brc_black_rectangle(tlc.x + label_size.width + pixel_buffer,
                                      tlc.y);
  cv::rectangle(image, tlc_black_rectangle, brc_black_rectangle,
                cv::Scalar(0, 0, 0), -1);
  // Put the label on the black rectangle.
  // Note: text origin starts from the bottom left corner of the text string in
  // the image and we add a pixel buffer along y to make it look better
  cv::putText(image, label, cv::Point(tlc.x, tlc.y - 2), kFontFace, kFontScale,
              cv::Scalar(255, 255, 255), kThickness);
  // draw bounding box with line thickness
  cv::rectangle(image, bounding_box, colour, bb_thickness);
}

void drawObjectPoseAxes(cv::Mat& image, const cv::Mat& K, const cv::Mat& D,
                        const std::vector<gtsam::Pose3>& poses_c, float scale) {
  cv::Mat K_float, D_float;
  K.copyTo(K_float);
  D.copyTo(D_float);

  K_float.convertTo(K_float, CV_32F);
  D_float.convertTo(D_float, CV_32F);

  // Define the 3D points for the axis (origin and points along X, Y, Z)
  static const std::vector<cv::Point3f> axis_points = {
      {0, 0, 0},      // Origin
      {scale, 0, 0},  // X-axis (red)
      {0, scale, 0},  // Y-axis (green)
      {0, 0, scale}   // Z-axis (blue)
  };

  for (const gtsam::Pose3& pose : poses_c) {
    auto [rot_matrix, tvec] = Pose2cvmats(pose);

    rot_matrix.convertTo(rot_matrix, CV_32F);
    tvec.convertTo(tvec, CV_32F);

    cv::Mat rvec;
    cv::Rodrigues(rot_matrix,
                  rvec);  // Convert rotation matrix to Rodrigues vector

    std::vector<cv::Point2f> image_points;
    cv::projectPoints(axis_points, rvec, tvec, K_float, D_float, image_points);

    // Draw axes on the image
    cv::line(image, image_points[0], image_points[1], cv::Scalar(0, 0, 255),
             2);  // X-axis in red
    cv::line(image, image_points[0], image_points[2], cv::Scalar(0, 255, 0),
             2);  // Y-axis in green
    cv::line(image, image_points[0], image_points[3], cv::Scalar(255, 0, 0),
             2);  // Z-axis in blue
  }
}

// void drawObjectPoseTrajectory(cv::Mat& image, const cv::Mat& K, const
// cv::Mat& D, const ObjectPoseMap& poses_c, FrameId frame_k, int keep) {
//   cv::Mat K_float, D_float;
//   K.copyTo(K_float);
//   D.copyTo(D_float);

//   K_float.convertTo(K_float, CV_32F);
//   D_float.convertTo(D_float, CV_32F);

//   constexpr int line_thickness = 3;
//   for (const auto& [object_id, per_frame_poses] : poses_c) {
//     const cv::Scalar colour = Color::uniqueId(object_id).bgra();

//     std::vector<cv::Point> cv_line;

//     size_t traj_size;
//     // draw all the poses
//     if (keep == -1) {
//       traj_size = per_frame_poses.size();
//     } else {
//       traj_size = std::min(keep, static_cast<int>(per_frame_poses.size()));
//     }

//     int count = 0;
//     for(auto rit = per_frame_poses.rbegin(); rit != per_frame_poses.rend();
//     ++rit) {
//       const gtsam::Pose3 pose = rit->second;

//       auto [rot_matrix, tvec] = Pose2cvmats(pose);
//       rot_matrix.convertTo(rot_matrix, CV_32F);
//       tvec.convertTo(tvec, CV_32F);

//       cv::Mat rvec;
//       cv::Rodrigues(rot_matrix,
//                     rvec);  // Convert rotation matrix to Rodrigues vector

//       std::vector<cv::Point2f> image_points;
//       cv::projectPoints(axis_points, rvec, tvec, K_float, D_float,
//       image_points);
//     }

//   }
// }

bool compareCvMatsUpToTol(const cv::Mat& mat1, const cv::Mat& mat2,
                          const double& tol) {
  CHECK_EQ(mat1.size(), mat2.size());
  CHECK_EQ(mat1.type(), mat2.type());

  // treat two empty mat as identical as well
  if (mat1.empty() && mat2.empty()) {
    LOG(WARNING) << "CvMatCmp: asked comparison of 2 empty matrices.";
    return true;
  }

  // Compare the two matrices!
  cv::Mat diff = mat1 - mat2;
  return cv::checkRange(diff, true, nullptr, -tol, tol);
}

const float FLOW_TAG_FLOAT = 202021.25f;
const char* FLOW_TAG_STRING = "PIEH";

cv::Mat readOpticalFlow(const std::string& path) {
  using namespace cv;

  Mat_<Point2f> flow;
  std::ifstream file(path.c_str(), std::ios_base::binary);
  if (!file.good()) return flow;  // no file - return empty matrix

  float tag;
  file.read((char*)&tag, sizeof(float));
  if (tag != FLOW_TAG_FLOAT) return flow;

  int width, height;

  file.read((char*)&width, 4);
  file.read((char*)&height, 4);

  flow.create(height, width);

  for (int i = 0; i < flow.rows; ++i) {
    for (int j = 0; j < flow.cols; ++j) {
      Point2f u;
      file.read((char*)&u.x, sizeof(float));
      file.read((char*)&u.y, sizeof(float));
      if (!file.good()) {
        flow.release();
        return flow;
      }

      flow(i, j) = u;
    }
  }
  file.close();
  return flow;
}

bool writeOpticalFlow(const std::string& path, const cv::Mat& flow) {
  using namespace cv;
  const int nChannels = 2;

  Mat input = flow;
  if (input.channels() != nChannels || input.depth() != CV_32F ||
      path.length() == 0)
    return false;

  std::ofstream file(path.c_str(), std::ofstream::binary);
  if (!file.good()) return false;

  int nRows, nCols;

  nRows = (int)input.size().height;
  nCols = (int)input.size().width;

  const int headerSize = 12;
  char header[headerSize];
  memcpy(header, FLOW_TAG_STRING, 4);
  // size of ints is known - has been asserted in the current function
  memcpy(header + 4, reinterpret_cast<const char*>(&nCols), sizeof(nCols));
  memcpy(header + 8, reinterpret_cast<const char*>(&nRows), sizeof(nRows));
  file.write(header, headerSize);
  if (!file.good()) return false;

  //    if ( input.isContinuous() ) //matrix is continous - treat it as a single
  //    row
  //    {
  //        nCols *= nRows;
  //        nRows = 1;
  //    }

  int row;
  char* p;
  for (row = 0; row < nRows; row++) {
    p = input.ptr<char>(row);
    file.write(p, nCols * nChannels * sizeof(float));
    if (!file.good()) return false;
  }
  file.close();
  return true;
}

}  // namespace utils
}  // namespace dyno
