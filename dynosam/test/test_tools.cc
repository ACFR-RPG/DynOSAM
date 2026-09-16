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
#include <gtest/gtest.h>
#include <gtsam/geometry/StereoCamera.h>

#include <cmath>

#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/Types.hpp"
#include "internal/helpers.hpp"

using namespace dyno;

TEST(VisionTools, determineOutlierIdsBasic) {
  TrackletIds tracklets = {1, 2, 3, 4, 5};
  TrackletIds inliers = {1, 2};

  TrackletIds expected_outliers = {3, 4, 5};
  TrackletIds outliers;
  determineOutlierIds(inliers, tracklets, outliers);
  EXPECT_EQ(expected_outliers, outliers);
}

TEST(VisionTools, determineOutlierIdsUnorderd) {
  TrackletIds tracklets = {12, 45, 1, 85, 3, 100};
  TrackletIds inliers = {3, 1, 100};

  TrackletIds expected_outliers = {12, 45, 85};
  TrackletIds outliers;
  determineOutlierIds(inliers, tracklets, outliers);
  EXPECT_EQ(expected_outliers, outliers);
}

TEST(VisionTools, determineOutlierIdsNoSubset) {
  TrackletIds tracklets = {12, 45, 1, 85, 3, 100};
  TrackletIds inliers = {12, 45, 1, 85, 3, 100};

  TrackletIds outliers = {4, 5, 6};  // also add a test that outliers is cleared
  determineOutlierIds(inliers, tracklets, outliers);
  EXPECT_TRUE(outliers.empty());
}

TEST(VisionTools, testMacVOUncertaintyPropogation) {
  Camera camera = dyno_testing::makeDefaultCamera();
  CameraParams params = camera.getParams();
  auto camera_impl = camera.getImplCamera();

  // make stereo camera
  const double base_line = 0.5;
  gtsam::Cal3_S2Stereo::shared_ptr stereo_params =
      boost::make_shared<gtsam::Cal3_S2Stereo>(
          params.fx(), params.fy(), 0, params.cu(), params.cv(), base_line);
  gtsam::StereoCamera stereo_camera(gtsam::Pose3::Identity(), stereo_params);

  Feature feature;
  Keypoint kp(params.cu(), params.cv() + 20);
  feature.keypoint(kp);
  feature.depth(1.0);

  // sigmas squared
  double kp_sigma_2 = 0.1;
  double depth_sigma_2 = 0.000005;

  // first check diagonal components of proposed macv matrix
  gtsam::Matrix32 J_keypoint;
  gtsam::Matrix31 J_depth;
  gtsam::Point3 landmark =
      camera_impl->backproject(feature.keypoint(), feature.depth(), boost::none,
                               J_keypoint, J_depth, boost::none);

  gtsam::StereoPoint2 stereo_kp = stereo_camera.project(landmark);
  EXPECT_EQ(kp(0), stereo_kp.uL());
  EXPECT_EQ(kp(1), stereo_kp.v());

  gtsam::Matrix33 J_stereo_point;
  stereo_camera.backproject2(stereo_kp, boost::none, J_stereo_point);

  gtsam::Point3 calc_landmark(
      ((feature.keypoint()(0) - params.cu()) * feature.depth()) / params.fx(),
      ((feature.keypoint()(1) - params.cv()) * feature.depth()) / params.fy(),
      feature.depth());
  EXPECT_TRUE(gtsam::assert_equal(calc_landmark, landmark));

  // form measurement covariance matrices
  gtsam::Matrix22 pixel_covariance_matrix;
  pixel_covariance_matrix << kp_sigma_2, 0.0, 0.0, kp_sigma_2;

  gtsam::Matrix33 stereo_pixel_covariance_matrix;
  stereo_pixel_covariance_matrix << kp_sigma_2, 0.0, 0.0, 0, kp_sigma_2, 0, 0,
      0, kp_sigma_2;

  // for depth uncertainty, we model it as a quadratic increase with distnace
  // double depth_covariance = depth_sigma * std::pow(depth, 2);
  double depth_covariance = depth_sigma_2;
  LOG(INFO) << "J_keypoint " << J_keypoint;
  // calcualte 3x3 covairance matrix
  gtsam::Matrix33 covariance =
      J_keypoint * pixel_covariance_matrix * J_keypoint.transpose();
  // J_depth * depth_covariance * J_depth.transpose();

  gtsam::Matrix33 stereo_covariance = J_stereo_point *
                                      stereo_pixel_covariance_matrix *
                                      J_stereo_point.transpose();

  LOG(INFO) << "Jacobian cov " << covariance;
  LOG(INFO) << "Stereo Jacobian cov " << stereo_covariance;

  double d_2 = std::pow(feature.depth(), 2);
  double u_2 = std::pow(feature.keypoint()(0), 2);
  double v_2 = std::pow(feature.keypoint()(1), 2);
  double fx_2 = std::pow(params.fx(), 2);
  double fy_2 = std::pow(params.fy(), 2);
  double cx_2 = std::pow(params.cu(), 2);
  double cy_2 = std::pow(params.cv(), 2);

  double mac_v_sigma_x = ((kp_sigma_2 + d_2) * (depth_sigma_2 + u_2) -
                          u_2 * d_2 + cx_2 * depth_sigma_2) /
                         fx_2;
  double mac_v_sigma_y = ((kp_sigma_2 + d_2) * (depth_sigma_2 + v_2) -
                          v_2 * d_2 + cy_2 * depth_sigma_2) /
                         fy_2;
  double mac_v_sigma_z = depth_sigma_2;

  gtsam::Matrix33 macvo_covariance;
  macvo_covariance << mac_v_sigma_x, 0, 0, 0, mac_v_sigma_y, 0, 0, 0,
      mac_v_sigma_z;

  LOG(INFO) << "macvo cov " << macvo_covariance;
}

bool findObjectContoursAndBoundingBoxes(
    const cv::Mat& mask, const ObjectIds& object_ids,
    std::vector<std::vector<std::vector<cv::Point>>>& all_contours,
    std::vector<cv::Rect>& bounding_boxes) {
  const auto total_start = std::chrono::steady_clock::now();

  CHECK_EQ(mask.type(), CV_8UC1);

  all_contours.clear();
  bounding_boxes.clear();

  all_contours.resize(object_ids.size());
  bounding_boxes.resize(object_ids.size());

  if (mask.empty() || object_ids.empty()) {
    return false;
  }

  // --------------------------------------------------------------------------
  // Map object ID -> index in object_ids.
  //
  // This lets the single image traversal below update only objects that we
  // actually care about.
  // --------------------------------------------------------------------------
  std::array<int, 256> object_to_index;
  object_to_index.fill(-1);

  for (size_t i = 0; i < object_ids.size(); ++i) {
    CHECK_LE(object_ids[i], 255);
    object_to_index[object_ids[i]] = static_cast<int>(i);
  }

  // --------------------------------------------------------------------------
  // Single full-image pass.
  //
  // Find the raw bounding box of every requested object simultaneously.
  // --------------------------------------------------------------------------
  const auto bbox_scan_start = std::chrono::steady_clock::now();

  std::array<int, 256> min_x;
  std::array<int, 256> min_y;
  std::array<int, 256> max_x;
  std::array<int, 256> max_y;

  min_x.fill(mask.cols);
  min_y.fill(mask.rows);
  max_x.fill(-1);
  max_y.fill(-1);

  for (int y = 0; y < mask.rows; ++y) {
    const uint8_t* row = mask.ptr<uint8_t>(y);

    for (int x = 0; x < mask.cols; ++x) {
      const uint8_t object_id = row[x];

      const int object_index = object_to_index[object_id];

      if (object_index < 0) {
        continue;
      }

      min_x[object_id] = std::min(min_x[object_id], x);
      min_y[object_id] = std::min(min_y[object_id], y);
      max_x[object_id] = std::max(max_x[object_id], x);
      max_y[object_id] = std::max(max_y[object_id], y);
    }
  }

  const auto bbox_scan_end = std::chrono::steady_clock::now();

  const double bbox_scan_ms =
      std::chrono::duration<double, std::milli>(bbox_scan_end - bbox_scan_start)
          .count();

  const auto start = std::chrono::steady_clock::now();

  for (int y = 0; y < mask.rows; ++y) {
    const uint8_t* row = mask.ptr<uint8_t>(y);

    for (int x = 0; x < mask.cols; ++x) {
      const uint8_t object_id = row[x];
      const int object_index = object_to_index[object_id];

      if (object_index < 0) {
        continue;
      }

      min_x[object_id] = std::min(min_x[object_id], x);
      min_y[object_id] = std::min(min_y[object_id], y);
      max_x[object_id] = std::max(max_x[object_id], x);
      max_y[object_id] = std::max(max_y[object_id], y);

      // bbox update
    }
  }

  const auto end = std::chrono::steady_clock::now();
  const double bbox_scan_empty_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  // --------------------------------------------------------------------------
  // Process each object using its ROI.
  // --------------------------------------------------------------------------
  static constexpr int contour_padding = 2;

  static const cv::Mat contour_dilate_element =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 5));

  double total_compare_ms = 0.0;
  double total_dilate_ms = 0.0;
  double total_contours_ms = 0.0;
  double total_bbox_ms = 0.0;

  for (size_t object_index = 0; object_index < object_ids.size();
       ++object_index) {
    const auto object_id = object_ids[object_index];

    CHECK_LE(object_id, 255);

    if (max_x[object_id] < 0) {
      std::cout << "Object " << static_cast<int>(object_id)
                << " | not present in mask\n";
      continue;
    }

    // ------------------------------------------------------------------------
    // Expand vertically by two pixels because the original implementation
    // applies a 1x5 dilation before findContours().
    // ------------------------------------------------------------------------
    const int x0 = min_x[object_id];
    const int x1 = max_x[object_id] + 1;

    const int y0 = std::max(0, min_y[object_id] - contour_padding);

    const int y1 = std::min(mask.rows, max_y[object_id] + 1 + contour_padding);

    const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);

    // ------------------------------------------------------------------------
    // Extract object mask inside ROI.
    // ------------------------------------------------------------------------
    const auto compare_start = std::chrono::steady_clock::now();

    cv::Mat object_mask;

    cv::compare(mask(roi), object_id, object_mask, cv::CMP_EQ);

    const auto compare_end = std::chrono::steady_clock::now();

    const double compare_ms =
        std::chrono::duration<double, std::milli>(compare_end - compare_start)
            .count();

    total_compare_ms += compare_ms;

    // ------------------------------------------------------------------------
    // Preserve original 1x5 vertical dilation.
    // ------------------------------------------------------------------------
    const auto dilate_start = std::chrono::steady_clock::now();

    cv::Mat dilated_object_mask;

    cv::dilate(object_mask, dilated_object_mask, contour_dilate_element,
               cv::Point(-1, -1));

    const auto dilate_end = std::chrono::steady_clock::now();

    const double dilate_ms =
        std::chrono::duration<double, std::milli>(dilate_end - dilate_start)
            .count();

    total_dilate_ms += dilate_ms;

    // ------------------------------------------------------------------------
    // Find contours.
    // ------------------------------------------------------------------------
    const auto contours_start = std::chrono::steady_clock::now();

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(dilated_object_mask, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_NONE);

    const auto contours_end = std::chrono::steady_clock::now();

    const double contours_ms =
        std::chrono::duration<double, std::milli>(contours_end - contours_start)
            .count();

    total_contours_ms += contours_ms;

    // ------------------------------------------------------------------------
    // Offset contour points back to global coordinates AND calculate the
    // bounding box at the same time.
    // ------------------------------------------------------------------------
    const auto bbox_start = std::chrono::steady_clock::now();

    cv::Rect object_bbox;

    bool bbox_initialised = false;

    for (auto& contour : contours) {
      for (auto& point : contour) {
        point.x += roi.x;
        point.y += roi.y;

        if (!bbox_initialised) {
          object_bbox = cv::Rect(point.x, point.y, 1, 1);

          bbox_initialised = true;
          continue;
        }

        const int current_min_x = object_bbox.x;
        const int current_min_y = object_bbox.y;

        const int current_max_x = object_bbox.x + object_bbox.width - 1;

        const int current_max_y = object_bbox.y + object_bbox.height - 1;

        const int new_min_x = std::min(current_min_x, point.x);

        const int new_min_y = std::min(current_min_y, point.y);

        const int new_max_x = std::max(current_max_x, point.x);

        const int new_max_y = std::max(current_max_y, point.y);

        object_bbox.x = new_min_x;
        object_bbox.y = new_min_y;
        object_bbox.width = new_max_x - new_min_x + 1;
        object_bbox.height = new_max_y - new_min_y + 1;
      }
    }

    const auto bbox_end = std::chrono::steady_clock::now();

    const double bbox_ms =
        std::chrono::duration<double, std::milli>(bbox_end - bbox_start)
            .count();

    total_bbox_ms += bbox_ms;

    all_contours[object_index] = std::move(contours);
    bounding_boxes[object_index] = object_bbox;

    std::cout << "Object " << static_cast<int>(object_id) << " | ROI "
              << roi.width << "x" << roi.height << " ("
              << roi.width * roi.height << " px)"
              << " | compare: " << compare_ms << " ms"
              << " | dilate: " << dilate_ms << " ms"
              << " | findContours: " << contours_ms << " ms"
              << " | bbox from contours: " << bbox_ms << " ms"
              << " | contours: " << all_contours[object_index].size() << "\n";
  }

  const auto total_end = std::chrono::steady_clock::now();

  std::cout << "Benchmark scan: " << bbox_scan_empty_ms << "ms\n";

  std::cout << "Object contour + bbox processing:\n"
            << "  Full-image bbox scan: " << bbox_scan_ms << " ms\n"
            << "  Total compare: " << total_compare_ms << " ms\n"
            << "  Total ROI dilate: " << total_dilate_ms << " ms\n"
            << "  Total findContours: " << total_contours_ms << " ms\n"
            << "  Total bbox from contours: " << total_bbox_ms << " ms\n"
            << "  TOTAL: "
            << std::chrono::duration<double, std::milli>(total_end -
                                                         total_start)
                   .count()
            << " ms\n";

  return true;
}

TEST(ObjectContours, CustomContoursAndBoundingBoxes) {
  // constexpr int width = 640;
  // constexpr int height = 480;
  constexpr int width = 1280;
  constexpr int height = 960;
  constexpr int num_objects = 8;

  constexpr int inner_thickness = 6;
  constexpr int contour_padding = 2;

  // ---------------------------------------------------------------------------
  // Generate deterministic random mask.
  // ---------------------------------------------------------------------------

  cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);

  cv::RNG rng(12345);

  for (uint8_t object_id = 1; object_id <= num_objects; ++object_id) {
    const cv::Point centre(rng.uniform(50, width - 50),
                           rng.uniform(50, height - 50));

    const int radius_x = rng.uniform(20, 70);
    const int radius_y = rng.uniform(20, 70);
    const int num_parts = rng.uniform(2, 6);

    for (int i = 0; i < num_parts; ++i) {
      const cv::Point offset(rng.uniform(-radius_x / 2, radius_x / 2),
                             rng.uniform(-radius_y / 2, radius_y / 2));

      const cv::Size size(rng.uniform(radius_x / 2, radius_x),
                          rng.uniform(radius_y / 2, radius_y));

      cv::ellipse(mask, centre + offset, size, rng.uniform(0.0, 180.0), 0.0,
                  360.0, cv::Scalar(object_id), cv::FILLED);
    }
  }

  // Add disconnected components to objects 1-3.
  // for (uint8_t object_id = 1;
  //      object_id <= 3;
  //      ++object_id) {
  //   const cv::Point centre(
  //       rng.uniform(50, width - 50),
  //       rng.uniform(50, height - 50));

  //   cv::circle(
  //       mask,
  //       centre,
  //       rng.uniform(10, 25),
  //       cv::Scalar(object_id),
  //       cv::FILLED);
  // }

  ObjectIds object_ids;

  for (uint8_t object_id = 1; object_id <= num_objects; ++object_id) {
    object_ids.push_back(object_id);
  }

  // ---------------------------------------------------------------------------
  // Run the implementation under test.
  //
  // Keep the original API:
  //
  //   mask
  //   object_ids
  //   contours
  //   bounding_boxes
  // ---------------------------------------------------------------------------

  std::vector<std::vector<std::vector<cv::Point>>> custom_contours;
  std::vector<cv::Rect> custom_bounding_boxes;

  ASSERT_TRUE(findObjectContoursAndBoundingBoxes(
      mask, object_ids, custom_contours, custom_bounding_boxes));

  ASSERT_EQ(custom_contours.size(), object_ids.size());

  ASSERT_EQ(custom_bounding_boxes.size(), object_ids.size());

  // ---------------------------------------------------------------------------
  // Reference original object bounding boxes.
  // ---------------------------------------------------------------------------

  std::vector<cv::Rect> reference_bounding_boxes(object_ids.size());

  for (size_t object_index = 0; object_index < object_ids.size();
       ++object_index) {
    const uint8_t object_id = object_ids[object_index];

    int min_x = mask.cols;
    int min_y = mask.rows;
    int max_x = -1;
    int max_y = -1;

    for (int y = 0; y < mask.rows; ++y) {
      const uint8_t* row = mask.ptr<uint8_t>(y);

      for (int x = 0; x < mask.cols; ++x) {
        if (row[x] != object_id) {
          continue;
        }

        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
      }
    }

    ASSERT_GE(max_x, 0);

    reference_bounding_boxes[object_index] =
        cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
  }

  // ---------------------------------------------------------------------------
  // Reference contours.
  //
  // This reproduces the ROI-based contour extraction used by the implementation
  // under test.
  // ---------------------------------------------------------------------------
  // const int outer_thickness = 40;
  static const cv::Mat contour_dilate_element =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 5));
  // static const cv::Mat contour_dilate_element =
  //     cv::getStructuringElement(
  //         cv::MORPH_RECT,
  //        cv::Size(2 * outer_thickness + 1, 2 * outer_thickness + 1));

  std::vector<std::vector<std::vector<cv::Point>>> reference_contours(
      object_ids.size());

  for (size_t object_index = 0; object_index < object_ids.size();
       ++object_index) {
    const uint8_t object_id = object_ids[object_index];

    const cv::Rect bbox = reference_bounding_boxes[object_index];

    const int x0 = bbox.x;
    const int x1 = bbox.x + bbox.width;

    const int y0 = std::max(0, bbox.y - contour_padding);

    const int y1 = std::min(mask.rows, bbox.y + bbox.height + contour_padding);

    const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);

    cv::Mat object_mask;

    cv::compare(mask(roi), object_id, object_mask, cv::CMP_EQ);

    cv::Mat dilated_object_mask;

    cv::dilate(object_mask, dilated_object_mask, contour_dilate_element,
               cv::Point(-1, -1));

    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(dilated_object_mask, reference_contours[object_index],
                     hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // Convert ROI coordinates back to global coordinates.
    for (auto& contour : reference_contours[object_index]) {
      for (auto& point : contour) {
        point.x += roi.x;
        point.y += roi.y;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Check contours and bounding boxes.
  // ---------------------------------------------------------------------------

  for (size_t object_index = 0; object_index < object_ids.size();
       ++object_index) {
    const uint8_t object_id = object_ids[object_index];

    // ASSERT_EQ(
    //     custom_bounding_boxes[object_index],
    //     reference_bounding_boxes[object_index])
    //     << "Bounding box mismatch for object "
    //     << static_cast<int>(object_id);

    // ASSERT_EQ(
    //     custom_contours[object_index].size(),
    //     reference_contours[object_index].size())
    //     << "Contour count mismatch for object "
    //     << static_cast<int>(object_id);

    // for (size_t contour_index = 0;
    //      contour_index <
    //          custom_contours[object_index].size();
    //      ++contour_index) {
    //   ASSERT_EQ(
    //       custom_contours[object_index][contour_index],
    //       reference_contours[object_index][contour_index])
    //       << "Contour mismatch for object "
    //       << static_cast<int>(object_id)
    //       << ", contour "
    //       << contour_index;
    // }
  }

  // ---------------------------------------------------------------------------
  // Reproduce the ORIGINAL inner erosion operation.
  //
  // This is intentionally independent of the implementation under test.
  //
  // Original:
  //
  //   eroded_mask = erode(
  //       thicc_boarder,
  //       21x21 kernel);
  //
  // ---------------------------------------------------------------------------

  cv::Mat thicc_boarder = cv::Mat::zeros(mask.size(), CV_8UC1);

  for (const uint8_t object_id : object_ids) {
    cv::Mat object_mask;

    cv::compare(mask, object_id, object_mask, cv::CMP_EQ);

    thicc_boarder.setTo(cv::Scalar(object_id), object_mask);
  }

  static const cv::Mat inner_element = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(2 * inner_thickness + 1, 2 * inner_thickness + 1));

  cv::Mat reference_eroded_mask;

  cv::erode(thicc_boarder, reference_eroded_mask, inner_element);

  // This is exactly the original:
  //
  //   thicc_inner_boarder_mask =
  //       thicc_boarder - eroded_mask;
  //
  cv::Mat reference_inner_border = thicc_boarder - reference_eroded_mask;

  // ---------------------------------------------------------------------------
  // Calculate reference inner bounding boxes.
  //
  // This replaces the original N calls to
  // findObjectBoundingBox(eroded_mask,...) with ONE traversal of the eroded
  // image.
  // ---------------------------------------------------------------------------

  std::array<int, 256> min_x;
  std::array<int, 256> min_y;
  std::array<int, 256> max_x;
  std::array<int, 256> max_y;

  min_x.fill(mask.cols);
  min_y.fill(mask.rows);
  max_x.fill(-1);
  max_y.fill(-1);

  for (int y = 0; y < reference_eroded_mask.rows; ++y) {
    const uint8_t* row = reference_eroded_mask.ptr<uint8_t>(y);

    for (int x = 0; x < reference_eroded_mask.cols; ++x) {
      const uint8_t object_id = row[x];

      if (object_id == 0) {
        continue;
      }

      min_x[object_id] = std::min(min_x[object_id], x);

      min_y[object_id] = std::min(min_y[object_id], y);

      max_x[object_id] = std::max(max_x[object_id], x);

      max_y[object_id] = std::max(max_y[object_id], y);
    }
  }

  std::vector<cv::Rect> reference_inner_bounding_boxes(object_ids.size());

  for (size_t object_index = 0; object_index < object_ids.size();
       ++object_index) {
    const uint8_t object_id = object_ids[object_index];

    if (max_x[object_id] < 0) {
      // The object disappeared completely after erosion.
      reference_inner_bounding_boxes[object_index] = cv::Rect();

      continue;
    }

    reference_inner_bounding_boxes[object_index] =
        cv::Rect(min_x[object_id], min_y[object_id],
                 max_x[object_id] - min_x[object_id] + 1,
                 max_y[object_id] - min_y[object_id] + 1);
  }

  // ---------------------------------------------------------------------------
  // Print inner bounding boxes.
  // ---------------------------------------------------------------------------

  for (size_t object_index = 0; object_index < object_ids.size();
       ++object_index) {
    const uint8_t object_id = object_ids[object_index];

    const cv::Rect& bbox = reference_inner_bounding_boxes[object_index];

    std::cout << "Object " << static_cast<int>(object_id)
              << " inner bbox: " << bbox.x << ", " << bbox.y << ", "
              << bbox.width << " x " << bbox.height << std::endl;
  }

  // ---------------------------------------------------------------------------
  // Debug visualisation.
  // ---------------------------------------------------------------------------

  cv::RNG viz_rng(54321);

  std::vector<cv::Scalar> colours;

  for (size_t i = 0; i < object_ids.size(); ++i) {
    colours.emplace_back(viz_rng.uniform(50, 255), viz_rng.uniform(50, 255),
                         viz_rng.uniform(50, 255));
  }

  // ---------------------------------------------------------------------------
  // Window 1: Original labelled mask.
  // ---------------------------------------------------------------------------

  cv::Mat original_viz;

  cv::normalize(mask, original_viz, 0, 255, cv::NORM_MINMAX, CV_8UC1);

  cv::applyColorMap(original_viz, original_viz, cv::COLORMAP_JET);

  cv::imshow("Original Object Mask", original_viz);

  // ---------------------------------------------------------------------------
  // Window 2: Original mask + detected bounding boxes.
  // ---------------------------------------------------------------------------

  cv::Mat bbox_viz(mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));

  for (size_t object_index = 0; object_index < object_ids.size();
       ++object_index) {
    const uint8_t object_id = object_ids[object_index];

    const cv::Scalar colour = colours[object_index];

    bbox_viz.setTo(colour, mask == object_id);

    cv::rectangle(bbox_viz, custom_bounding_boxes[object_index], colour, 2);

    cv::putText(bbox_viz, std::to_string(static_cast<int>(object_id)),
                custom_bounding_boxes[object_index].tl() + cv::Point(0, -5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, colour, 2);
  }

  // cv::resize(
  //     bbox_viz,
  //     small_bbox_viz,
  //     cv::Size(),
  //     0.5,
  //     0.5,
  //     cv::INTER_NEAREST);

  cv::imshow("Original Mask + Bounding Boxes", bbox_viz);

  // ---------------------------------------------------------------------------
  // Window 3: EXACT 10-pixel eroded mask.
  // ---------------------------------------------------------------------------

  cv::Mat eroded_viz;

  cv::normalize(reference_eroded_mask, eroded_viz, 0, 255, cv::NORM_MINMAX,
                CV_8UC1);

  cv::applyColorMap(eroded_viz, eroded_viz, cv::COLORMAP_JET);

  // cv::resize(
  //     eroded_viz,
  //     eroded_viz,
  //     cv::Size(),
  //     0.5,
  //     0.5,
  //     cv::INTER_NEAREST);

  cv::imshow("10px Eroded Object Mask", eroded_viz);

  // ---------------------------------------------------------------------------
  // Window 4: Eroded mask + inner bounding boxes.
  // ---------------------------------------------------------------------------

  cv::Mat inner_bbox_viz(mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));

  for (size_t object_index = 0; object_index < object_ids.size();
       ++object_index) {
    const uint8_t object_id = object_ids[object_index];

    const cv::Scalar colour = colours[object_index];

    inner_bbox_viz.setTo(colour, reference_eroded_mask == object_id);

    const cv::Rect bbox = reference_inner_bounding_boxes[object_index];

    if (bbox.area() > 0) {
      cv::rectangle(inner_bbox_viz, bbox, colour, 2);

      cv::putText(inner_bbox_viz, std::to_string(static_cast<int>(object_id)),
                  bbox.tl() + cv::Point(0, -5), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  colour, 2);
    }
  }

  // cv::resize(
  //     inner_bbox_viz,
  //     small_inner_bbox_viz,
  //     cv::Size(),
  //     0.5,
  //     0.5,
  //     cv::INTER_NEAREST);

  cv::imshow("10px Eroded Mask + Inner Bounding Boxes", inner_bbox_viz);

  // ---------------------------------------------------------------------------
  // Window 5: Inner border.
  // ---------------------------------------------------------------------------

  cv::Mat inner_border_viz;

  cv::normalize(reference_inner_border, inner_border_viz, 0, 255,
                cv::NORM_MINMAX, CV_8UC1);

  cv::applyColorMap(inner_border_viz, inner_border_viz, cv::COLORMAP_JET);

  // cv::resize(
  //     inner_border_viz,
  //     inner_border_viz,
  //     cv::Size(),
  //     0.5,
  //     0.5,
  //     cv::INTER_NEAREST);

  cv::imshow("10px Inner Border", inner_border_viz);

  // Keep windows open.
  cv::waitKey(0);
  cv::destroyAllWindows();
}

#include <limits>
#include <opencv2/opencv.hpp>
#include <vector>

TEST(ObjectBoundingBoxes, Profile) {
  constexpr int width = 640 * 2;
  constexpr int height = 480 * 2;
  constexpr int num_objects = 8;
  constexpr int iterations = 1000;

  cv::Mat object_mask = cv::Mat::zeros(height, width, CV_8UC1);

  cv::RNG rng(12345);

  ObjectIds object_ids;
  for (uint8_t object_id = 1; object_id <= num_objects; ++object_id) {
    const cv::Point centre(rng.uniform(50, width - 50),
                           rng.uniform(50, height - 50));

    const int radius_x = rng.uniform(20, 70);
    const int radius_y = rng.uniform(20, 70);
    const int num_parts = rng.uniform(2, 6);

    object_ids.push_back(object_id);

    for (int i = 0; i < num_parts; ++i) {
      const cv::Point offset(rng.uniform(-radius_x / 2, radius_x / 2),
                             rng.uniform(-radius_y / 2, radius_y / 2));

      const cv::Size size(rng.uniform(radius_x / 2, radius_x),
                          rng.uniform(radius_y / 2, radius_y));

      cv::ellipse(object_mask, centre + offset, size, rng.uniform(0.0, 180.0),
                  0.0, 360.0, cv::Scalar(object_id), cv::FILLED);
    }
  }

  std::vector<cv::Rect> boxes;
  // Warm up.
  for (int i = 0; i < 100; ++i) {
    vision_tools::getObjectBoundingBoxes(object_mask, object_ids, boxes);
  }

  const auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; ++i) {
    vision_tools::getObjectBoundingBoxes(object_mask, object_ids, boxes);
  }

  const auto end = std::chrono::steady_clock::now();

  const double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  const double average_ms = total_ms / static_cast<double>(iterations);

  std::cout << "Object bounding box profiling:\n"
            << "  Image:      " << width << " x " << height << '\n'
            << "  Objects:    " << num_objects << '\n'
            << "  Iterations: " << iterations << '\n'
            << "  Total:      " << total_ms << " ms\n"
            << "  Average:    " << average_ms << " ms\n";

  // Convert the label mask to a displayable 8-bit image.
  cv::Mat display;
  object_mask.convertTo(display, CV_8UC1, 50.0);

  // Convert to BGR so that bounding boxes can be drawn in colour.
  cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);

  for (int i = 0; i < num_objects; ++i) {
    if (boxes[i].area() <= 0) continue;

    cv::rectangle(display, boxes[i], cv::Scalar(0, 255, 0), 2);

    cv::putText(display, std::to_string(i + 1),
                boxes[i].tl() + cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);
  }

  cv::imshow("Object Masks + Bounding Boxes", display);

  cv::waitKey(0);
  cv::destroyWindow("Object Masks + Bounding Boxes");
}
