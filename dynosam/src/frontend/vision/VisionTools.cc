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

#include "dynosam/frontend/vision/VisionTools.hpp"

#include <algorithm>  // std::set_difference, std::sort
#include <cmath>
#include <execution>
#include <future>
#include <thread>
#include <vector>  // std::vector

#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam_common/Cuda.hpp"
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {

namespace vision_tools {

// void outlierRejectHomography(
//     const cv::Mat& previous_points,
//     const cv::Mat& current_points,
//     cv::Mat& inlier_mask)
// {
//   CHECK_EQ(previous_points.rows, current_points.rows);
//   // Minimum number of points required for RANSAC
//   if(previous_points.rows >= 4) {
//     cv::findHomography(previous_points, current_points, cv::RANSAC, 5.0,
//     inlier_mask);
//   }
//   else {
//     // If not enough points, assume all are outliers
//     inlier_mask = cv::Mat::ones(previous_points.rows, 0, CV_8U);
//   }
// }

// helper function to homography
cv::Mat findHomography(const std::vector<cv::Point2f>& previous,
                       const std::vector<cv::Point2f>& current,
                       double repr_threshold, const int max_iters) {
  CHECK_EQ(previous.size(), current.size());

  // Minimum number of points required for RANSAC
  if (previous.size() >= 4) {
    cv::Mat mask;
    cv::findHomography(previous, current, cv::RANSAC, repr_threshold, mask,
                       max_iters);
    return mask;
  } else {
    // If not enough points, assume all are outliers
    return cv::Mat::zeros(previous.size(), 1, CV_8U);
  }
}

cv::Mat findEssential(const std::vector<cv::Point2f>& previous,
                      const std::vector<cv::Point2f>& current,
                      const cv::Mat& K) {
  CHECK_EQ(previous.size(), current.size());

  // Minimum number of points required for RANSAC
  if (previous.size() >= 4) {
    cv::Mat mask;
    cv::findEssentialMat(previous, current, K, cv::RANSAC, 0.999, 1.0, 500,
                         mask);
    return mask;
  } else {
    // If not enough points, assume all are outliers
    return cv::Mat::ones(previous.size(), 0, CV_8U);
  }
}

void outlierRejectEssential(const std::vector<cv::Point2f>& previous,
                            const std::vector<cv::Point2f>& current,
                            const TrackletIds& tracklet_ids, const cv::Mat& K,
                            std::vector<cv::Point2f>& verified_previous,
                            std::vector<cv::Point2f>& verified_current,
                            TrackletIds& verified_tracklet_ids) {
  CHECK_EQ(tracklet_ids.size(), previous.size());
  const cv::Mat geometric_verification_mask =
      findEssential(previous, current, K);

  for (int i = 0; i < geometric_verification_mask.rows; ++i) {
    if (geometric_verification_mask.at<uchar>(i)) {
      verified_current.push_back(current.at(i));
      verified_previous.push_back(previous.at(i));
      verified_tracklet_ids.push_back(tracklet_ids.at(i));
    }
  }
}

void outlierRejectEssential(const std::vector<cv::Point2f>& previous,
                            const std::vector<cv::Point2f>& current,
                            const cv::Mat& K,
                            std::vector<cv::Point2f>& verified_previous,
                            std::vector<cv::Point2f>& verified_current) {
  const cv::Mat geometric_verification_mask =
      findEssential(previous, current, K);

  for (int i = 0; i < geometric_verification_mask.rows; ++i) {
    if (geometric_verification_mask.at<uchar>(i)) {
      verified_current.push_back(current.at(i));
      verified_previous.push_back(previous.at(i));
    }
  }
}

void outlierRejectHomography(const std::vector<cv::Point2f>& previous,
                             const std::vector<cv::Point2f>& current,
                             const TrackletIds& tracklet_ids,
                             std::vector<cv::Point2f>& verified_previous,
                             std::vector<cv::Point2f>& verified_current,
                             TrackletIds& verified_tracklet_ids) {
  CHECK_EQ(tracklet_ids.size(), previous.size());
  const cv::Mat geometric_verification_mask = findHomography(previous, current);

  for (int i = 0; i < geometric_verification_mask.rows; ++i) {
    if (geometric_verification_mask.at<uchar>(i)) {
      verified_current.push_back(current.at(i));
      verified_previous.push_back(previous.at(i));
      verified_tracklet_ids.push_back(tracklet_ids.at(i));
    }
  }
}

void outlierRejectHomography(const std::vector<cv::Point2f>& previous,
                             const std::vector<cv::Point2f>& current,
                             std::vector<cv::Point2f>& verified_previous,
                             std::vector<cv::Point2f>& verified_current) {
  const cv::Mat geometric_verification_mask = findHomography(previous, current);

  for (int i = 0; i < geometric_verification_mask.rows; ++i) {
    if (geometric_verification_mask.at<uchar>(i)) {
      verified_current.push_back(current.at(i));
      verified_previous.push_back(previous.at(i));
    }
  }
}

// hella vibe-coded!
// experimentally we see up to a 10ms improvement over the previous
// implementation!
ObjectIds getObjectLabelsParallelDynamic(const cv::Mat& image) {
  CV_Assert(image.type() == CV_32SC1);

  const int numThreads = cv::getNumThreads();

  // Each worker thread gets its own isolated bucket vector to store discovered
  // IDs
  std::vector<std::vector<int>> thread_buckets(numThreads);
  for (int t = 0; t < numThreads; ++t) {
    thread_buckets[t].reserve(
        256);  // Allocate small initial workspace per thread
  }

  // Run parallel slice loop over image rows
  cv::parallel_for_(cv::Range(0, image.rows), [&](const cv::Range& range) {
    const int threadId = cv::getThreadNum();
    std::vector<int>& local_bucket = thread_buckets[threadId];

    // Thread-local cache variable to avoid logging identical consecutive pixels
    int last_id = -1;

    for (int row = range.start; row < range.end; ++row) {
      const int* rowPtr = image.ptr<int>(row);
      for (int col = 0; col < image.cols; ++col) {
        const int current_id = rowPtr[col];

        // Fast internal filter: avoid processing if it matches the pixel we
        // *just* checked
        if (current_id == last_id) continue;
        last_id = current_id;

        if (current_id != background_label) {
          local_bucket.push_back(current_id);
        }
      }
    }
  });

  // --- MERGE & DE-DUPLICATE PHASE ---
  // Flatten all thread-local collections into a single target result vector
  size_t total_elements = 0;
  for (int t = 0; t < numThreads; ++t) {
    total_elements += thread_buckets[t].size();
  }

  ObjectIds result;
  result.reserve(total_elements);
  for (int t = 0; t < numThreads; ++t) {
    result.insert(result.end(), thread_buckets[t].begin(),
                  thread_buckets[t].end());
  }

  // In-place sort and deduplicate using standard library algorithms
  std::sort(result.begin(), result.end());
  auto last = std::unique(result.begin(), result.end());
  result.erase(last, result.end());

  return result;
}

ObjectIds getObjectLabels(const cv::Mat& image) {
  return getObjectLabelsParallelDynamic(image);
}

void getObjectBoundingBoxes(const cv::Mat& mask, const ObjectIds& object_ids,
                            std::vector<cv::Rect>& bounding_boxes) {
  CV_Assert(!mask.empty());
  CV_Assert(mask.channels() == 1);

  const int rows = mask.rows;
  const int cols = mask.cols;
  const int num_objects = static_cast<int>(object_ids.size());

  if (num_objects == 0) return;

  // -------------------------------------------------------------------------
  // Build ID -> requested-object-index lookup.
  //
  // This allows the inner pixel loop to avoid an unordered_map lookup.
  // -------------------------------------------------------------------------

  int max_id = 0;

  for (const int id : object_ids) max_id = std::max(max_id, id);

  std::vector<int> id_to_index(max_id + 1, -1);

  for (int i = 0; i < num_objects; ++i) {
    const int id = object_ids[i];

    if (id >= 0) id_to_index[id] = i;
  }

  struct ObjectBoundingBox {
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = -1;
    int max_y = -1;

    inline void update(const int x, const int y) {
      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      max_x = std::max(max_x, x);
      max_y = std::max(max_y, y);
    }

    inline bool valid() const { return max_x >= 0; }

    inline cv::Rect rect() const {
      return cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
    }
  };

  // -------------------------------------------------------------------------
  // Serial implementation for smaller images.
  // -------------------------------------------------------------------------

  constexpr int PARALLEL_PIXEL_THRESHOLD = 640 * 480;

  if (rows * cols < PARALLEL_PIXEL_THRESHOLD) {
    std::vector<ObjectBoundingBox> boxes(num_objects);

    if (mask.type() == CV_8UC1) {
      for (int y = 0; y < rows; ++y) {
        const uchar* row = mask.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x) {
          const int id = row[x];

          if (id <= max_id) {
            const int index = id_to_index[id];

            if (index >= 0) boxes[index].update(x, y);
          }
        }
      }
    } else if (mask.type() == CV_16UC1) {
      for (int y = 0; y < rows; ++y) {
        const uint16_t* row = mask.ptr<uint16_t>(y);

        for (int x = 0; x < cols; ++x) {
          const int id = row[x];

          if (id <= max_id) {
            const int index = id_to_index[id];

            if (index >= 0) boxes[index].update(x, y);
          }
        }
      }
    } else if (mask.type() == CV_32SC1) {
      for (int y = 0; y < rows; ++y) {
        const int* row = mask.ptr<int>(y);

        for (int x = 0; x < cols; ++x) {
          const int id = row[x];

          if (id >= 0 && id <= max_id) {
            const int index = id_to_index[id];

            if (index >= 0) boxes[index].update(x, y);
          }
        }
      }
    } else {
      CV_Error(cv::Error::StsUnsupportedFormat,
               "mask must be CV_8UC1, CV_16UC1 or CV_32SC1");
    }

    std::vector<cv::Rect> result(num_objects);

    for (int i = 0; i < num_objects; ++i) {
      if (boxes[i].valid()) result[i] = boxes[i].rect();
    }

    bounding_boxes = std::move(result);
  }

  // -------------------------------------------------------------------------
  // Parallel implementation.
  // -------------------------------------------------------------------------

  const int nthreads = cv::getNumThreads();

  std::vector<std::vector<ObjectBoundingBox>> thread_boxes(
      nthreads, std::vector<ObjectBoundingBox>(num_objects));

  cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
    const int tid = cv::getThreadNum();

    auto& boxes = thread_boxes[tid];

    if (mask.type() == CV_8UC1) {
      for (int y = range.start; y < range.end; ++y) {
        const uchar* row = mask.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x) {
          const int id = row[x];

          if (id <= max_id) {
            const int index = id_to_index[id];

            if (index >= 0) boxes[index].update(x, y);
          }
        }
      }
    } else if (mask.type() == CV_16UC1) {
      for (int y = range.start; y < range.end; ++y) {
        const uint16_t* row = mask.ptr<uint16_t>(y);

        for (int x = 0; x < cols; ++x) {
          const int id = row[x];

          if (id <= max_id) {
            const int index = id_to_index[id];

            if (index >= 0) boxes[index].update(x, y);
          }
        }
      }
    } else if (mask.type() == CV_32SC1) {
      for (int y = range.start; y < range.end; ++y) {
        const int* row = mask.ptr<int>(y);

        for (int x = 0; x < cols; ++x) {
          const int id = row[x];

          if (id >= 0 && id <= max_id) {
            const int index = id_to_index[id];

            if (index >= 0) boxes[index].update(x, y);
          }
        }
      }
    }
  });

  // -------------------------------------------------------------------------
  // Merge per-thread results.
  // -------------------------------------------------------------------------

  std::vector<ObjectBoundingBox> boxes(num_objects);

  for (int t = 0; t < nthreads; ++t) {
    for (int i = 0; i < num_objects; ++i) {
      const auto& src = thread_boxes[t][i];

      if (!src.valid()) continue;

      auto& dst = boxes[i];

      dst.min_x = std::min(dst.min_x, src.min_x);

      dst.min_y = std::min(dst.min_y, src.min_y);

      dst.max_x = std::max(dst.max_x, src.max_x);

      dst.max_y = std::max(dst.max_y, src.max_y);
    }
  }

  // -------------------------------------------------------------------------
  // Return rectangles in exactly the same order as object_ids.
  // -------------------------------------------------------------------------

  std::vector<cv::Rect> result(num_objects);

  for (int i = 0; i < num_objects; ++i) {
    if (boxes[i].valid()) result[i] = boxes[i].rect();
  }

  bounding_boxes = std::move(result);
}

// std::vector<std::vector<int>> trackDynamic(const FrontendParams& params,
//                                            const Frame& previous_frame,
//                                            Frame::Ptr current_frame) {
//   auto& objects_by_instance_label = current_frame->object_observations_;

//   auto& previous_dynamic_feature_container =
//   previous_frame.dynamic_features_; auto& current_dynamic_feature_container =
//   current_frame->dynamic_features_;

//   ObjectIds instance_labels_to_remove;

//   for (auto& [instance_label, object_observation] :
//   objects_by_instance_label) {
//     double obj_center_depth = 0, sf_min = 100, sf_max = 0, sf_mean = 0,
//            sf_count = 0;
//     std::vector<int> sf_range(10, 0);

//     const size_t num_object_features =
//         object_observation.object_features_.size();
//     // LOG(INFO) << "tracking object observation with instance label " <<
//     // instance_label << " and " << num_object_features << " features";

//     int feature_pairs_valid = 0;
//     int num_found = 0;
//     for (const TrackletId tracklet_id : object_observation.object_features_)
//     {
//       if (previous_dynamic_feature_container.exists(tracklet_id)) {
//         num_found++;
//         CHECK(current_dynamic_feature_container.exists(tracklet_id));

//         Feature::Ptr current_feature =
//             current_dynamic_feature_container.getByTrackletId(tracklet_id);
//         Feature::Ptr previous_feature =
//             previous_dynamic_feature_container.getByTrackletId(tracklet_id);

//         if (!previous_feature->usable()) {
//           current_feature->markInvalid();
//           continue;
//         }

//         // this can happen in situations such as the updateDepths when depths
//         >
//         // thresh are marked invalud
//         if (!current_feature->usable()) {
//           continue;
//         }

//         CHECK(!previous_feature->isStatic());
//         CHECK(!current_feature->isStatic());

//         Landmark lmk_previous =
//         previous_frame.backProjectToWorld(tracklet_id); Landmark lmk_current
//         = current_frame->backProjectToWorld(tracklet_id);

//         Landmark flow_world = lmk_current - lmk_previous;
//         double sf_norm = flow_world.norm();

//         feature_pairs_valid++;

//         if (sf_norm < params.scene_flow_magnitude) sf_count = sf_count + 1;
//         if (sf_norm < sf_min) sf_min = sf_norm;
//         if (sf_norm > sf_max) sf_max = sf_norm;
//         sf_mean = sf_mean + sf_norm;

//         {
//           if (0.0 <= sf_norm && sf_norm < 0.05)
//             sf_range[0] = sf_range[0] + 1;
//           else if (0.05 <= sf_norm && sf_norm < 0.1)
//             sf_range[1] = sf_range[1] + 1;
//           else if (0.1 <= sf_norm && sf_norm < 0.2)
//             sf_range[2] = sf_range[2] + 1;
//           else if (0.2 <= sf_norm && sf_norm < 0.4)
//             sf_range[3] = sf_range[3] + 1;
//           else if (0.4 <= sf_norm && sf_norm < 0.8)
//             sf_range[4] = sf_range[4] + 1;
//           else if (0.8 <= sf_norm && sf_norm < 1.6)
//             sf_range[5] = sf_range[5] + 1;
//           else if (1.6 <= sf_norm && sf_norm < 3.2)
//             sf_range[6] = sf_range[6] + 1;
//           else if (3.2 <= sf_norm && sf_norm < 6.4)
//             sf_range[7] = sf_range[7] + 1;
//           else if (6.4 <= sf_norm && sf_norm < 12.8)
//             sf_range[8] = sf_range[8] + 1;
//           else if (12.8 <= sf_norm && sf_norm < 25.6)
//             sf_range[9] = sf_range[9] + 1;
//         }
//       }
//     }

//     VLOG(10) << "Number feature pairs valid " << feature_pairs_valid
//              << " out of " << num_object_features << " for instance  "
//              << instance_label << " num found " << num_found;

//     // if no points found (i.e tracked)
//     // dont do anything as this is a new object so we cannot say if its
//     dynamic
//     // or not
//     if (num_found == 0) {
//       // TODO: i guess?
//       object_observation.marked_as_moving_ = true;
//     }
//     if (sf_count / num_object_features > params.scene_flow_percentage ||
//         num_object_features < 30)
//     // else if (sf_count/num_object_features>params.scene_flow_percentage ||
//     // num_object_features < 15)
//     {
//       // label this object as static background
//       // LOG(INFO) << "Instance object " << instance_label << " to static for
//       // frame " << current_frame->frame_id_;
//       instance_labels_to_remove.push_back(instance_label);
//     } else {
//       // LOG(INFO) << "Instance object " << instance_label << " marked as
//       // dynamic";
//       object_observation.marked_as_moving_ = true;
//     }
//   }

//   // we do the removal after the iteration so as not to mess up the loop
//   for (const auto label : instance_labels_to_remove) {
//     VLOG(30) << "Removing label " << label;
//     // TODO: this is really really slow!!
//     current_frame->moveObjectToStatic(label);
//     // LOG(INFO) << "Done Removing label " << label;
//   }

//   return std::vector<std::vector<int>>();
// }

bool findObjectBoundingBox(
    const cv::Mat& mask, ObjectId object_id, cv::Rect& detected_rect,
    std::vector<std::vector<cv::Point>>& detected_contours) {
  // cv::Mat mask_copy = mask.clone();

  // cv::Mat obj_mask = (mask_copy == object_id);
  cv::Mat obj_mask = (mask == object_id);

  cv::Mat dilated_obj_mask;
  // dilate to fill any small holes in the mask to get a more complete set of
  // contours
  // cv::Mat dilate_element = cv::getStructuringElement(
  //     cv::MORPH_RECT, cv::Size(1, 11));  // a rectangle of 1*5
  // cv::Mat dilate_element = cv::getStructuringElement(
  //     cv::MORPH_RECT, cv::Size(1, 5));  // a rectangle of 1*5
  static const cv::Mat dilate_element =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 5));
  cv::dilate(obj_mask, dilated_obj_mask, dilate_element, cv::Point(-1, -1));

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(dilated_obj_mask, contours, hierarchy, cv::RETR_TREE,
                   cv::CHAIN_APPROX_NONE);

  detected_contours = std::move(contours);

  if (detected_contours.empty()) {
    detected_rect = cv::Rect();
    return false;
  } else if (detected_contours.size() == 1u) {
    detected_rect = cv::boundingRect(detected_contours.at(0));
  } else {
    std::vector<cv::Rect> rectangles;
    for (auto it : detected_contours) {
      rectangles.push_back(cv::boundingRect(it));
    }
    cv::Rect merged_rect = rectangles[0];
    for (const auto& r : rectangles) {
      merged_rect |= r;
    }
    detected_rect = merged_rect;
  }
  return true;
}

bool findObjectBoundingBox(const cv::Mat& mask, ObjectId object_id,
                           cv::Rect& detected_rect) {
  std::vector<std::vector<cv::Point>> detected_contours;
  auto result =
      findObjectBoundingBox(mask, object_id, detected_rect, detected_contours);
  (void)detected_contours;
  return result;
}

bool findObjectBoundingBox(
    const cv::Mat& mask, ObjectId object_id,
    std::vector<std::vector<cv::Point>>& detected_contours) {
  cv::Rect detected_rect;
  auto result =
      findObjectBoundingBox(mask, object_id, detected_rect, detected_contours);
  (void)detected_rect;
  return result;
}

void shrinkMask(const cv::Mat& mask, cv::Mat& shrunk_mask, int erosion_size) {
  shrunk_mask = cv::Mat::zeros(mask.size(), mask.type());
  shrunk_mask.setTo(background_label);

  const ObjectIds original_object_labels = getObjectLabels(mask);

  const cv::Mat element = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1));

  for (const auto object_id : original_object_labels) {
    cv::Mat obj_mask = (mask == object_id);
    cv::Mat eroded_mask;
    cv::erode(obj_mask, eroded_mask, element);
    shrunk_mask = shrunk_mask.setTo(object_id, eroded_mask);
  }
}

bool findObjectContoursAndBoundingBoxes(
    const cv::Mat& mask, const ObjectIds& object_ids,
    std::vector<std::vector<std::vector<cv::Point>>>& all_contours,
    std::vector<cv::Rect>& bounding_boxes) {
  const auto total_start = std::chrono::steady_clock::now();

  CV_Assert(mask.type() == CV_32SC1);

  all_contours.clear();
  bounding_boxes.clear();

  all_contours.resize(object_ids.size());
  bounding_boxes.resize(object_ids.size());

  if (mask.empty() || object_ids.empty()) {
    // TODO: still fill detection mask?
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
    const ObjectId* row = mask.ptr<ObjectId>(y);

    for (int x = 0; x < mask.cols; ++x) {
      const ObjectId object_id = row[x];

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

void computeObjectMaskBoundaryMaskHelper(
    ObjectBoundaryMaskResult& result, const cv::Mat& mask, int thickness,
    bool use_as_feature_detection_mask,
    std::function<ObjectIds()> get_object_labels) {
  cv::Mat thicc_boarder;  // god im so funny
  cv::Scalar fill_colour;

  thicc_boarder = cv::Mat::zeros(mask.size(), CV_8UC1);
  const int outer_thickness = thickness;

  // background should be 255 as we're can detect in this region and boarder
  // region should be zero
  if (use_as_feature_detection_mask) {
    result.boundary_mask = cv::Mat(mask.size(), CV_8U, cv::Scalar(255));
    fill_colour = cv::Scalar(0);
  } else {
    result.boundary_mask = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));
    fill_colour = cv::Scalar(255);
  }

  cv::Mat viz = cv::Mat(mask.size(), CV_8UC3, cv::Scalar(0));

  result.objects_detected = get_object_labels();
  // this basically just creates a full mask over the existing masks using the
  // detected contours
  for (const auto object_id : result.objects_detected) {
    std::vector<std::vector<cv::Point>> detected_contours;
    // NOTE: if we use the object detection result I guess the discovered
    // rectangle here could be different to detection rectangle!
    cv::Rect detected_rect;
    CHECK(vision_tools::findObjectBoundingBox(mask, object_id, detected_rect,
                                              detected_contours));

    CHECK_LE(object_id, 255);  // works only with uint8 types...
    cv::drawContours(thicc_boarder, detected_contours, -1, object_id,
                     cv::FILLED);
    result.object_bounding_boxes.push_back(detected_rect);
  }

  // Dilate the mask to expand outwards by 'thickness' pixels
  cv::Mat dilated_mask;
  cv::dilate(thicc_boarder, dilated_mask,
             cv::getStructuringElement(
                 cv::MORPH_RECT,
                 cv::Size(2 * outer_thickness + 1, 2 * outer_thickness + 1)));
  // Compute the outer border mask
  cv::Mat thicc_outer_boarder_mask = dilated_mask - thicc_boarder;

  // Additionally erode a little but on the inner-side of the mask
  // This helps get rid of pixels directly on the boarder which often have poor
  // depth
  static constexpr int inner_thickness = 10;
  // Generate the inner border
  cv::Mat eroded_mask;
  cv::erode(thicc_boarder, eroded_mask,
            cv::getStructuringElement(
                cv::MORPH_RECT,
                cv::Size(2 * inner_thickness + 1, 2 * inner_thickness + 1)));
  cv::Mat thicc_inner_boarder_mask = thicc_boarder - eroded_mask;

  // iterate over the objects in the eroded mask again to calculate their
  // bounding box's this will be 'very approximate' area which features can be
  // detected for object j it is approximate because it may not well be
  // approximated by a bounding box
  for (const auto object_id : result.objects_detected) {
    cv::Rect detected_rect;
    vision_tools::findObjectBoundingBox(eroded_mask, object_id, detected_rect);
    result.inner_boarder_object_bounding_boxes.push_back(detected_rect);
  }

  // set boarder pixels to fill colour (e.g. zero if to be used as feature
  // detection mask)
  result.boundary_mask.setTo(fill_colour, thicc_outer_boarder_mask);
  result.boundary_mask.setTo(fill_colour, thicc_inner_boarder_mask);

  result.labelled_boundary_mask = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));
  cv::bitwise_or(thicc_outer_boarder_mask, thicc_inner_boarder_mask,
                 result.labelled_boundary_mask);

  result.is_feature_detection_mask = use_as_feature_detection_mask;
}

// void computeObjectMaskBoundaryMaskHelper(
//     ObjectBoundaryMaskResult& result, const cv::Mat& mask, int thickness,
//     bool use_as_feature_detection_mask,
//     std::function<ObjectIds()> get_object_labels) {

//   std::vector<std::vector<std::vector<cv::Point>>> custom_contours;
//   std::vector<cv::Rect> custom_bounding_boxes;

//   constexpr int inner_thickness = 6;
//   constexpr int contour_padding = 2;

//   cv::Scalar fill_colour;
//   if (use_as_feature_detection_mask) {
//     result.boundary_mask = cv::Mat(mask.size(), CV_8U, cv::Scalar(255));
//     fill_colour = cv::Scalar(0);
//   } else {
//     result.boundary_mask = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));
//     fill_colour = cv::Scalar(255);
//   }

//   const ObjectIds object_ids = get_object_labels();
//   result.objects_detected = object_ids;

//   findObjectContoursAndBoundingBoxes(
//       mask,
//       object_ids,
//       custom_contours,
//       custom_bounding_boxes);

//   // ASSERT_EQ(
//   //     custom_contours.size(),
//   //     object_ids.size());

//   // ASSERT_EQ(
//   //     custom_bounding_boxes.size(),
//   //     object_ids.size());

//   //
//   ---------------------------------------------------------------------------
//   // Reference original object bounding boxes.
//   //
//   ---------------------------------------------------------------------------

//   std::vector<cv::Rect>& reference_bounding_boxes =
//   result.object_bounding_boxes; reference_bounding_boxes.resize(
//   object_ids.size());
//   // (
//       // object_ids.size());

//   for (size_t object_index = 0;
//        object_index < object_ids.size();
//        ++object_index) {
//     const ObjectId object_id =
//         object_ids[object_index];

//     int min_x = mask.cols;
//     int min_y = mask.rows;
//     int max_x = -1;
//     int max_y = -1;

//     for (int y = 0;
//          y < mask.rows;
//          ++y) {
//       const ObjectId* row =
//           mask.ptr<ObjectId>(y);

//       for (int x = 0;
//            x < mask.cols;
//            ++x) {
//         if (row[x] != object_id) {
//           continue;
//         }

//         min_x = std::min(min_x, x);
//         min_y = std::min(min_y, y);
//         max_x = std::max(max_x, x);
//         max_y = std::max(max_y, y);
//       }
//     }

//     reference_bounding_boxes[object_index] =
//         cv::Rect(
//             min_x,
//             min_y,
//             max_x - min_x + 1,
//             max_y - min_y + 1);
//   }

//   LOG(INFO) << "Here";
//   //
//   ---------------------------------------------------------------------------
//   // Reference contours.
//   //
//   // This reproduces the ROI-based contour extraction used by the
//   implementation
//   // under test.
//   //
//   ---------------------------------------------------------------------------
//   // const int outer_thickness = 40;
//   static const cv::Mat contour_dilate_element =
//       cv::getStructuringElement(
//           cv::MORPH_RECT,
//           cv::Size(1, 5));
//   // static const cv::Mat contour_dilate_element =
//   //     cv::getStructuringElement(
//   //         cv::MORPH_RECT,
//   //        cv::Size(2 * outer_thickness + 1, 2 * outer_thickness + 1));

//   std::vector<std::vector<std::vector<cv::Point>>>
//       reference_contours(object_ids.size());

//   LOG(INFO) << "Here";

//   for (size_t object_index = 0;
//        object_index < object_ids.size();
//        ++object_index) {
//     const ObjectId object_id =
//         object_ids[object_index];

//     const cv::Rect bbox =
//         reference_bounding_boxes[object_index];

//     const int x0 = bbox.x;
//     const int x1 = bbox.x + bbox.width;

//     const int y0 =
//         std::max(
//             0,
//             bbox.y - contour_padding);

//     const int y1 =
//         std::min(
//             mask.rows,
//             bbox.y +
//                 bbox.height +
//                 contour_padding);

//     const cv::Rect roi(
//         x0,
//         y0,
//         x1 - x0,
//         y1 - y0);

//     cv::Mat object_mask;

//     cv::compare(
//         mask(roi),
//         object_id,
//         object_mask,
//         cv::CMP_EQ);

//     cv::Mat dilated_object_mask;

//     cv::dilate(
//         object_mask,
//         dilated_object_mask,
//         contour_dilate_element,
//         cv::Point(-1, -1));

//     std::vector<cv::Vec4i> hierarchy;

//     cv::findContours(
//         dilated_object_mask,
//         reference_contours[object_index],
//         hierarchy,
//         cv::RETR_TREE,
//         cv::CHAIN_APPROX_NONE);

//     // Convert ROI coordinates back to global coordinates.
//     for (auto& contour :
//          reference_contours[object_index]) {
//       for (auto& point : contour) {
//         point.x += roi.x;
//         point.y += roi.y;
//       }
//     }
//   }

//   LOG(INFO) << "Here";

//   //
//   ---------------------------------------------------------------------------
//   // Check contours and bounding boxes.
//   //
//   ---------------------------------------------------------------------------

//   for (size_t object_index = 0;
//        object_index < object_ids.size();
//        ++object_index) {
//     const ObjectId object_id =
//         object_ids[object_index];

//     // ASSERT_EQ(
//     //     custom_bounding_boxes[object_index],
//     //     reference_bounding_boxes[object_index])
//     //     << "Bounding box mismatch for object "
//     //     << static_cast<int>(object_id);

//     // ASSERT_EQ(
//     //     custom_contours[object_index].size(),
//     //     reference_contours[object_index].size())
//     //     << "Contour count mismatch for object "
//     //     << static_cast<int>(object_id);

//     // for (size_t contour_index = 0;
//     //      contour_index <
//     //          custom_contours[object_index].size();
//     //      ++contour_index) {
//     //   ASSERT_EQ(
//     //       custom_contours[object_index][contour_index],
//     //       reference_contours[object_index][contour_index])
//     //       << "Contour mismatch for object "
//     //       << static_cast<int>(object_id)
//     //       << ", contour "
//     //       << contour_index;
//     // }
//   }

//   //
//   ---------------------------------------------------------------------------
//   // Reproduce the ORIGINAL inner erosion operation.
//   //
//   // This is intentionally independent of the implementation under test.
//   //
//   // Original:
//   //
//   //   eroded_mask = erode(
//   //       thicc_boarder,
//   //       21x21 kernel);
//   //
//   //
//   ---------------------------------------------------------------------------

//   cv::Mat thicc_boarder =
//       cv::Mat::zeros(
//           mask.size(),
//           CV_8UC1);

//   for (const ObjectId object_id : object_ids) {
//     cv::Mat object_mask;

//     cv::compare(
//         mask,
//         object_id,
//         object_mask,
//         cv::CMP_EQ);

//     thicc_boarder.setTo(
//         cv::Scalar(object_id),
//         object_mask);
//   }

//   LOG(INFO) << "Here";

//   static const cv::Mat inner_element =
//       cv::getStructuringElement(
//           cv::MORPH_RECT,
//           cv::Size(
//               2 * inner_thickness + 1,
//               2 * inner_thickness + 1));

//   cv::Mat reference_eroded_mask;

//   cv::erode(
//       thicc_boarder,
//       reference_eroded_mask,
//       inner_element);

//   // This is exactly the original:
//   //
//   //   thicc_inner_boarder_mask =
//   //       thicc_boarder - eroded_mask;
//   //
//   cv::Mat reference_inner_border =
//       thicc_boarder -
//       reference_eroded_mask;

//   //
//   ---------------------------------------------------------------------------
//   // Calculate reference inner bounding boxes.
//   //
//   // This replaces the original N calls to
//   findObjectBoundingBox(eroded_mask,...)
//   // with ONE traversal of the eroded image.
//   //
//   ---------------------------------------------------------------------------

//   std::array<int, 256> min_x;
//   std::array<int, 256> min_y;
//   std::array<int, 256> max_x;
//   std::array<int, 256> max_y;

//   min_x.fill(mask.cols);
//   min_y.fill(mask.rows);
//   max_x.fill(-1);
//   max_y.fill(-1);

//   for (int y = 0;
//        y < reference_eroded_mask.rows;
//        ++y) {
//     const uint8_t* row =
//         reference_eroded_mask.ptr<uint8_t>(y);

//     for (int x = 0;
//          x < reference_eroded_mask.cols;
//          ++x) {
//       const uint8_t object_id = row[x];

//       if (object_id == 0) {
//         continue;
//       }

//       min_x[object_id] =
//           std::min(min_x[object_id], x);

//       min_y[object_id] =
//           std::min(min_y[object_id], y);

//       max_x[object_id] =
//           std::max(max_x[object_id], x);

//       max_y[object_id] =
//           std::max(max_y[object_id], y);
//     }
//   }

//   LOG(INFO) << "Here";

//   std::vector<cv::Rect>& reference_inner_bounding_boxes =
//   result.inner_boarder_object_bounding_boxes;
//   reference_inner_bounding_boxes.resize(object_ids.size());

//   result.boundary_mask.setTo(fill_colour, reference_eroded_mask);
//   result.labelled_boundary_mask = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));

//   LOG(INFO) << "Here";

//   for (size_t object_index = 0;
//        object_index < object_ids.size();
//        ++object_index) {
//     const uint8_t object_id =
//         static_cast<uint8_t>(object_ids[object_index]);

//     if (max_x[object_id] < 0) {
//       // The object disappeared completely after erosion.
//       reference_inner_bounding_boxes[object_index] =
//           cv::Rect();

//       continue;
//     }

//     //TODO: if disappear make sure it is removed from the boundary mask

//     reference_inner_bounding_boxes[object_index] =
//         cv::Rect(
//             min_x[object_id],
//             min_y[object_id],
//             max_x[object_id] -
//                 min_x[object_id] + 1,
//             max_y[object_id] -
//                 min_y[object_id] + 1);

//     result.labelled_boundary_mask.setTo(
//         object_id,
//         reference_eroded_mask == object_id);
//   }

//   LOG(INFO) << "Here";

//   result.is_feature_detection_mask = use_as_feature_detection_mask;

//   //
//   ---------------------------------------------------------------------------
//   // Print inner bounding boxes.
//   //
//   ---------------------------------------------------------------------------

//   for (size_t object_index = 0;
//        object_index < object_ids.size();
//        ++object_index) {
//     const uint8_t object_id =
//         object_ids[object_index];

//     const cv::Rect& bbox =
//         reference_inner_bounding_boxes[object_index];

//     std::cout
//         << "Object "
//         << static_cast<int>(object_id)
//         << " inner bbox: "
//         << bbox.x << ", "
//         << bbox.y << ", "
//         << bbox.width << " x "
//         << bbox.height
//         << std::endl;
//   }

//   //
//   ---------------------------------------------------------------------------
//   // Debug visualisation.
//   //
//   ---------------------------------------------------------------------------

//   // cv::RNG viz_rng(54321);

//   // std::vector<cv::Scalar> colours;

//   // for (size_t i = 0;
//   //      i < object_ids.size();
//   //      ++i) {
//   //   colours.emplace_back(
//   //       viz_rng.uniform(50, 255),
//   //       viz_rng.uniform(50, 255),
//   //       viz_rng.uniform(50, 255));
//   // }

//   //
//   ---------------------------------------------------------------------------
//   // Window 1: Original labelled mask.
//   //
//   ---------------------------------------------------------------------------

//   // cv::Mat original_viz;

//   // cv::normalize(
//   //     mask,
//   //     original_viz,
//   //     0,
//   //     255,
//   //     cv::NORM_MINMAX,
//   //     CV_8UC1);

//   // cv::applyColorMap(
//   //     original_viz,
//   //     original_viz,
//   //     cv::COLORMAP_JET);

//   // cv::imshow(
//   //     "Original Object Mask",
//       // original_viz);

//   //
//   ---------------------------------------------------------------------------
//   // Window 2: Original mask + detected bounding boxes.
//   //
//   ---------------------------------------------------------------------------

//   // cv::Mat bbox_viz(
//   //     mask.size(),
//   //     CV_8UC3,
//   //     cv::Scalar(0, 0, 0));

//   // for (size_t object_index = 0;
//   //      object_index < object_ids.size();
//   //      ++object_index) {
//   //   const uint8_t object_id =
//   //       object_ids[object_index];

//   //   const cv::Scalar colour =
//   //       colours[object_index];

//   //   bbox_viz.setTo(
//   //       colour,
//   //       mask == object_id);

//   //   cv::rectangle(
//   //       bbox_viz,
//   //       custom_bounding_boxes[object_index],
//   //       colour,
//   //       2);

//   //   cv::putText(
//   //       bbox_viz,
//   //       std::to_string(
//   //           static_cast<int>(object_id)),
//   //       custom_bounding_boxes[object_index].tl() +
//   //           cv::Point(0, -5),
//   //       cv::FONT_HERSHEY_SIMPLEX,
//   //       0.6,
//   //       colour,
//   //       2);
//   // }

//   // cv::resize(
//   //     bbox_viz,
//   //     small_bbox_viz,
//   //     cv::Size(),
//   //     0.5,
//   //     0.5,
//   //     cv::INTER_NEAREST);

//   // cv::imshow(
//   //     "Original Mask + Bounding Boxes",
//   //     bbox_viz);

//   //
//   ---------------------------------------------------------------------------
//   // Window 3: EXACT 10-pixel eroded mask.
//   //
//   ---------------------------------------------------------------------------

//   // cv::Mat eroded_viz;

//   // cv::normalize(
//   //     reference_eroded_mask,
//   //     eroded_viz,
//   //     0,
//   //     255,
//   //     cv::NORM_MINMAX,
//   //     CV_8UC1);

//   // cv::applyColorMap(
//   //     eroded_viz,
//   //     eroded_viz,
//   //     cv::COLORMAP_JET);

//   // // cv::resize(
//   // //     eroded_viz,
//   // //     eroded_viz,
//   // //     cv::Size(),
//   // //     0.5,
//   // //     0.5,
//   // //     cv::INTER_NEAREST);

//   // cv::imshow(
//   //     "10px Eroded Object Mask",
//   //     eroded_viz);

//   //
//   ---------------------------------------------------------------------------
//   // Window 4: Eroded mask + inner bounding boxes.
//   //
//   ---------------------------------------------------------------------------

//   // cv::Mat inner_bbox_viz(
//   //     mask.size(),
//   //     CV_8UC3,
//   //     cv::Scalar(0, 0, 0));

//   // for (size_t object_index = 0;
//   //      object_index < object_ids.size();
//   //      ++object_index) {
//   //   const uint8_t object_id =
//   //       object_ids[object_index];

//   //   const cv::Scalar colour =
//   //       colours[object_index];

//   //   inner_bbox_viz.setTo(
//   //       colour,
//   //       reference_eroded_mask == object_id);

//   //   const cv::Rect bbox =
//   //       reference_inner_bounding_boxes[object_index];

//   //   if (bbox.area() > 0) {
//   //     cv::rectangle(
//   //         inner_bbox_viz,
//   //         bbox,
//   //         colour,
//   //         2);

//   //     cv::putText(
//   //         inner_bbox_viz,
//   //         std::to_string(
//   //             static_cast<int>(object_id)),
//   //         bbox.tl() +
//   //             cv::Point(0, -5),
//   //         cv::FONT_HERSHEY_SIMPLEX,
//   //         0.6,
//   //         colour,
//   //         2);
//   //   }
//   // }

//   // // cv::resize(
//   // //     inner_bbox_viz,
//   // //     small_inner_bbox_viz,
//   // //     cv::Size(),
//   // //     0.5,
//   // //     0.5,
//   // //     cv::INTER_NEAREST);

//   // cv::imshow(
//   //     "10px Eroded Mask + Inner Bounding Boxes",
//   //     inner_bbox_viz);

//   // //
//   ---------------------------------------------------------------------------
//   // // Window 5: Inner border.
//   // //
//   ---------------------------------------------------------------------------

//   // cv::Mat inner_border_viz;

//   // cv::normalize(
//   //     reference_inner_border,
//   //     inner_border_viz,
//   //     0,
//   //     255,
//   //     cv::NORM_MINMAX,
//   //     CV_8UC1);

//   // cv::applyColorMap(
//   //     inner_border_viz,
//   //     inner_border_viz,
//   //     cv::COLORMAP_JET);

// }

void computeObjectMaskBoundaryMask(ObjectBoundaryMaskResult& result,
                                   const cv::Mat& mask, int thickness,
                                   bool use_as_feature_detection_mask) {
  computeObjectMaskBoundaryMaskHelper(
      result, mask, thickness, use_as_feature_detection_mask,
      [&mask]() -> ObjectIds { return vision_tools::getObjectLabels(mask); });
}

void computeObjectMaskBoundaryMask(
    ObjectBoundaryMaskResult& result,
    const ObjectDetectionResult& detection_result, int thickness,
    bool use_as_feature_detection_mask) {
  if (detection_result.num() == 0) {
    return;
  }

  computeObjectMaskBoundaryMaskHelper(result, detection_result.labelled_mask,
                                      thickness, use_as_feature_detection_mask,
                                      [&detection_result]() -> ObjectIds {
                                        return detection_result.objectIds();
                                      });
}

void relabelMasks(const cv::Mat& mask, cv::Mat& relabelled_mask,
                  const ObjectIds& old_labels, const ObjectIds& new_labels) {
  if (old_labels.size() != new_labels.size()) {
    throw std::invalid_argument(
        "Old mask and new mask must have the same size");
  }

  // Create a map from old mask to new mask
  std::unordered_map<ObjectId, ObjectId> label_map;
  for (size_t i = 0; i < old_labels.size(); ++i) {
    label_map[old_labels[i]] = new_labels[i];
  }

  mask.copyTo(relabelled_mask);
  // / Relabel the pixels
  for (int r = 0; r < relabelled_mask.rows; ++r) {
    for (int c = 0; c < relabelled_mask.cols; ++c) {
      ObjectId pixelValue = relabelled_mask.at<ObjectId>(r, c);
      if (label_map.find(pixelValue) != label_map.end()) {
        relabelled_mask.at<ObjectId>(r, c) = label_map[pixelValue];
      }
    }
  }
}

gtsam::FastMap<ObjectId, Histogram> makeTrackletLengthHistorgram(
    const Frame::Ptr frame, const std::vector<size_t>& bins) {
  // one for every object + 1 for static points
  gtsam::FastMap<ObjectId, Histogram> histograms;

  const auto& dyamic_features = frame->dynamic_features_;
  auto itr = dyamic_features.beginObjectIterator();
  for (; itr != dyamic_features.endObjectIterator(); itr++) {
    auto [object_id, feature_per_object] = *itr;

    Histogram hist(bh::make_histogram(bh::axis::variable<>(bins)));
    hist.name_ = "tacklet-length-" + std::to_string(object_id);

    for (auto feature : feature_per_object) {
      // const Feature::Ptr feature =
      // dyamic_features.getByTrackletId(tracklet_id);
      CHECK(feature);
      CHECK_EQ(feature->objectId(), object_id);
      if (feature->usable()) {
        hist.histogram_(feature->age());
      }
    }

    histograms.insert2(object_id, hist);
  }

  // collect dynamic features
  // for (const auto& [object_id, observations] :
  // frame->getObjectObservations()) {
  //   Histogram hist(bh::make_histogram(bh::axis::variable<>(bins)));
  //   hist.name_ = "tacklet-length-" + std::to_string(object_id);

  //   // for (auto tracklet_id : observations.object_features) {
  //   //   // const Feature::Ptr feature = frame->at(tracklet_id);
  //   //   // CHECK(feature);
  //   //   // if (feature->usable()) {
  //   //   //   hist.histogram_(feature->age());
  //   //   // }
  //   // }

  //   histograms.insert2(object_id, hist);
  // }

  // collect static features
  Histogram static_hist(bh::make_histogram(bh::axis::variable<>(bins)));
  static_hist.name_ = "tacklet-length-0";
  for (const auto& static_feature : frame->static_features_.usableIterator()) {
    static_hist.histogram_(static_feature->age());
  }
  histograms.insert2(background_label, static_hist);
  return histograms;
}

cv::Mat depthTo3D(const ImageWrapper<ImageType::Depth>& depth_image,
                  const cv::Mat& K) {
  const cv::Mat& depth_map = depth_image;
  int H = depth_map.rows;
  int W = depth_map.cols;

  // Camera intrinsic parameters
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  // Generate pixel grid
  cv::Mat u_grid, v_grid;
  cv::Mat u = cv::Mat::zeros(H, W, CV_64F);
  cv::Mat v = cv::Mat::zeros(H, W, CV_64F);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      u.at<double>(y, x) = static_cast<double>(x);
      v.at<double>(y, x) = static_cast<double>(y);
    }
  }

  // Normalize pixel coordinates by the intrinsic matrix
  cv::Mat X_norm = (u - cx) / fx;
  cv::Mat Y_norm = (v - cy) / fy;

  // Depth map scaling for 3D coordinates
  cv::Mat X_3D = X_norm.mul(depth_map);
  cv::Mat Y_3D = Y_norm.mul(depth_map);

  cv::Mat Z_3D = depth_map.clone();

  std::vector<cv::Mat> channels = {X_3D, Y_3D, Z_3D};
  cv::Mat point_cloud;
  cv::merge(channels, point_cloud);

  return point_cloud;  // 3-channel float matrix (H x W x 3)
}

void writeOutProjectMaskAndDepthMap(
    const ImageWrapper<ImageType::Depth>& depth_image,
    const ImageWrapper<ImageType::SemanticMask>& mask_image,
    const Camera& camera, FrameId frame_id) {
  cv::Mat point_cloud =
      depthTo3D(depth_image, camera.getParams().getCameraMatrix());

  const cv::Mat& mask = mask_image;

  // new mask of double type to match the type of the output cloud
  cv::Mat mask_double;
  mask.copyTo(mask_double);
  mask_double.convertTo(mask_double, CV_64F);

  std::vector<cv::Mat> channels = {point_cloud, mask_double};

  cv::Mat projected_cloud;
  cv::merge(channels, projected_cloud);

  static const auto folder_name = "project_mask";
  const std::string output_folder = getOutputFilePath(folder_name);

  // create write out directly if it does not exist
  createDirectory(output_folder);

  const std::string file_name =
      output_folder + "/" + std::to_string(frame_id) + ".yml";
  cv::FileStorage file(file_name, cv::FileStorage::WRITE);
  file << "matrix" << projected_cloud;
  file.release();
}

// TODO: specifically this is one type of noise using a specific RGBD
// measurement model
std::pair<gtsam::Vector3, gtsam::Matrix33> backProjectAndCovariance(
    const Feature& feature, const Camera& camera, double pixel_sigma,
    double depth_sigma) {
  const auto gtsam_camera = camera.getImplCamera();
  const auto keypoint = feature.keypoint();
  const auto u = keypoint(0);
  const auto v = keypoint(1);

  CHECK(feature.hasDepth());
  const auto depth = feature.depth();
  const auto& cam_params = camera.getParams();
  const auto fx = cam_params.fx();
  const auto fy = cam_params.fy();
  const auto cx = cam_params.cu();
  const auto cy = cam_params.cv();

  // Jacobian J of backprojection w.r.t. (u, v, d) assuming pinhole camera
  gtsam::Matrix33 J;
  J << depth / fx, 0, (u - cx) / fx, 0, depth / fy, (v - cy) / fy, 0, 0, 1;

  double pixel_sigma2 = pixel_sigma * pixel_sigma;
  double depth_sigma2 = depth_sigma * depth_sigma;
  gtsam::Matrix33 sigma_uvd =
      (Eigen::Vector3d(pixel_sigma2, pixel_sigma2, depth_sigma2)).asDiagonal();

  // Propagate to 3D covariance
  gtsam::Matrix33 sigma_3d = J * sigma_uvd * J.transpose();

  // Back project point
  gtsam::Point3 landmark = gtsam_camera->backproject(keypoint, depth);
  return {landmark, sigma_3d};
}

// void writeOutProjectMaskAndDepthMap(const ImageWrapper<ImageType::Depth>&
// depth_image, const ImageWrapper<ImageType::MotionMask>& mask_image, const
// Camera& camera, FrameId frame_id) {
//   writeOutProjectMaskAndDepthMap(depth_image,
//   ImageWrapper<ImageType::SemanticMask>(static_cast<const
//   cv::Mat&>(mask_image)), camera, frame_id);
// }

}  // namespace vision_tools

// void RGBDProcessor::updateMovingObjects(const Frame& previous_frame,
// Frame::Ptr current_frame,  cv::Mat& debug) const {
//   const cv::Mat& rgb =
//   current_frame->tracking_images_.get<ImageType::RGBMono>();

//   rgb.copyTo(debug);

//   const gtsam::Pose3& previous_pose = previous_frame.T_world_camera_;
//   const gtsam::Pose3& current_pose = current_frame->T_world_camera_;

//   const auto previous_dynamic_feature_container =
//   previous_frame.dynamic_features_; const auto
//   current_dynamic_feature_container = current_frame->dynamic_features_;

//   //iterate over each object seen in the previous frame and collect features
//   in current and previous frames to determine scene flow for(auto&
//   [object_id, current_object_observation] :
//   current_frame->object_observations_) {

//     int object_track_count = 0; //number of tracked points on the object
//     int sf_count = 0; //number of points on the object with a sufficient
//     scene flow thresh

//     const TrackletIds& object_features =
//     current_object_observation.object_features_; for(const auto tracklet_id :
//     object_features) {
//       if(previous_dynamic_feature_container.exists(tracklet_id)) {
//         CHECK(current_dynamic_feature_container.exists(tracklet_id));

//         Feature::Ptr current_feature =
//         current_dynamic_feature_container.getByTrackletId(tracklet_id);
//         Feature::Ptr previous_feature =
//         previous_dynamic_feature_container.getByTrackletId(tracklet_id);

//         if(!previous_feature->usable()) {
//           current_feature->markInvalid();
//           continue;
//         }

//         Landmark lmk_previous, lmk_current;
//         camera_->backProject(previous_feature->keypoint_,
//         previous_feature->depth_, &lmk_previous, previous_pose);
//         camera_->backProject(current_feature->keypoint_,
//         current_feature->depth_, &lmk_current, current_pose);

//         Landmark flow_world = lmk_previous - lmk_current;
//         double sf_norm = flow_world.norm();

//         if(sf_norm > params_.scene_flow_magnitude) {
//           sf_count++;
//         }

//         object_track_count++;
//       }
//     }

//     if(sf_count < 50) {
//       continue;
//     }
//     double average_flow_count = (double)sf_count /
//     (double)object_track_count;

//     LOG(INFO) << "Num points that are dynamic " << average_flow_count << "/"
//     << params_.scene_flow_percentage << " for object " << object_id;
//     if(average_flow_count > params_.scene_flow_percentage) {
//       current_object_observation.marked_as_moving_ = true;

//       static const cv::Scalar blue(255, 0, 0);

//       for(TrackletId track : object_features) {
//         Feature::Ptr current_feature =
//         current_dynamic_feature_container.getByTrackletId(track); const
//         Keypoint& px = current_feature->keypoint_; cv::circle(debug,
//         utils::gtsamPointToCV(px), 6, blue, 1);
//       }

//       //only debug stuff

//     }

//   }

// }

void determineOutlierIds(const TrackletIds& inliers,
                         const TrackletIds& tracklets, TrackletIds& outliers) {
  VLOG_IF(1, inliers.size() > tracklets.size())
      << "Usage warning: inlier size (" << inliers.size()
      << ") > tracklets size (" << tracklets.size()
      << "). Are you parsing inliers as tracklets incorrectly?";
  outliers.clear();
  TrackletIds inliers_sorted(inliers.size()),
      tracklets_sorted(tracklets.size());
  std::copy(inliers.begin(), inliers.end(), inliers_sorted.begin());
  std::copy(tracklets.begin(), tracklets.end(), tracklets_sorted.begin());

  std::sort(inliers_sorted.begin(), inliers_sorted.end());
  std::sort(tracklets_sorted.begin(), tracklets_sorted.end());

  // full set A (tracklets) must be first and inliers MUST be a subset of A for
  // the set_difference function to work
  std::set_difference(tracklets_sorted.begin(), tracklets_sorted.end(),
                      inliers_sorted.begin(), inliers_sorted.end(),
                      std::inserter(outliers, outliers.begin()));
}

}  // namespace dyno
