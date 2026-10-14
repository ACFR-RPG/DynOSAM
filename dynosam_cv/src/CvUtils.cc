#include "dynosam_cv/CvUtils.hpp"

#include "dynosam_common/Cuda.hpp"
#include "dynosam_common/viz/Colour.hpp"

namespace dyno {

template <>
std::string to_string<cv::Size>(const cv::Size& t) {
  return "[h=" + std::to_string(t.height) + " w=" + std::to_string(t.width) +
         "]";
}

// / helper function to homography
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
                      const std::vector<cv::Point2f>& current, const cv::Mat& K,
                      double threshold, const int max_iters) {
  CHECK_EQ(previous.size(), current.size());

  // Minimum number of points required for Essential matrix
  if (previous.size() >= 5) {
    cv::Mat mask;
    cv::findEssentialMat(previous, current, K, cv::RANSAC, 0.999, threshold,
                         max_iters, mask);
    return mask;
  } else {
    // If not enough points, assume all are outliers
    return cv::Mat::zeros(previous.size(), 1, CV_8U);
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

namespace utils {

bool opencvCudaAvailable(std::string* message) {
  if constexpr (isOpencvCudaEnabled()) {
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
      if (message)
        *message = "OpenCV CUDA support found. Devices available: " +
                   std::to_string(cv::cuda::getCudaEnabledDeviceCount());
      return true;
    } else {
      if (message)
        *message =
            "OpenCV CUDA support found but No CUDA support or no CUDA devices";
      return false;
    }
  } else {
    if (message)
      *message =
          "OPENCV CUDA not detected from OpenCV itself. Check it has been "
          "compiled correctly!";
    return false;
  }
}

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
              cv::Scalar(255, 255, 255), kThickness, cv::LINE_AA);
  // draw bounding box with line thickness
  cv::rectangle(image, bounding_box, colour, bb_thickness, cv::LINE_AA);
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
