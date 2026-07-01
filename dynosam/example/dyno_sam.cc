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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <Eigen/Dense>
#include <png++/png.hpp>

#include "dynosam/dataprovider/ClusterSlamDataProvider.hpp"
#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "dynosam/dataprovider/TartanAirShibuya.hpp"
#include "dynosam/dataprovider/ViodeDataProvider.hpp"
#include "dynosam/dataprovider/VirtualKittiDataProvider.hpp"
#include "dynosam/frontend/anms/anms/nanoflann.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/OccupancyGrid2D.hpp"
#include "dynosam/pipeline/PipelineManager.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_nn/PyObjectDetector.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/ImageContainer.hpp"

DEFINE_string(path_to_kitti, "/root/data/kitti", "Path to KITTI dataset");
// TODO: (jesse) many better ways to do this with ros - just for now
DEFINE_string(
    params_path, "dynosam/params",
    "Path to the folder containing the yaml files with the VIO parameters.");

#include "dynosam/dataprovider/DynopetsDataProvider.hpp"
#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "dynosam/dataprovider/OMDDataProvider.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"

// // 1. Define the nanoflann adapter interface for dynamically growing
// cv::Point2f vectors struct PointCloudAdapter {
//     const std::vector<cv::Point2f>& points;

//     inline size_t kf_get_point_count() const { return points.size(); }

//     inline float kf_get_pt(const size_t idx, const size_t dim) const {
//         if (dim == 0) return points[idx].x;
//         return points[idx].y;
//     }

//     // Optional bounding box method
//     template <class BBOX> bool kf_get_box(BBOX& /* bb */) const { return
//     false; }
// };

// // Define our dynamic 2D KD-Tree type
// using KDTreeType = nanoflann::KDTreeSingleIndexDynamicAdaptor<
//     nanoflann::L2_Simple_Adaptor<float, PointCloudAdapter>,
//     PointCloudAdapter,
//     2 /* dimensions */
// >;

using namespace dyno;

// Structure to temporarily hold corner candidates for sorting
struct CornerCandidate {
  cv::Point2f pt;
  float score;

  // Sort descending by score
  bool operator>(const CornerCandidate& other) const {
    return score > other.score;
  }
};

std::vector<std::vector<cv::Point2f>> goodFeaturesToTrackBatched(
    const cv::Mat& src, const std::vector<cv::Mat>& masks,
    const std::vector<int>& maxCorners, const std::vector<float>& minDistances,
    float qualityLevel = 0.01, int blockSize = 3) {
  using namespace dyno;
  utils::ChronoTimingStats t("gfft_batch");
  CV_Assert(src.type() == CV_8UC1 || src.type() == CV_32FC1);
  CV_Assert(masks.size() == maxCorners.size());

  // --- STEP 1: Compute the Eigenvalue Map ONCE for the whole frame ---
  cv::Mat eig;
  // cornerMinEigenVal handles the Sobel derivatives internally in a highly
  // optimized pass
  utils::ChronoTimingStats t1("corner_min_eigen");
  cv::cornerMinEigenVal(src, eig, blockSize, 3);
  t1.stop();

  // Find the global maximum corner score across the entire image
  double maxVal = 0;
  cv::minMaxLoc(eig, nullptr, &maxVal);

  // Establish the baseline absolute threshold based on global max quality
  const float threshold = static_cast<float>(maxVal * qualityLevel);

  // --- STEP 2: Local Non-Maximum Suppression (NMS) via Dilation ---
  // OpenCV's internal GFTT uses a dilation trick to find local maxima
  // efficiently
  utils::ChronoTimingStats t2("dilate");
  cv::Mat localMax;
  cv::dilate(eig, localMax, cv::Mat());
  t2.stop();

  std::vector<std::vector<cv::Point2f>> batchedResults(masks.size());

  utils::ChronoTimingStats t3("masks_loop");
  // --- STEP 3: Process Each Mask Independently to Fulfill Specific Quotas ---
  cv::parallel_for_(
      cv::Range(0, static_cast<int>(masks.size())),
      [&](const cv::Range& range) {
        for (int m = range.start; m < range.end; ++m) {
          const cv::Mat& mask = masks[m];
          const int maxFeatureCount = maxCorners[m];
          const int minDistance = minDistances[m];

          if (maxFeatureCount <= 0) continue;

          const float minDistanceSq = minDistance * minDistance;

          /// Pre-calculate spatial grid properties based on the image size
          const int cell_size = cvRound(minDistance);
          const float inv_cell_size = 1.0f / static_cast<float>(cell_size);
          const int grid_width = (src.cols + cell_size - 1) / cell_size;
          const int grid_height = (src.rows + cell_size - 1) / cell_size;

          // Thread-local vector to avoid memory contention across cores
          std::vector<CornerCandidate> candidates;
          // Pre-allocate a reasonable chunk of memory to prevent vector
          // reallocations
          candidates.reserve(256);

          // Cached matrix dimensional details
          const int rows = eig.rows;
          const int cols = eig.cols;

          // Collect local peaks inside mask boundaries
          for (int y = 0; y < rows; ++y) {
            const float* eig_ptr = eig.ptr<float>(y);
            const float* max_ptr = localMax.ptr<float>(y);
            const uchar* mask_ptr = mask.empty() ? nullptr : mask.ptr<uchar>(y);

            for (int x = 0; x < cols; ++x) {
              const float val = eig_ptr[x];

              if (val > threshold && val == max_ptr[x] &&
                  (!mask_ptr || mask_ptr[x])) {
                candidates.push_back(
                    {cv::Point2f(static_cast<float>(x), static_cast<float>(y)),
                     val});
              }
            }
          }

          if (candidates.empty()) continue;

          // Parallel sort if candidate list is huge, otherwise standard sort is
          // fine
          std::sort(candidates.begin(), candidates.end(),
                    std::greater<CornerCandidate>());

          // Reference directly into our thread-safe outer preallocated matrix
          // structure
          std::vector<cv::Point2f>& acceptedCorners = batchedResults[m];
          acceptedCorners.reserve(maxFeatureCount);

          std::vector<int> grid_heads(grid_width * grid_height, -1);
          std::vector<int> next_point_idx;
          next_point_idx.reserve(maxFeatureCount);

          for (const auto& candidate : candidates) {
            if (static_cast<int>(acceptedCorners.size()) >= maxFeatureCount) {
              break;
            }

            // Calculate current grid cell coordinates using fast multiplication
            int x_cell = static_cast<int>(candidate.pt.x * inv_cell_size);
            int y_cell = static_cast<int>(candidate.pt.y * inv_cell_size);

            // Enforce 3x3 search bounds around the center cell
            int x1 = std::max(0, x_cell - 1);
            int y1 = std::max(0, y_cell - 1);
            int x2 = std::min(grid_width - 1, x_cell + 1);
            int y2 = std::min(grid_height - 1, y_cell + 1);

            bool good = true;

            for (int yy = y1; yy <= y2; ++yy) {
              for (int xx = x1; xx <= x2; ++xx) {
                int cell_idx = yy * grid_width + xx;
                int p_idx = grid_heads[cell_idx];

                // Traverse the flat linked list for this specific cell bucket
                while (p_idx != -1) {
                  const cv::Point2f& accepted = acceptedCorners[p_idx];
                  float dx = candidate.pt.x - accepted.x;
                  float dy = candidate.pt.y - accepted.y;

                  if (dx * dx + dy * dy < minDistanceSq) {
                    good = false;
                    goto distance_check_failed;
                  }
                  p_idx =
                      next_point_idx[p_idx];  // Move to next point in bucket
                }
              }
            }

          distance_check_failed:
            if (good) {
              int center_cell_idx = y_cell * grid_width + x_cell;

              // Insert current point index to the head of this cell's linked
              // list
              next_point_idx.push_back(grid_heads[center_cell_idx]);
              grid_heads[center_cell_idx] =
                  static_cast<int>(acceptedCorners.size());

              acceptedCorners.push_back(candidate.pt);
            }
          }

          // Enforce Distance Constraints per Mask Area
          // for (const auto& candidate : candidates) {
          //     if (static_cast<int>(acceptedCorners.size()) >=
          //     maxFeatureCount) {
          //         break;
          //     }

          //     bool keep = true;
          //     // Cache friendly linear iteration
          //     for (const auto& accepted : acceptedCorners) {
          //         float dx = candidate.pt.x - accepted.x;
          //         float dy = candidate.pt.y - accepted.y;
          //         if ((dx * dx + dy * dy) < minDistanceSq) {
          //             keep = false;
          //             break;
          //         }
          //     }

          //     if (keep) {
          //         acceptedCorners.push_back(candidate.pt);
          //     }
          // }
        }
      });

  return batchedResults;
}

/**
 * @brief High-performance KLT tracker that flattens batched features into a
 * single OpenCV call to maximize SIMD efficiency, then unflattens the surviving
 * tracks.
 * * @param prevImg      Grayscale source frame (CV_8UC1)
 * @param nextImg      Grayscale target frame (CV_8UC1)
 * @param prevBatched  The tracked features from the previous frame, grouped by
 * mask/object ID
 * @return std::vector<std::vector<cv::Point2f>> Cleanly unflattened features
 * tracking into nextImg
 */
gtsam::FastMap<ObjectId, std::vector<cv::Point2f>> trackFeaturesUnified(
    const cv::Mat& prevImg, const cv::Mat& nextImg,
    const gtsam::FastMap<ObjectId, std::vector<cv::Point2f>>& prevBatched,
    const cv::Mat& currObjectMask) {
  CV_Assert(prevImg.type() == CV_8UC1 && nextImg.type() == CV_8UC1);

  // 1. --- FLATTENING PHASE ---
  // Count total points across all masks to make a single allocation
  size_t totalPoints = 0;
  for (const auto& [_, vec] : prevBatched) {
    totalPoints += vec.size();
  }

  struct Tracklet2DVectors {
    std::vector<cv::Point2f> current;
    std::vector<cv::Point2f> previous;
  };

  gtsam::FastMap<ObjectId, Tracklet2DVectors> tracks_per_object;
  gtsam::FastMap<ObjectId, std::vector<cv::Point2f>> nextBatched;

  if (totalPoints == 0) return nextBatched;  // Nothing to track

  std::vector<cv::Point2f> flatPrev;
  flatPrev.reserve(totalPoints);

  // Track which object ID (mask index) each flattened point belongs to
  dyno::ObjectIds pointOwnerId;
  pointOwnerId.reserve(totalPoints);

  for (const auto& [j, vec] : prevBatched) {
    for (const auto& pt : vec) {
      // could do this with vector operations
      flatPrev.push_back(pt);
      pointOwnerId.push_back(j);
    }
    // Estimate allocation space (assuming most features survive)
    // nextBatched[j].reserve(vec.size());
    Tracklet2DVectors vecs;
    vecs.current.reserve(vec.size());
    vecs.previous.reserve(vec.size());
    tracks_per_object[j] = vecs;
  }
  // for (size_t m = 0; m < prevBatched.size(); ++m) {
  //   LOG(INFO) << "Prev object id " << prevObjectIds[m];
  //     for (const auto& pt : prevBatched[m]) {
  //         flatPrev.push_back(pt);
  //         pointOwnerId.push_back(m);
  //     }
  // }

  // 2. --- SINGLE-PASS KLT EXECUTION ---
  std::vector<cv::Point2f> flatNext = flatPrev;
  std::vector<uchar> status;
  std::vector<float> err;

  const cv::Size winSize(21, 21);
  const int maxLevel = 3;
  const cv::TermCriteria criteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

  // One single call allows OpenCV to run hot loops across contiguous memory
  // blocks
  cv::calcOpticalFlowPyrLK(prevImg, nextImg, flatPrev, flatNext, status, err,
                           winSize, maxLevel, criteria);

  // 3. --- UNFLATTENING PHASE ---
  for (size_t i = 0; i < flatPrev.size(); ++i) {
    // Check if KLT tracking succeeded and point remains inside image boundaries
    // use 2i to check image boundaries
    cv::Point2i flextNextInt = static_cast<cv::Point2i>(flatNext[i]);
    if (status[i] && err[i] < 10.0f && flextNextInt.x >= 0 &&
        flextNextInt.x < nextImg.cols && flextNextInt.y >= 0 &&
        flextNextInt.y < nextImg.rows) {
      auto j = pointOwnerId[i];

      // LOG(INFO) << "obj j " << j << " with curr object mask " <<
      // currObjectMask.at<dyno::ObjectId>(flatNext[i]);

      if (j != currObjectMask.at<dyno::ObjectId>(flextNextInt)) {
        continue;
      }

      tracks_per_object[j].current.push_back(flatNext[i]);
      tracks_per_object[j].previous.push_back(flatPrev[i]);
      // nextBatched[j].push_back(flatNext[i]);
    }
  }

  // homograph adds about 1ms!
  for (const auto& [object_id, good_tracks] : tracks_per_object) {
    Tracklet2DVectors verified_tracks;
    verified_tracks.current.reserve(good_tracks.current.size());
    verified_tracks.previous.reserve(good_tracks.previous.size());

    vision_tools::outlierRejectHomography(
        good_tracks.previous, good_tracks.current, verified_tracks.previous,
        verified_tracks.current);

    if (verified_tracks.current.size() > 0) {
      // TODO: do ANMS supression here!
      nextBatched[object_id] = verified_tracks.current;
    }
  }

  return nextBatched;
}

cv::Mat drawBatchedFeatures(
    const cv::Mat& image,
    const gtsam::FastMap<ObjectId, std::vector<cv::Point2f>>& batchedFeatures) {
  cv::Mat canvas;

  // Ensure we are drawing on a 3-channel color image
  if (image.channels() == 1) {
    cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);
  } else {
    canvas = image.clone();
  }

  // Initialize a random number generator for distinct colors
  cv::RNG rng(12345);

  for (const auto& [j, features] : batchedFeatures) {
    cv::Scalar color = dyno::Color::uniqueObjectId(j).bgra();
    for (const auto& point : features) {
      // Draw a solid circle at the feature location
      cv::circle(canvas, point, 4, color, -1, cv::LINE_AA);

      // Optional: Draw a small outer ring to make it pop
      cv::circle(canvas, point, 6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
  }

  return canvas;
}

struct TrackingDetails {
  ObjectId object_id{};
  FrameId last_detection{};
  size_t num_last_detected_features{};
};

int main(int argc, char* argv[]) {
  using namespace dyno;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;
  FLAGS_v = 30;

  // KittiDataLoader::Params params;
  // KittiDataLoader loader("/root/data/vdo_slam/kitti/kitti/0004/", params);
  // ClusterSlamDataLoader loader("/root/data/cluster_slam/CARLA-S2");
  // loader.setStartingFrame(600);
  OMDDataLoader loader(
      "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/");

  // TartanAirShibuyaLoader
  // loader("/root/data/TartanAir_shibuya/RoadCrossing07/");
  // ViodeLoader loader("/root/data/VIODE/city_day/mid");

  // auto detector = dyno::PyObjectDetectorWrapper::CreateYoloDetector();
  // CHECK_NOTNULL(detector);

  // DynopetsLoader loader("/root/data/dynopets_mocap/VAL10Seqs/4_laptop");

  FrontendParams fp;
  // fp.tracker_params.feature_detector_type =
  // TrackerParams::FeatureDetectorType::GFFT_CUDA;
  fp.tracker_params.max_nr_keypoints_before_anms = 1000;
  fp.tracker_params.max_dynamic_features_per_frame = 300;
  fp.tracker_params.prefer_provided_optical_flow = false;
  fp.tracker_params.prefer_provided_object_detection = true;
  fp.tracker_params.max_dynamic_feature_age = 1000;
  // fp.tracker_params.feature_detector_type =
  // TrackerParams::FeatureDetectorType::ORB_SLAM_ORB;
  fp.tracker_params.min_distance_btw_tracked_and_detected_static_features = 15;

  auto camera = std::make_shared<Camera>(loader.getCameraParams());
  auto tracker = std::make_shared<FeatureTracker>(fp, camera);

  // std::vector<std::vector<cv::Point2f>> previousBatchFeatures;
  gtsam::FastMap<ObjectId, std::vector<cv::Point2f>> previousBatchFeatures;
  gtsam::FastMap<ObjectId, TrackingDetails> tracking_details;
  ObjectIds prevObjectIds;

  loader.registerImageContainerCallback([&](ImageContainer::Ptr container)
                                            -> void {
    LOG(INFO) << container->frameId() << " " << container->timestamp();

    // cv::Mat rgb_viz, motion_viz, depth_viz;
    // // motion_viz =
    // //     ImageType::MotionMask::toRGB(container->objectMotionMask());
    // depth_viz = ImageType::Depth::toRGB(container->depth());
    // // rgb_viz = container->rgb();

    // // // cv::imshow("RGB", rgb_viz);
    // cv::imshow("Depth", depth_viz);
    // // // cv::imshow("Motion", motion_viz);
    // // // cv::waitKey(1);

    // // ImageContainer image_container(frame_id, timestamp);
    // // image_container.rgb(rgb)
    // //     .depth(depth)
    // //     .opticalFlow(optical_flow)
    // //     .objectMotionMask(motion);

    auto frame_id = container->frameId();
    auto timestamp = container->timestamp();

    auto frame = tracker->track(frame_id, timestamp, *container);

    // LOG(INFO) << "Batched extraction: " << time_ms << " [ms]";

    // cv::waitKey(0);
    Frame::Ptr previous_frame = tracker->getPreviousFrame();
    utils::ChronoTimingStats batch_all_t("batched.all");
    const cv::Mat object_masks = container->objectMotionMask();
    const cv::Mat current_mono = ImageType::RGBMono::toMono(container->rgb());

    utils::ChronoTimingStats labels_t("batched.get_labels");
    ObjectIds object_ids = vision_tools::getObjectLabels(object_masks);
    labels_t.stop();

    if (!previous_frame) {
      std::vector<cv::Mat> masks;
      std::vector<int> maxCorners;
      std::vector<float> minDistances;

      for (auto object_id : object_ids) {
        cv::Mat obj_mask = (object_masks == object_id);

        masks.push_back(obj_mask);
        maxCorners.push_back(300);
        minDistances.push_back(4);
      }

      cv::Mat object_masks_binary = object_masks > 0;
      cv::Mat static_mask;
      cv::bitwise_not(object_masks_binary, static_mask);

      masks.push_back(static_mask);
      maxCorners.push_back(1000);
      object_ids.push_back(0);
      minDistances.push_back(15);

      utils::ChronoTimingStats detection_t("batched.detection");
      auto detected_features = goodFeaturesToTrackBatched(
          current_mono, masks, maxCorners, minDistances, 0.01);
      auto time_ms = detection_t.stop();

      gtsam::FastMap<ObjectId, std::vector<cv::Point2f>> detected_feature_map;
      for (size_t i = 0; i < detected_features.size(); i++) {
        if (detected_features[i].size() > 0) {
          detected_feature_map[object_ids[i]] = detected_features[i];

          TrackingDetails details;
          details.object_id = object_ids[i];
          details.num_last_detected_features = detected_features[i].size();

          tracking_details[object_ids[i]] = details;
        }
      }
      previousBatchFeatures = detected_feature_map;

      cv::Mat viz = drawBatchedFeatures(container->rgb(), detected_feature_map);
      cv::imshow("batch Tracks", viz);
    }

    // LOG(INFO) << to_string(tracker->getTrackerInfo());

    if (previous_frame) {
      // ImageTracksParams track_viz_params(true);
      // track_viz_params.show_intermediate_tracking = true;
      // cv::Mat tracking = tracker->computeFeatureTracks(
      //     *previous_frame, *frame, track_viz_params);

      // cv::imshow("Tracks", tracking);

      auto previous_mono =
          ImageType::RGBMono::toMono(previous_frame->imageContainer().rgb());

      utils::ChronoTimingStats detection_t("batched.track");
      auto tracked_features = trackFeaturesUnified(
          previous_mono, current_mono, previousBatchFeatures, object_masks);
      detection_t.stop();

      // masks will be updated to include currently tracked points
      gtsam::FastMap<ObjectId, cv::Mat> mask_map;
      for (auto object_id : object_ids) {
        cv::Mat obj_mask = (object_masks == object_id);
        mask_map[object_id] = obj_mask;

        // masks.push_back(obj_mask);
        // maxCorners.push_back(300);
        // minDistances.push_back(4);
      }

      cv::Mat object_masks_binary = object_masks > 0;
      cv::Mat static_mask;
      cv::bitwise_not(object_masks_binary, static_mask);
      mask_map[0] = static_mask;

      std::set<ObjectId> object_needs_detection;
      for (auto& [j, per_object_tracks] : tracked_features) {
        LOG(INFO) << "Tracked features for j=" << j;
        int distance = j > 0 ? 8 : 15;
        for (const auto& kp : per_object_tracks) {
          // mark as location to ignore when doing feature detection
          cv::circle(mask_map[j], kp, distance, cv::Scalar(0), cv::FILLED);
        }

        auto num_tracked = per_object_tracks.size();
        auto num_previous = tracking_details[j].num_last_detected_features;
        // not since previous track but since the last detecion!
        const double survival_ratio =
            num_previous > 0 ? (double)num_tracked / (double)num_previous : 0.0;
        const bool poor_tracking = survival_ratio < 0.4;
        const bool too_few_tracks = static_cast<int>(num_tracked) < 30;
        const bool needs_detection = poor_tracking || too_few_tracks;

        if (needs_detection) {
          // hack for now
          LOG(INFO) << "Replacing tracks for poorly tracked object " << j;
          // per_object_tracks = detected_feature_map[j];
          object_needs_detection.insert(j);

        } else {
          LOG(INFO) << "Good tracks for object " << j;
          // say previous object is well tracked
          // previous_objects.insert(j);
        }
      }

      std::vector<cv::Mat> masks;
      std::vector<int> maxCorners;
      std::vector<float> minDistances;
      ObjectIds objects_ids_for_detection;

      for (const auto& [j, masks_j] : mask_map) {
        // if does not exist in current tracking and we have detections
        // add as new object
        bool is_new = !tracked_features.exists(j);
        if (object_needs_detection.count(j) > 0 || is_new) {
          LOG(INFO) << "Object j " << j << " needs detection";
          masks.push_back(masks_j);
          objects_ids_for_detection.push_back(j);

          // TODO: recompute maxCorners
          if (j > 0) {
            maxCorners.push_back(300);
            minDistances.push_back(8);
          } else {
            maxCorners.push_back(1000);
            minDistances.push_back(15);
          }
        }
      }

      auto detected_features = goodFeaturesToTrackBatched(
          current_mono, masks, maxCorners, minDistances, 0.01);

      gtsam::FastMap<ObjectId, std::vector<cv::Point2f>> detected_feature_map;
      for (size_t i = 0; i < detected_features.size(); i++) {
        if (detected_features[i].size() > 0) {
          detected_feature_map[objects_ids_for_detection[i]] =
              detected_features[i];

          TrackingDetails details;
          details.object_id = objects_ids_for_detection[i];
          details.num_last_detected_features = detected_features[i].size();
          tracking_details[details.object_id] = details;
        }
      }

      // for now just replace featues
      for (const auto& [j, per_object_tracks] : detected_feature_map) {
        // tracked_features[j] = per_object_tracks;
        // add tracks to existing tracks!
        tracked_features[j].insert(tracked_features[j].begin(),
                                   per_object_tracks.begin(),
                                   per_object_tracks.end());
      }

      cv::Mat tracked_viz =
          drawBatchedFeatures(container->rgb(), tracked_features);

      cv::imshow("batch Tracks", tracked_viz);
      previousBatchFeatures = tracked_features;
    }
    batch_all_t.stop();
    // else {
    //   previousBatchFeatures = detected_feature_map;

    // }

    LOG(INFO) << utils::Statistics::Print();

    cv::waitKey(0);
  });

  // loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
  //                        cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth,
  //                        cv::Mat motion, gtsam::Pose3,
  //                        GroundTruthInputPacket) -> bool {
  // LOG(INFO) << utils::Statistics::Print();
  // loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
  //                        cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth,
  //                        cv::Mat motion, GroundTruthInputPacket,
  //                        std::optional<ImuMeasurements> imu_measurements,
  //                        std::optional<cv::Mat>) -> bool {
  // loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
  //                        cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth,
  //                        cv::Mat motion, GroundTruthInputPacket,
  //                        std::optional<cv::Mat>) -> bool {
  // LOG(INFO) << frame_id << " " << timestamp;

  // cv::Mat of_viz, motion_viz, depth_viz;
  // of_viz = ImageType::OpticalFlow::toRGB(optical_flow);
  // motion_viz = ImageType::MotionMask::toRGB(motion);
  // depth_viz = ImageType::Depth::toRGB(depth);

  // ImageContainerDeprecate::Ptr container = ImageContainerDeprecate::Create(
  //     timestamp, frame_id, ImageWrapper<ImageType::RGBMono>(rgb),
  //     ImageWrapper<ImageType::Depth>(depth),
  //     ImageWrapper<ImageType::OpticalFlow>(optical_flow),
  //     ImageWrapper<ImageType::MotionMask>(motion));

  // cv::Mat boarder_mask;
  // vision_tools::computeObjectMaskBoundaryMask(
  //     motion,
  //     boarder_mask,
  //     8
  // );

  // cv::Scalar red = dyno::Color::red();

  // const ObjectIds instance_labels = vision_tools::getObjectLabels(motion);
  // for(const auto object_id : instance_labels) {
  //     std::vector<std::vector<cv::Point>> detected_contours;
  //     vision_tools::findObjectBoundingBox(motion,
  //     object_id,detected_contours);

  //     cv::drawContours(boarder_mask, detected_contours, -1, red, 8);
  // }

  // cv::imshow("Mask with boarder", boarder_mask);

  // cv::imshow("RGB", rgb);
  // cv::imshow("OF", of_viz);
  // cv::imshow("Motion", motion_viz);
  // // cv::waitKey(1);
  // cv::imshow("Depth", depth_viz);

  // auto object_detection_result = detector->process(rgb);
  // cv::imshow("Detection Result", object_detection_result.colouredMask());

  // ImageContainer image_container(frame_id, timestamp);
  // image_container.rgb(rgb)
  //     .depth(depth)
  //     .opticalFlow(optical_flow)
  //     .objectMotionMask(motion);
  // // image_container.rgb(rgb).depth(depth).opticalFlow(optical_flow);
  // auto frame = tracker->track(frame_id, timestamp, image_container);
  // Frame::Ptr previous_frame = tracker->getPreviousFrame();

  // if(frame_id == 605) {
  //   auto all_tracks = frame->static_features_.collectTracklets();
  //   frame->static_features_.markOutliers(all_tracks);
  // }

  // // motion_viz =
  // ImageType::MotionMask::toRGB(frame->image_container_.get<ImageType::MotionMask>());
  // // // cv::imshow("Motion", motion_viz);

  // cv::Mat tracking;
  // if (previous_frame) {
  //   ImageTracksParams track_viz_params(true);
  //   track_viz_params.show_intermediate_tracking = true;
  //   tracking = tracker->computeFeatureTracks(*previous_frame, *frame,
  //                                          track_viz_params);

  // if (imu_measurements) {
  //   const auto previous_timestamp = previous_frame->getTimestamp();

  //   CHECK_GE(imu_measurements->timestamps_[0], previous_timestamp);
  //   CHECK_LT(imu_measurements
  //                ->timestamps_[imu_measurements->timestamps_.cols() - 1],
  //            timestamp);

  //   LOG(INFO) << "Gotten imu messages!";

  //   CHECK(imu_measurements->synchronised_frame_id);
  //   CHECK_EQ(imu_measurements->synchronised_frame_id.value(),
  //            frame->getFrameId());
  // }
  // }
  // if (!tracking.empty()) cv::imshow("Tracking", tracking);

  // LOG(INFO) << to_string(tracker->getTrackerInfo());
  // const std::string path = "/root/results/misc/";
  // if (previous_frame && (char)cv::waitKey(0) == 's') {
  //   LOG(INFO) << "Saving...";
  //   // cv::imwrite(path + "omd_su4_rgb.png", rgb);
  //   // cv::imwrite(path + "omd_su4_of.png", of_viz);
  //   // cv::imwrite(path + "omd_su4_motion.png", motion_viz);
  //   // cv::imwrite(path + "omd_su4_depth.png", depth_viz);
  //   // cv::imwrite(
  //   //     path + "cluster_tracking_new" + std::to_string(frame_id) +
  //   ".png",
  //   //     tracking);
  // }
  //   cv::waitKey(1);

  //   return true;
  // });

  while (loader.spin()) {
  }
}

// #include "dynosam/dataprovider/ProjectAriaDataProvider.hpp"
// #include "dynosam/frontend/vision/VisionTools.hpp"

// int main(int argc, char* argv[]) {

//     using namespace dyno;
//     google::ParseCommandLineFlags(&argc, &argv, true);
//     google::InitGoogleLogging(argv[0]);
//     FLAGS_logtostderr = 1;
//     FLAGS_colorlogtostderr = 1;
//     FLAGS_log_prefix = 1;

//     // ClusterSlamDataLoader loader("/root/data/cluster_slam/CARLA-S1");
//     ProjectARIADataLoader loader("/root/data/zed/acfr_3_moving_medium/");

//     loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
//     cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth, cv::Mat motion) -> bool
//     {

//         LOG(INFO) << frame_id << " " << timestamp;

//         cv::imshow("RGB", rgb);
//         cv::imshow("OF", ImageType::OpticalFlow::toRGB(optical_flow));
//         cv::imshow("Motion", ImageType::MotionMask::toRGB(motion));
//         cv::imshow("Depth", ImageType::Depth::toRGB(depth));

//         cv::Mat shrunk_mask;
//         vision_tools::shrinkMask(motion, shrunk_mask, 20);
//         cv::imshow("Shrunk Motion",
//         ImageType::MotionMask::toRGB(shrunk_mask));

//         cv::waitKey(1);
//         return true;
//     });

//     while(loader.spin()) {}

// }
