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
#include <opencv2/cudaimgproc.hpp>
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
#include "dynosam/frontend/vision/FeatureTrackerFast.hpp"
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

// /** Corner responses for a single object (including the background) */
// struct Corners {
//   ObjectId object_id{-1};
//   std::vector<cv::Point2f> keypoints{};
//   std::vector<float> responses{};
//   std::vector<bool> valid{};
// };

// /** Tracked corner responses for a single object */
// struct CornerTracks : public Corners {
//   TrackletIds tracklet_ids{};
// };

struct CornerResponses {
  std::vector<cv::Point2f> keypoints;
  std::vector<float> responses;

  inline size_t size() const {
    CHECK_EQ(keypoints.size(), responses.size());
    return keypoints.size();
  }

  inline void reserve(size_t n) {
    keypoints.reserve(n);
    responses.reserve(n);
  }

  inline void push_back(const CornerCandidate& candidate) {
    keypoints.push_back(candidate.pt);
    responses.push_back(candidate.score);
  }

  std::vector<KeypointCV> toKeypoints() const {
    std::vector<KeypointCV> keypoints_cv(size());
    for (size_t i = 0; i < size(); i++) {
      keypoints_cv.emplace_back(keypoints[i], 0.0, 0.0, responses[i]);
    }
    return keypoints_cv;
  }
};

struct CornerTracks {
  std::vector<cv::Point2f> keypoints{};
  TrackletIds tracklet_ids{};

  CornerTracks& operator+=(const CornerTracks& other) {
    keypoints.insert(keypoints.begin(), other.keypoints.begin(),
                     other.keypoints.end());
    tracklet_ids.insert(tracklet_ids.begin(), other.tracklet_ids.begin(),
                        other.tracklet_ids.end());
    return *this;
  }

  inline size_t size() const {
    CHECK_EQ(keypoints.size(), tracklet_ids.size());
    return keypoints.size();
  }
};

struct VerificationInfo {
  //! Num tracks after outlier rejection (ie. verification)
  size_t num_tracks{0};
  //! Number of tracks in the previous frame
  size_t num_previous_tracks{0};

  float survivalRatio() const {
    if (num_previous_tracks > 0 && num_tracks > 0) {
      return (float)num_tracks / num_previous_tracks;
    } else {
      return 0.0;
    }
  }
};

struct DetailedCornerTracks {
  CornerTracks corner_tracks;
  VerificationInfo info;
};

class FeatureTrackerBatch {
 private:
  TrackletIdManager& tracklet_id_manager;
  cv::Mat prev_mono_;
  //! Image pyramid for the previous mono frame
  std::vector<cv::Mat> prev_mono_pyr_;
  // //! Binary image used feature detection mask for the previous frame
  // cv::Mat prev_detection_mask_;
  //! Object mask for the previous frame
  cv::Mat prev_object_mask_;

  //! Page locked host allocations which own the image memory. Owns memory
  cv::cuda::HostMem h_mono_;
  cv::cuda::HostMem h_eig_;
  //! cv::Mat headers pointing to the page-locked allocations above. Does not
  //! own memory
  cv::Mat mono_;
  cv::Mat eig_;

  cv::cuda::GpuMat d_mono_;
  cv::cuda::GpuMat d_eig_;

  cv::cuda::Stream stream_;
  cv::Ptr<cv::cuda::CornernessCriteria> detector_;

  struct TrackingInfo {
    ObjectId object_id{};
    FrameId last_detection{};
    size_t num_last_detected_features{0};
    size_t num_last_tracked{0};
  };
  //! Global tracking info (includes all objects)
  gtsam::FastMap<ObjectId, TrackingInfo> tracking_infos_;
  gtsam::FastMap<ObjectId, CornerTracks> previous_tracks_;

  FeatureBlockContainer previous_features_;

  struct SingleDetectionParam {
    ObjectId object_id;
    //! Binary object/detection mask
    cv::Mat mask;
    cv::Rect bbox;
    // num total corners to extract
    int max_corners;
    // int corners_after_anms;
    float min_distance;
  };

  // optical flow params
  const cv::Size win_size_;
  const int max_level_;
  const cv::TermCriteria criteria_;

 public:
  FeatureTrackerBatch()
      : tracklet_id_manager(TrackletIdManager::instance()),
        win_size_(21, 21),
        max_level_(3),
        criteria_(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.001) {}
  ~FeatureTrackerBatch() {}

  void initDeviceMemory(const cv::Size& size) {
    // / ------------------------------------------------------------------
    // Page-locked host memory
    //
    // HostMem owns the allocations. The cv::Mat objects below are only
    // headers pointing into these allocations.
    // ------------------------------------------------------------------

    h_mono_ = cv::cuda::HostMem(size, CV_8UC1, cv::cuda::HostMem::PAGE_LOCKED);

    h_eig_ = cv::cuda::HostMem(size, CV_32FC1, cv::cuda::HostMem::PAGE_LOCKED);

    // Create cv::Mat headers over the pinned memory.
    mono_ = h_mono_.createMatHeader();
    eig_ = h_eig_.createMatHeader();

    // ------------------------------------------------------------------
    // GPU memory
    // ------------------------------------------------------------------

    d_mono_.create(size, CV_8UC1);
    d_eig_.create(size, CV_32FC1);

    // ------------------------------------------------------------------
    // CUDA corner detector
    //
    // Equivalent to:
    //
    // cv::cornerMinEigenVal(mono, eig, blockSize, 3);
    // ------------------------------------------------------------------

    detector_ = cv::cuda::createMinEigenValCorner(CV_8UC1, 3, 3);
  }

  // eventually params and binary detection mask
  // gtsam::FastMap<ObjectId, DetailedCornerTracks> trackGfftBatched(
  //     const cv::Mat& mono, const cv::Mat& object_mask) {
  //   CHECK(!prev_mono_.empty());
  //   CV_Assert(prev_mono_.type() == CV_8UC1 && mono.type() == CV_8UC1);

  //   // 1. --- FLATTENING PHASE ---
  //   // Count total points across all masks to make a single allocation
  //   size_t totalPoints = 0;
  //   for (const auto& [_, vec] : previous_tracks_) {
  //     totalPoints += vec.size();
  //   }

  //   struct Tracklet2DVectors {
  //     std::vector<cv::Point2f> current;
  //     std::vector<cv::Point2f> previous;
  //     TrackletIds tracklet_ids{};
  //   };

  //   gtsam::FastMap<ObjectId, Tracklet2DVectors> tracks_per_object;
  //   gtsam::FastMap<ObjectId, DetailedCornerTracks> nextBatched;

  //   if (totalPoints == 0) return nextBatched;  // Nothing to track

  //   std::vector<cv::Point2f> flatPrev;
  //   flatPrev.reserve(totalPoints);

  //   // Track which object ID (mask index) each flattened point belongs to
  //   dyno::ObjectIds pointOwnerId;
  //   pointOwnerId.reserve(totalPoints);
  //   //! Track which tracklet id each flattened point belongs too
  //   dyno::ObjectIds pointTrackletId;
  //   pointTrackletId.reserve(totalPoints);

  //   gtsam::FastMap<ObjectId, size_t> num_previous_points;
  //   for (const auto& [j, corner_tracks] : previous_tracks_) {
  //     num_previous_points[j] = 0;
  //     for (size_t i = 0; i < corner_tracks.size(); i++) {
  //       // if inlier!
  //       flatPrev.push_back(corner_tracks.keypoints[i]);
  //       pointOwnerId.push_back(j);
  //       pointTrackletId.push_back(corner_tracks.tracklet_ids[i]);
  //       num_previous_points[j]++;
  //     }

  //     // Estimate allocation space (assuming most features survive)
  //     Tracklet2DVectors vecs;
  //     vecs.current.reserve(corner_tracks.size());
  //     vecs.previous.reserve(corner_tracks.size());
  //     vecs.tracklet_ids.reserve(corner_tracks.size());
  //     tracks_per_object[j] = vecs;
  //   }

  //   // 2. --- SINGLE-PASS KLT EXECUTION ---
  //   std::vector<cv::Point2f> flatNext = flatPrev;
  //   std::vector<uchar> forward_status;
  //   std::vector<float> forward_err;

  //   std::vector<cv::Mat> current_mono_pyr;
  //   buildOpticalFlowPyramid(mono, current_mono_pyr);

  //   // One single call allows OpenCV to run hot loops across contiguous
  //   memory
  //   // blocks
  //   cv::calcOpticalFlowPyrLK(prev_mono_pyr_, current_mono_pyr, flatPrev,
  //                            flatNext, forward_status, forward_err,
  //                            win_size_, max_level_, criteria_, 0);

  //   // now do reverse flow
  //   std::vector<uchar> reverse_status;
  //   std::vector<float> reverse_err;

  //   std::vector<cv::Point2f> flatReverse = flatNext;
  //   cv::calcOpticalFlowPyrLK(current_mono_pyr, prev_mono_pyr_, flatNext,
  //                            flatReverse, reverse_status, reverse_err,
  //                            win_size_, max_level_, criteria_,
  //                            cv::OPTFLOW_USE_INITIAL_FLOW);

  //   // 3. --- UNFLATTENING PHASE ---
  //   static constexpr float kMaxErr = 20.0f;
  //   for (size_t i = 0; i < flatPrev.size(); ++i) {
  //     const bool both_status_good =
  //         forward_status.at(i) && reverse_status.at(i);
  //     const bool within_distance =
  //         utils::distance(flatPrev.at(i), flatReverse.at(i)) <= 0.5;
  //     const bool within_error =
  //         reverse_err[i] < kMaxErr && forward_err[i] < kMaxErr;

  //     // Check if KLT tracking succeeded and point remains inside image
  //     // boundaries use 2i to check image boundaries
  //     if (both_status_good && within_distance && within_error) {
  //       forward_status.at(i) = 1;
  //     } else {
  //       forward_status.at(i) = 0;
  //     }

  //     cv::Point2i flextNextInt = static_cast<cv::Point2i>(flatNext[i]);
  //     if (forward_status[i] && flextNextInt.x >= 0 &&
  //         flextNextInt.x < mono.cols && flextNextInt.y >= 0 &&
  //         flextNextInt.y < mono.rows) {
  //       auto j = pointOwnerId[i];
  //       auto tracklet_id = pointTrackletId[i];

  //       // LOG(INFO) << "obj j " << j << " with curr object mask " <<
  //       // currObjectMask.at<dyno::ObjectId>(flatNext[i]);

  //       if (j != object_mask.at<dyno::ObjectId>(flextNextInt)) {
  //         continue;
  //       }

  //       tracks_per_object[j].current.push_back(flatNext[i]);
  //       tracks_per_object[j].previous.push_back(flatPrev[i]);
  //       tracks_per_object[j].tracklet_ids.push_back(tracklet_id);
  //     }
  //   }

  //   // homograph adds about 1ms!
  //   for (const auto& [object_id, good_tracks] : tracks_per_object) {
  //     Tracklet2DVectors verified_tracks;
  //     verified_tracks.current.reserve(good_tracks.current.size());
  //     verified_tracks.previous.reserve(good_tracks.previous.size());
  //     verified_tracks.tracklet_ids.reserve(good_tracks.previous.size());

  //     vision_tools::outlierRejectHomography(
  //         good_tracks.previous, good_tracks.current,
  //         good_tracks.tracklet_ids, verified_tracks.previous,
  //         verified_tracks.current, verified_tracks.tracklet_ids);

  //     if (verified_tracks.current.size() > 0) {
  //       DetailedCornerTracks detailed_tracks;
  //       detailed_tracks.corner_tracks.keypoints =
  //           std::move(verified_tracks.current);
  //       detailed_tracks.corner_tracks.tracklet_ids =
  //           std::move(verified_tracks.tracklet_ids);

  //       detailed_tracks.info.num_previous_tracks =
  //           num_previous_points[object_id];
  //       detailed_tracks.info.num_tracks =
  //       detailed_tracks.corner_tracks.size();

  //       nextBatched[object_id] = detailed_tracks;
  //     }
  //   }

  //   // update previous image pyramid for reuse
  //   prev_mono_pyr_ = current_mono_pyr;

  //   return nextBatched;
  // }

  FeatureBlockContainer trackGfftBatched(const cv::Mat& mono,
                                         const cv::Mat& object_mask) {
    CHECK(!prev_mono_.empty());
    CV_Assert(prev_mono_.type() == CV_8UC1 && mono.type() == CV_8UC1);

    gtsam::FastMap<ObjectId, FeatureBlockContainer::FeatureData>
        tracks_per_object;
    gtsam::FastMap<ObjectId, size_t> num_previous_points;
    for (const auto& object_view : previous_features_.objectViews()) {
      size_t num_points = object_view.size();
      auto object_id = object_view.objectId();
      num_previous_points[object_id] = num_points;

      FeatureBlockContainer::FeatureData data;
      data.points.reserve(num_points);
      data.previous_points.reserve(num_points);
      data.ids.reserve(num_points);

      data.errors.reserve(num_points);
      data.status.reserve(num_points);

      tracks_per_object[object_id] = data;

      LOG(INFO) << "Preparing featue tracking structures j=" << object_id
                << " n=" << num_points;
    }
    // for (const auto& [j, corner_tracks] : previous_tracks_) {
    //   num_previous_points[j] = 0;
    //   for (size_t i = 0; i < corner_tracks.size(); i++) {
    //     // if inlier!
    //     flatPrev.push_back(corner_tracks.keypoints[i]);
    //     pointOwnerId.push_back(j);
    //     pointTrackletId.push_back(corner_tracks.tracklet_ids[i]);
    //     num_previous_points[j]++;
    //   }

    //   // Estimate allocation space (assuming most features survive)
    //   Tracklet2DVectors vecs;
    //   vecs.current.reserve(corner_tracks.size());
    //   vecs.previous.reserve(corner_tracks.size());
    //   vecs.tracklet_ids.reserve(corner_tracks.size());
    //   tracks_per_object[j] = vecs;
    // }

    // 2. --- SINGLE-PASS KLT EXECUTION ---
    std::vector<cv::Point2f> flatNext = previous_features_.points;
    const auto& flatPrev = previous_features_.points;
    std::vector<uchar> forward_status;
    std::vector<float> forward_err;

    std::vector<cv::Mat> current_mono_pyr;
    buildOpticalFlowPyramid(mono, current_mono_pyr);

    // One single call allows OpenCV to run hot loops across contiguous memory
    // blocks
    cv::calcOpticalFlowPyrLK(prev_mono_pyr_, current_mono_pyr, flatPrev,
                             flatNext, forward_status, forward_err, win_size_,
                             max_level_, criteria_, 0);

    // now do reverse flow
    std::vector<uchar> reverse_status;
    std::vector<float> reverse_err;

    std::vector<cv::Point2f> flatReverse = flatNext;
    cv::calcOpticalFlowPyrLK(current_mono_pyr, prev_mono_pyr_, flatNext,
                             flatReverse, reverse_status, reverse_err,
                             win_size_, max_level_, criteria_,
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    // 3. --- UNFLATTENING PHASE ---
    static constexpr float kMaxErr = 20.0f;
    for (size_t i = 0; i < flatPrev.size(); ++i) {
      const bool both_status_good =
          forward_status.at(i) && reverse_status.at(i);
      const bool within_distance =
          utils::distance(flatPrev.at(i), flatReverse.at(i)) <= 0.5;
      const bool within_error =
          reverse_err[i] < kMaxErr && forward_err[i] < kMaxErr;

      // Check if KLT tracking succeeded and point remains inside image
      // boundaries use 2i to check image boundaries
      if (both_status_good && within_distance && within_error) {
        forward_status.at(i) = 1;
      } else {
        forward_status.at(i) = 0;
      }

      cv::Point2i flextNextInt = static_cast<cv::Point2i>(flatNext[i]);
      if (forward_status[i] && flextNextInt.x >= 0 &&
          flextNextInt.x < mono.cols && flextNextInt.y >= 0 &&
          flextNextInt.y < mono.rows) {
        auto object_id = previous_features_.object_ids[i];
        auto tracklet_id = previous_features_.ids[i];

        // LOG(INFO) << "obj j " << j << " with curr object mask " <<
        // currObjectMask.at<dyno::ObjectId>(flatNext[i]);

        if (object_id != object_mask.at<dyno::ObjectId>(flextNextInt)) {
          continue;
        }

        tracks_per_object[object_id].points.push_back(flatNext[i]);
        tracks_per_object[object_id].previous_points.push_back(flatPrev[i]);
        tracks_per_object[object_id].ids.push_back(tracklet_id);

        tracks_per_object[object_id].status.push_back(1);
        tracks_per_object[object_id].errors.push_back(forward_err[i]);
      }
    }

    gtsam::FastMap<ObjectId, FeatureBlockContainer::FeatureData>
        verified_tracks_per_object;

    for (const auto& [object_id, good_tracks] : tracks_per_object) {
      // cv::Mat inlier_mask;
      // outlierRejectHomography(
      //   good_tracks.previousPointsMat(),
      //   good_tracks.pointsMat(),
      //   inlier_mask);
      cv::Mat inlier_mask = vision_tools::findHomography(
          good_tracks.previous_points, good_tracks.points);

      auto num_good_points = good_tracks.size();

      // replace FeatureTracks
      FeatureBlockContainer::FeatureData verified_tracks;
      verified_tracks.points.reserve(num_good_points);
      verified_tracks.previous_points.reserve(num_good_points);
      verified_tracks.ids.reserve(num_good_points);
      verified_tracks.errors.reserve(num_good_points);
      verified_tracks.status.reserve(num_good_points);

      for (int i = 0; i < inlier_mask.rows; ++i) {
        if (inlier_mask.at<uchar>(i)) {
          verified_tracks.points.push_back(good_tracks.points[i]);
          verified_tracks.previous_points.push_back(
              good_tracks.previous_points[i]);
          verified_tracks.ids.push_back(good_tracks.ids[i]);
          verified_tracks.errors.push_back(good_tracks.errors[i]);
          verified_tracks.status.push_back(good_tracks.status[i]);
        }
      }

      // if we actually have any tracks
      // this will remove any objects with no tracks!
      if (verified_tracks.size() > 0) {
        LOG(INFO) << "j= " << object_id << "inlier/outlier "
                  << verified_tracks.size() << "/" << num_good_points;
        verified_tracks_per_object[object_id] = verified_tracks;
      }
    }

    // // homograph adds about 1ms!
    // for (const auto& [object_id, good_tracks] : tracks_per_object) {
    //   Tracklet2DVectors verified_tracks;
    //   verified_tracks.current.reserve(good_tracks.current.size());
    //   verified_tracks.previous.reserve(good_tracks.previous.size());
    //   verified_tracks.tracklet_ids.reserve(good_tracks.previous.size());

    //   vision_tools::outlierRejectHomography(
    //       good_tracks.previous, good_tracks.current,
    //       good_tracks.tracklet_ids, verified_tracks.previous,
    //       verified_tracks.current, verified_tracks.tracklet_ids);

    //   if (verified_tracks.current.size() > 0) {
    //     DetailedCornerTracks detailed_tracks;
    //     detailed_tracks.corner_tracks.keypoints =
    //         std::move(verified_tracks.current);
    //     detailed_tracks.corner_tracks.tracklet_ids =
    //         std::move(verified_tracks.tracklet_ids);

    //     detailed_tracks.info.num_previous_tracks =
    //         num_previous_points[object_id];
    //     detailed_tracks.info.num_tracks =
    //     detailed_tracks.corner_tracks.size();

    //     nextBatched[object_id] = detailed_tracks;
    //   }
    // }

    // update previous image pyramid for reuse
    prev_mono_pyr_ = current_mono_pyr;

    FeatureBlockContainer tracked_features(verified_tracks_per_object);
    tracked_features.printDebugInfo();
    return tracked_features;
  }

  void buildOpticalFlowPyramid(const cv::Mat& mono,
                               std::vector<cv::Mat>& pyramid) const {
    pyramid.resize(max_level_ + 1);
    cv::buildOpticalFlowPyramid(mono, pyramid, win_size_, max_level_, false,
                                cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT,
                                true  // critical for reuse
    );
  }

  FeatureBlockContainer detectGfftBatched(
      const cv::Mat& mono,
      const std::vector<SingleDetectionParam>& detection_params,
      float qualityLevel = 0.01, int blockSize = 3) {
    using namespace dyno;
    utils::ChronoTimingStats t("gfft_batch");
    CV_Assert(mono.type() == CV_8UC1 || mono.type() == CV_32FC1);
    // CV_Assert(masks.size() == maxCorners.size());

    if (detection_params.empty()) {
      return {};
    }

    // --- STEP 1: Compute the Eigenvalue Map ONCE for the whole frame ---
    // cv::Mat eig;
    // cornerMinEigenVal handles the Sobel derivatives internally in a highly
    // optimized pass
    // utils::ChronoTimingStats t1("corner_min_eigen");
    // cv::cornerMinEigenVal(mono, eig, blockSize, 3);
    // t1.stop();
    // Copy normal CPU memory -> pinned memory.
    //
    utils::ChronoTimingStats t1("corner_min_eigen");
    // If your camera/image pipeline can write directly into mono_,
    // this copy can be eliminated entirely.
    mono.copyTo(mono_);

    // Pinned host memory -> GPU.
    d_mono_.upload(mono_, stream_);

    // GPU min-eigenvalue corner response.
    detector_->compute(d_mono_, d_eig_, stream_);

    // GPU -> pinned host memory.
    d_eig_.download(eig_, stream_);

    // Because compute() returns a CPU cv::Mat, we need to wait before
    // returning it.
    stream_.waitForCompletion();
    t1.stop();

    // Find the global maximum corner score across the entire image
    utils::ChronoTimingStats tmin("minMaxLoc");
    double maxVal = 0;
    cv::minMaxLoc(eig_, nullptr, &maxVal);
    tmin.stop();

    // Establish the baseline absolute threshold based on global max quality
    const float threshold = static_cast<float>(maxVal * qualityLevel);

    // --- STEP 2: Local Non-Maximum Suppression (NMS) via Dilation ---
    // OpenCV's internal GFTT uses a dilation trick to find local maxima
    // efficiently
    utils::ChronoTimingStats t2("dilate");
    cv::Mat localMax;
    cv::dilate(eig_, localMax, cv::Mat());
    t2.stop();

    // calculate distance transform for each mask
    utils::ChronoTimingStats distance_t("distance masks");
    // this can take up to 3-4ms
    std::vector<cv::Mat> distanceTransformMasks(detection_params.size());
    for (size_t i = 0; i < detection_params.size(); i++) {
      cv::distanceTransform(detection_params[i].mask, distanceTransformMasks[i],
                            cv::DIST_L2, 3);
    }
    distance_t.stop();

    std::vector<CornerResponses> batchedResults(detection_params.size());

    utils::ChronoTimingStats t3("masks_loop");
    cv::parallel_for_(
        cv::Range(0, static_cast<int>(detection_params.size())),
        [&](const cv::Range& range) {
          for (int m = range.start; m < range.end; ++m) {
            const auto& params = detection_params[m];

            const cv::Mat& mask = params.mask;
            const cv::Mat& dist = distanceTransformMasks[m];

            const int maxFeatureCount = params.max_corners;
            const float minDistance = params.min_distance;

            // terms[m].first = detection_params[m].object_id;

            if (maxFeatureCount <= 0) continue;

            // --------------------------------------------------------------
            // Restrict the entire detection process to the object bounding
            // box rather than scanning the entire image.
            // --------------------------------------------------------------

            const cv::Rect& bbox = params.bbox;

            if (bbox.empty()) continue;

            const int x0 = bbox.x;
            const int y0 = bbox.y;
            const int x1 = bbox.x + bbox.width;
            const int y1 = bbox.y + bbox.height;

            const float minDistanceSq = minDistance * minDistance;

            // --------------------------------------------------------------
            // Grid
            // --------------------------------------------------------------

            const int cellSize = std::max(1, cvRound(minDistance));
            const float invCellSize = 1.0f / static_cast<float>(cellSize);

            const int gridWidth = (bbox.width + cellSize - 1) / cellSize;

            const int gridHeight = (bbox.height + cellSize - 1) / cellSize;

            // --------------------------------------------------------------
            // Candidate storage
            // --------------------------------------------------------------

            std::vector<CornerCandidate> candidates;
            candidates.reserve(256);

            // --------------------------------------------------------------
            // Find local maxima
            //
            // IMPORTANT:
            // Scan only the bounding box.
            // --------------------------------------------------------------

            for (int y = y0; y < y1; ++y) {
              const float* eigPtr = eig_.ptr<float>(y);
              const float* maxPtr = localMax.ptr<float>(y);

              const uchar* maskPtr =
                  mask.empty() ? nullptr : mask.ptr<uchar>(y);

              const float* distPtr =
                  dist.empty() ? nullptr : dist.ptr<float>(y);

              for (int x = x0; x < x1; ++x) {
                // Check mask FIRST.
                //
                // For small dynamic object masks this avoids reading
                // the distance transform for almost every background
                // pixel in the bounding box.
                if (maskPtr && !maskPtr[x]) continue;

                const float val = eigPtr[x];

                if (val <= threshold) continue;

                if (val != maxPtr[x]) continue;

                // Mask-edge constraint
                if (distPtr && distPtr[x] <= 5.0f) continue;

                candidates.push_back(
                    {cv::Point2f(static_cast<float>(x), static_cast<float>(y)),
                     val});
              }
            }

            if (candidates.empty()) continue;

            // --------------------------------------------------------------
            // Sort strongest corners first.
            // --------------------------------------------------------------

            std::sort(candidates.begin(), candidates.end(),
                      std::greater<CornerCandidate>());

            // --------------------------------------------------------------
            // Output
            // --------------------------------------------------------------

            CornerResponses& acceptedCorners = batchedResults[m];
            // std::vector<cv::Point2f>& accepted_corners =
            // terms[m].second.points;

            // acceptedCorners.clear();
            acceptedCorners.reserve(maxFeatureCount);

            // One linked-list head per spatial cell.
            std::vector<int> gridHeads(gridWidth * gridHeight, -1);

            std::vector<int> nextPointIdx;
            nextPointIdx.reserve(maxFeatureCount);

            // --------------------------------------------------------------
            // Greedy GFTT distance suppression
            // --------------------------------------------------------------

            for (const CornerCandidate& candidate : candidates) {
              if (static_cast<int>(acceptedCorners.size()) >= maxFeatureCount) {
                break;
              }

              // Coordinates relative to bounding box.
              const int localX = static_cast<int>(candidate.pt.x) - x0;

              const int localY = static_cast<int>(candidate.pt.y) - y0;

              const int xCell = static_cast<int>(localX * invCellSize);

              const int yCell = static_cast<int>(localY * invCellSize);

              const int x1Cell = std::max(0, xCell - 1);
              const int y1Cell = std::max(0, yCell - 1);
              const int x2Cell = std::min(gridWidth - 1, xCell + 1);
              const int y2Cell = std::min(gridHeight - 1, yCell + 1);

              bool good = true;

              for (int yy = y1Cell; yy <= y2Cell && good; ++yy) {
                const int rowOffset = yy * gridWidth;

                for (int xx = x1Cell; xx <= x2Cell; ++xx) {
                  int pIdx = gridHeads[rowOffset + xx];

                  while (pIdx != -1) {
                    const cv::Point2f& accepted =
                        acceptedCorners.keypoints[pIdx];

                    const float dx = candidate.pt.x - accepted.x;

                    const float dy = candidate.pt.y - accepted.y;

                    if (dx * dx + dy * dy < minDistanceSq) {
                      good = false;
                      break;
                    }

                    pIdx = nextPointIdx[pIdx];
                  }

                  if (!good) break;
                }
              }

              if (!good) continue;

              const int cellIdx = yCell * gridWidth + xCell;

              const int pointIdx = static_cast<int>(acceptedCorners.size());

              nextPointIdx.push_back(gridHeads[cellIdx]);

              gridHeads[cellIdx] = pointIdx;

              acceptedCorners.push_back(candidate);
            }
          }
        });

    t3.stop();

    utils::ChronoTimingStats t_terms("make_blocks");
    std::vector<std::pair<ObjectId, FeatureBlockContainer::FeatureData>> terms(
        detection_params.size());
    for (size_t i = 0; i < detection_params.size(); i++) {
      terms[i].first = detection_params[i].object_id;

      const CornerResponses& corner_responses = batchedResults[i];
      terms[i].second.points = corner_responses.keypoints;
      terms[i].second.errors.resize(corner_responses.size());
      terms[i].second.previous_points.resize(corner_responses.size());
      terms[i].second.ids.resize(corner_responses.size());
      terms[i].second.status.resize(corner_responses.size());
      terms[i].second.errors.resize(corner_responses.size());
    }

    FeatureBlockContainer feature_blocks(terms);
    t_terms.stop();

    LOG(INFO) << "Detection:";
    feature_blocks.printDebugInfo();

    utils::ChronoTimingStats t4("sub_pixe_refine");

    if (feature_blocks.size() == 0) {
      return feature_blocks;
    }

    const cv::Size window_size = cv::Size(5, 5);
    const cv::Size zero_zone = cv::Size(-1, -1);
    const cv::TermCriteria criteria = cv::TermCriteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001);

    // // flattern batched results (bit gross but should be fast)
    // std::vector<cv::Point2f> points;
    // std::vector<size_t> sizes;

    // for (const auto& object : batchedResults) {
    //   sizes.push_back(object.size());
    //   points.insert(points.end(), object.keypoints.begin(),
    //                 object.keypoints.end());
    // }

    // if (points.empty()) {
    //   return batchedResults;
    // }

    cv::cornerSubPix(mono, feature_blocks.points, window_size, zero_zone,
                     criteria);

    // // Split back into per-object vectors.
    // size_t offset = 0;

    // for (size_t i = 0; i < batchedResults.size(); ++i) {
    //   std::copy(points.begin() + offset, points.begin() + offset + sizes[i],
    //             batchedResults[i].keypoints.begin());

    //   offset += sizes[i];
    // }

    return feature_blocks;
  }

  // void track(const cv::Mat& rgb, const cv::Mat& object_masks) {
  //   utils::ChronoTimingStats track_t("batched.track");

  //   utils::ChronoTimingStats labels_t("batched.get_labels");
  //   ObjectIds object_ids = vision_tools::getObjectLabels(object_masks);
  //   labels_t.stop();

  //   cv::Mat current_mono = ImageType::RGBMono::toMono(rgb);

  //   utils::ChronoTimingStats bb_t("batched.get_bb");
  //   std::vector<cv::Rect> bounding_boxes;
  //   vision_tools::getObjectBoundingBoxes(object_masks, object_ids,
  //                                        bounding_boxes);
  //   bb_t.stop();

  //   LOG(INFO) << "With object ids= " << container_to_string(object_ids);

  //   std::map<ObjectId, size_t> object_index;
  //   for (size_t i = 0; i < object_ids.size(); i++) {
  //     object_index[object_ids[i]] = i;
  //   }

  //   if (prev_mono_.empty()) {
  //     initDeviceMemory(current_mono.size());

  //     // TODO: fill small holes before creating detection mask?
  //     // construct detection params per objects
  //     std::vector<SingleDetectionParam> detection_params(object_ids.size() +
  //     1); for (size_t i = 0u; i < object_ids.size(); i++) {
  //       const auto object_id = object_ids[i];

  //       cv::Mat obj_mask = (object_masks == object_id);
  //       // TODO: combine with detection mask to not detect on existing
  //       tracks!

  //       detection_params[i].object_id = object_id;
  //       detection_params[i].mask = obj_mask;
  //       detection_params[i].bbox = bounding_boxes[object_index[object_id]];
  //       detection_params[i].max_corners = 600;
  //       detection_params[i].min_distance = 8;
  //     }
  //     cv::Mat object_masks_binary = object_masks > 0;
  //     cv::Mat static_mask;
  //     cv::bitwise_not(object_masks_binary, static_mask);

  //     cv::Rect image_rect;
  //     image_rect.x = 0;
  //     image_rect.y = 0;
  //     image_rect.width = current_mono.cols;
  //     image_rect.height = current_mono.rows;

  //     detection_params[object_ids.size()].object_id = 0;
  //     detection_params[object_ids.size()].mask = static_mask;
  //     detection_params[object_ids.size()].bbox = image_rect;
  //     detection_params[object_ids.size()].max_corners = 1000;
  //     detection_params[object_ids.size()].min_distance = 15;
  //     object_ids.push_back(0);

  //     utils::ChronoTimingStats detect_t("batched.detect");
  //     auto detected_features =
  //         detectGfftBatched(current_mono, detection_params, 0.01);
  //     detect_t.stop();

  //     gtsam::FastMap<ObjectId, CornerTracks> detected_feature_map;
  //     for (size_t i = 0; i < detected_features.size(); i++) {
  //       const ObjectId object_id = object_ids[i];
  //       if (detected_features[i].size() > 0) {
  //         CornerTracks tracks;
  //         tracks.keypoints = detected_features[i].keypoints;
  //         // fill new trackletids
  //         tracks.tracklet_ids.reserve(tracks.keypoints.size());
  //         for (const auto& corner : tracks.keypoints) {
  //           TrackletId tracklet_to_use =
  //               tracklet_id_manager.getAndIncrementTrackletId();
  //           tracks.tracklet_ids.push_back(tracklet_to_use);
  //         }

  //         detected_feature_map[object_id] = tracks;

  //         TrackingInfo details;
  //         details.object_id = object_id;
  //         details.num_last_detected_features = tracks.keypoints.size();

  //         tracking_infos_[object_id] = details;
  //       }
  //     }
  //     previous_tracks_ = std::move(detected_feature_map);
  //     // fill detection mask
  //     prev_mono_ = current_mono;
  //     prev_object_mask_ = object_masks;

  //     buildOpticalFlowPyramid(prev_mono_, prev_mono_pyr_);

  //     cv::Mat viz = drawBatchedFeatures(rgb, previous_tracks_);
  //     cv::imshow("batch Tracks", viz);

  //   } else {
  //     utils::ChronoTimingStats feature_track_t("batched.track_gfft");

  //     auto detailed_tracks = trackGfftBatched(current_mono, object_masks);
  //     feature_track_t.stop();

  //     // fill tracks with tracking only as we will eventially discovard the
  //     // verification info?
  //     gtsam::FastMap<ObjectId, CornerTracks> tracks;

  //     // waste of copying?
  //     for (const auto& [object_id, detailed_tracks_j] : detailed_tracks) {
  //       tracks[object_id] = detailed_tracks_j.corner_tracks;
  //     }

  //     // masks will be updated to include currently tracked points
  //     gtsam::FastMap<ObjectId, cv::Mat> mask_map;
  //     for (auto object_id : object_ids) {
  //       cv::Mat obj_mask = (object_masks == object_id);
  //       mask_map[object_id] = obj_mask;
  //     }

  //     const auto outer_thickness = 10;
  //     cv::Mat object_masks_binary = object_masks > 0;
  //     // to ensure the validatiy of the mask we will do some dilation
  //     // to both fill holes and to expant the object masks so we dont
  //     // attempt to track anywhere near the objects
  //     // this is slightly conservative but is helpful in practice
  //     cv::dilate(object_masks_binary, object_masks_binary,
  //                cv::getStructuringElement(cv::MORPH_RECT,
  //                                          cv::Size(2 * outer_thickness + 1,
  //                                                   2 * outer_thickness +
  //                                                   1)));

  //     cv::Mat static_mask;
  //     cv::bitwise_not(object_masks_binary, static_mask);
  //     mask_map[0] = static_mask;

  //     std::set<ObjectId> object_needs_detection;
  //     for (auto& [j, detailed_tracks_j] : detailed_tracks) {
  //       const auto& corner_tracks_j = detailed_tracks_j.corner_tracks;
  //       const auto& verification_info_j = detailed_tracks_j.info;

  //       LOG(INFO) << "Tracked features for j=" << j;
  //       int distance = j > 0 ? 8 : 15;
  //       for (const auto& kp : corner_tracks_j.keypoints) {
  //         // mark as location to ignore when doing feature detection
  //         cv::circle(mask_map[j], kp, distance, cv::Scalar(0), cv::FILLED);
  //       }

  //       auto num_tracked = verification_info_j.num_tracks;
  //       const float survival_ratio = verification_info_j.survivalRatio();
  //       const bool poor_tracking = survival_ratio < 0.4;

  //       const auto min_allowed_tracks = j > 0 ? 20 : 200;
  //       const bool too_few_tracks =
  //           static_cast<int>(num_tracked) < min_allowed_tracks;

  //       bool needs_detection = poor_tracking || too_few_tracks;

  //       LOG(INFO) << "Tracked j=" << j << " n= " << num_tracked
  //                 << " poor_tracking= " <<
  //                 tracking_infos_[j].num_last_tracked
  //                 << " SR=" << survival_ratio;

  //       // // update number tracked (this may then need to change after
  //       // retroactive tracking!) tracking_infos_[j].num_last_tracked =
  //       // num_tracked;

  //       const bool is_object = j > 0;
  //       if (is_object) {
  //         const cv::Rect& detected_bounding_box =
  //             bounding_boxes[object_index[j]];
  //         const cv::Rect tracked_bounding_box =
  //             cv::boundingRect(corner_tracks_j.keypoints);

  //         const double iou =
  //             utils::calculateIoU(detected_bounding_box,
  //             tracked_bounding_box);
  //         const auto min_iou = 0.5;

  //         const bool small_iou = iou < min_iou;

  //         needs_detection = needs_detection || small_iou;

  //         // if detection rectangle is tiny (less than 100 pixels in area)
  //         just
  //         // ignore
  //         if (detected_bounding_box.area() < 80) {
  //           needs_detection = false;
  //         }
  //       }

  //       if (needs_detection) {
  //         // hack for now
  //         LOG(INFO) << "Replacing tracks for poorly tracked object " << j;
  //         // per_object_tracks = detected_feature_map[j];
  //         object_needs_detection.insert(j);

  //       } else {
  //         LOG(INFO) << "Good tracks for object " << j;
  //       }
  //     }

  //     std::vector<SingleDetectionParam> detection_params;
  //     ObjectIds objects_ids_for_detection;

  //     for (const auto& [j, masks_j] : mask_map) {
  //       LOG(INFO) << "Looking at mask j=" << j;
  //       // if does not exist in current tracking and we have detections
  //       // add as new object
  //       bool is_new = !detailed_tracks.exists(j);
  //       if (object_needs_detection.count(j) > 0 || is_new) {
  //         LOG(INFO) << "Object j " << j << " needs detection";
  //         // masks.push_back(masks_j);
  //         objects_ids_for_detection.push_back(j);

  //         // const int current_tracks = is_new ? 0 : tracks.at(j).size();

  //         // at the detection stage detect more than is necessary
  //         // but not this many! Detect only a bit above what is needed!
  //         const int max_corners = j > 0 ? 600 : 1000;
  //         // const int tracked_needed =
  //         //     std::max(desired_tracks - current_tracks, 0);
  //         const int distance = j > 0 ? 8 : 15;

  //         cv::Rect bbox;
  //         if (j > 0) {
  //           bbox = bounding_boxes[object_index[j]];
  //         } else {
  //           bbox.x = 0;
  //           bbox.y = 0;
  //           bbox.width = current_mono.cols;
  //           bbox.height = current_mono.rows;
  //         }

  //         SingleDetectionParam detection_param;
  //         // mask includes marked location of existing feature points
  //         detection_param.mask = masks_j;
  //         detection_param.bbox = bbox;
  //         detection_param.object_id = j;
  //         detection_param.max_corners = max_corners;
  //         detection_param.min_distance = distance;
  //         detection_params.push_back(detection_param);
  //       }
  //     }

  //     if (!objects_ids_for_detection.empty()) {
  //       utils::ChronoTimingStats detect_t("batched.detect");
  //       auto detected_features =
  //           detectGfftBatched(current_mono, detection_params, 0.01);
  //       detect_t.stop();

  //       // TODO: ANMS to cut down to only the set of features we want!

  //       AdaptiveNonMaximumSuppression non_maximum_supression(
  //           AnmsAlgorithmType::RangeTree);

  //       static constexpr float kTolerance = 0.01;
  //       static Eigen::MatrixXd binning_mask;

  //       gtsam::FastMap<ObjectId, CornerTracks> detected_feature_map;
  //       for (size_t i = 0; i < detected_features.size(); i++) {
  //         if (detected_features[i].size() > 0) {
  //           ObjectId object_id = objects_ids_for_detection[i];

  //           const int desired_tracks = object_id > 0 ? 300 : 800;
  //           int current_tracks =
  //               detailed_tracks.exists(object_id)
  //                   ? detailed_tracks.at(object_id).corner_tracks.size()
  //                   : 0;
  //           const int tracked_needed =
  //               std::max(desired_tracks - current_tracks, 0);

  //           std::vector<cv::Point2f>& new_keypoints =
  //               detected_features[i].keypoints;
  //           if (tracked_needed > 0) {
  //             std::vector<KeypointCV> keypoints =
  //                 detected_features[i].toKeypoints();
  //             std::vector<KeypointCV>& max_keypoints = keypoints;

  //             max_keypoints = non_maximum_supression.suppressNonMax(
  //                 keypoints, tracked_needed, kTolerance, current_mono.cols,
  //                 current_mono.rows, 5, 5, binning_mask);

  //             LOG(INFO) << "J=" << object_id << " tracks = " <<
  //             current_tracks
  //                       << " needed=" << tracked_needed << " after anms "
  //                       << max_keypoints.size();

  //             // std::vector<cv::Point2f> points_after_anms;
  //             cv::KeyPoint::convert(max_keypoints, new_keypoints);
  //           }

  //           CornerTracks new_tracks;
  //           new_tracks.keypoints = new_keypoints;
  //           // fill new trackletids
  //           new_tracks.tracklet_ids.reserve(new_tracks.keypoints.size());
  //           for (auto i = 0u; i < new_tracks.keypoints.size(); i++) {
  //             TrackletId tracklet_to_use =
  //                 tracklet_id_manager.getAndIncrementTrackletId();
  //             new_tracks.tracklet_ids.push_back(tracklet_to_use);
  //           }

  //           detected_feature_map[object_id] = new_tracks;

  //           TrackingInfo details;
  //           details.object_id = object_id;
  //           details.num_last_detected_features = new_tracks.keypoints.size();

  //           // TODO: missing last number of tracking
  //           tracking_infos_[object_id] = details;
  //         }
  //       }

  //       // for now just replace featues
  //       for (const auto& [j, per_object_tracks] : detected_feature_map) {
  //         // tracked_features[j] = per_object_tracks;
  //         // add tracks to existing tracks!
  //         tracks[j] += per_object_tracks;
  //       }
  //     }

  //     std::stringstream ss;
  //     ss << "Objects in tracks j=";
  //     for (const auto& [j, _] : tracks) {
  //       ss << j << " ";
  //     }

  //     LOG(INFO) << ss.str();

  //     previous_tracks_ = std::move(tracks);
  //     // fill detection mask
  //     prev_mono_ = current_mono;
  //     prev_object_mask_ = object_masks;

  //     cv::Mat viz = drawBatchedFeatures(rgb, previous_tracks_);
  //     cv::imshow("batch Tracks", viz);

  //     // cv::waitKey(0);
  //   }
  // }

  void track(const cv::Mat& rgb, const cv::Mat& object_masks) {
    utils::ChronoTimingStats track_t("batched.track");

    utils::ChronoTimingStats labels_t("batched.get_labels");
    ObjectIds object_ids = vision_tools::getObjectLabels(object_masks);
    labels_t.stop();

    cv::Mat current_mono = ImageType::RGBMono::toMono(rgb);

    utils::ChronoTimingStats bb_t("batched.get_bb");
    std::vector<cv::Rect> bounding_boxes;
    vision_tools::getObjectBoundingBoxes(object_masks, object_ids,
                                         bounding_boxes);
    bb_t.stop();

    LOG(INFO) << "With object ids= " << container_to_string(object_ids);

    std::map<ObjectId, size_t> object_index;
    for (size_t i = 0; i < object_ids.size(); i++) {
      object_index[object_ids[i]] = i;
    }

    if (prev_mono_.empty()) {
      initDeviceMemory(current_mono.size());

      std::vector<SingleDetectionParam> detection_params(object_ids.size() + 1);
      for (size_t i = 0u; i < object_ids.size(); i++) {
        const auto object_id = object_ids[i];

        cv::Mat obj_mask = (object_masks == object_id);
        // TODO: combine with detection mask to not detect on existing tracks!

        detection_params[i].object_id = object_id;
        detection_params[i].mask = obj_mask;
        detection_params[i].bbox = bounding_boxes[object_index[object_id]];
        detection_params[i].max_corners = 600;
        detection_params[i].min_distance = 8;
      }
      cv::Mat object_masks_binary = object_masks > 0;
      cv::Mat static_mask;
      cv::bitwise_not(object_masks_binary, static_mask);

      cv::Rect image_rect;
      image_rect.x = 0;
      image_rect.y = 0;
      image_rect.width = current_mono.cols;
      image_rect.height = current_mono.rows;

      detection_params[object_ids.size()].object_id = 0;
      detection_params[object_ids.size()].mask = static_mask;
      detection_params[object_ids.size()].bbox = image_rect;
      detection_params[object_ids.size()].max_corners = 1000;
      detection_params[object_ids.size()].min_distance = 15;
      object_ids.push_back(0);

      utils::ChronoTimingStats detect_t("batched.detect");
      FeatureBlockContainer detected_features =
          detectGfftBatched(current_mono, detection_params, 0.01);
      detect_t.stop();

      prev_mono_ = current_mono;
      previous_features_ = detected_features;

      buildOpticalFlowPyramid(prev_mono_, prev_mono_pyr_);

      cv::Mat viz = drawBatchedFeatures(rgb, detected_features);
      cv::imshow("batch Tracks", viz);

    } else {
      utils::ChronoTimingStats feature_track_t("batched.track_gfft");

      FeatureBlockContainer tracked_features =
          trackGfftBatched(current_mono, object_masks);
      feature_track_t.stop();

      // masks will be updated to include currently tracked points
      gtsam::FastMap<ObjectId, cv::Mat> mask_map;
      for (auto object_id : object_ids) {
        cv::Mat obj_mask = (object_masks == object_id);
        mask_map[object_id] = obj_mask;
      }

      const auto outer_thickness = 10;
      cv::Mat object_masks_binary = object_masks > 0;
      // to ensure the validatiy of the mask we will do some dilation
      // to both fill holes and to expant the object masks so we dont
      // attempt to track anywhere near the objects
      // this is slightly conservative but is helpful in practice
      cv::dilate(object_masks_binary, object_masks_binary,
                 cv::getStructuringElement(cv::MORPH_RECT,
                                           cv::Size(2 * outer_thickness + 1,
                                                    2 * outer_thickness + 1)));

      cv::Mat static_mask;
      cv::bitwise_not(object_masks_binary, static_mask);
      mask_map[0] = static_mask;

      std::set<ObjectId> object_needs_detection;

      for (const auto& object_view : tracked_features.objectViews()) {
        const auto j = object_view.objectId();

        // should only be inliers!
        const auto num_tracked = object_view.size();
        LOG(INFO) << "Tracked features for j=" << j << " n=" << num_tracked;
        int distance = j > 0 ? 8 : 15;
        for (size_t i = 0; i < num_tracked; i++) {
          // mark as location to ignore when doing feature detection
          cv::circle(mask_map[j], object_view.points()[i], distance,
                     cv::Scalar(0), cv::FILLED);
        }

        const auto min_allowed_tracks = j > 0 ? 20 : 200;
        const bool too_few_tracks =
            static_cast<int>(num_tracked) < min_allowed_tracks;

        // bool needs_detection = poor_tracking || too_few_tracks;
        bool needs_detection = too_few_tracks;

        const bool is_object = j > 0;
        if (is_object) {
          const cv::Rect& detected_bounding_box =
              bounding_boxes[object_index[j]];
          const cv::Rect tracked_bounding_box =
              cv::boundingRect(object_view.pointsMat());

          const double iou =
              utils::calculateIoU(detected_bounding_box, tracked_bounding_box);
          const auto min_iou = 0.5;

          const bool small_iou = iou < min_iou;

          needs_detection = needs_detection || small_iou;

          // if detection rectangle is tiny (less than 100 pixels in area) just
          // ignore
          if (detected_bounding_box.area() < 80) {
            needs_detection = false;
          }
        }

        if (needs_detection) {
          // hack for now
          LOG(INFO) << "Replacing tracks for poorly tracked object " << j;
          // per_object_tracks = detected_feature_map[j];
          object_needs_detection.insert(j);

        } else {
          LOG(INFO) << "Good tracks for object " << j;
        }
      }

      std::vector<SingleDetectionParam> detection_params;
      ObjectIds objects_ids_for_detection;

      for (const auto& [j, masks_j] : mask_map) {
        LOG(INFO) << "Looking at mask j=" << j;
        // if does not exist in current tracking and we have detections
        // add as new object
        bool is_new = !tracked_features.containsObject(j);
        if (object_needs_detection.count(j) > 0 || is_new) {
          LOG(INFO) << "Object j " << j << " needs detection";
          // masks.push_back(masks_j);
          objects_ids_for_detection.push_back(j);

          // const int current_tracks = is_new ? 0 : tracks.at(j).size();

          // at the detection stage detect more than is necessary
          // but not this many! Detect only a bit above what is needed!
          const int max_corners = j > 0 ? 600 : 1000;
          // const int tracked_needed =
          //     std::max(desired_tracks - current_tracks, 0);
          const int distance = j > 0 ? 8 : 15;

          cv::Rect bbox;
          if (j > 0) {
            bbox = bounding_boxes[object_index[j]];
          } else {
            bbox.x = 0;
            bbox.y = 0;
            bbox.width = current_mono.cols;
            bbox.height = current_mono.rows;
          }

          SingleDetectionParam detection_param;
          // mask includes marked location of existing feature points
          detection_param.mask = masks_j;
          detection_param.bbox = bbox;
          detection_param.object_id = j;
          detection_param.max_corners = max_corners;
          detection_param.min_distance = distance;
          detection_params.push_back(detection_param);
        }
      }

      if (!objects_ids_for_detection.empty()) {
        utils::ChronoTimingStats detect_t("batched.detect");
        // TODO: still need to handle tracklet ids!
        FeatureBlockContainer detected_features =
            detectGfftBatched(current_mono, detection_params, 0.01);
        detect_t.stop();

        // TODO: ANMS to cut down to only the set of features we want!

        AdaptiveNonMaximumSuppression non_maximum_supression(
            AnmsAlgorithmType::RangeTree);

        static constexpr float kTolerance = 0.01;
        static Eigen::MatrixXd binning_mask;

        // gtsam::FastMap<ObjectId, CornerTracks> detected_feature_map;
        // for (size_t i = 0; i < detected_features.size(); i++) {
        //   if (detected_features[i].size() > 0) {
        //     ObjectId object_id = objects_ids_for_detection[i];

        //     const int desired_tracks = object_id > 0 ? 300 : 800;
        //     int current_tracks =
        //         detailed_tracks.exists(object_id)
        //             ? detailed_tracks.at(object_id).corner_tracks.size()
        //             : 0;
        //     const int tracked_needed =
        //         std::max(desired_tracks - current_tracks, 0);

        //     std::vector<cv::Point2f>& new_keypoints =
        //         detected_features[i].keypoints;
        //     if (tracked_needed > 0) {
        //       std::vector<KeypointCV> keypoints =
        //           detected_features[i].toKeypoints();
        //       std::vector<KeypointCV>& max_keypoints = keypoints;

        //       max_keypoints = non_maximum_supression.suppressNonMax(
        //           keypoints, tracked_needed, kTolerance, current_mono.cols,
        //           current_mono.rows, 5, 5, binning_mask);

        //       LOG(INFO) << "J=" << object_id << " tracks = " <<
        //       current_tracks
        //                 << " needed=" << tracked_needed << " after anms "
        //                 << max_keypoints.size();

        //       // std::vector<cv::Point2f> points_after_anms;
        //       cv::KeyPoint::convert(max_keypoints, new_keypoints);
        //     }

        //     CornerTracks new_tracks;
        //     new_tracks.keypoints = new_keypoints;
        //     // fill new trackletids
        //     new_tracks.tracklet_ids.reserve(new_tracks.keypoints.size());
        //     for (auto i = 0u; i < new_tracks.keypoints.size(); i++) {
        //       TrackletId tracklet_to_use =
        //           tracklet_id_manager.getAndIncrementTrackletId();
        //       new_tracks.tracklet_ids.push_back(tracklet_to_use);
        //     }

        //     detected_feature_map[object_id] = new_tracks;

        //     TrackingInfo details;
        //     details.object_id = object_id;
        //     details.num_last_detected_features = new_tracks.keypoints.size();

        //     // TODO: missing last number of tracking
        //     tracking_infos_[object_id] = details;
        //   }
        // }

        // // for now just replace featues
        // for (const auto& [j, per_object_tracks] : detected_feature_map) {
        //   // tracked_features[j] = per_object_tracks;
        //   // add tracks to existing tracks!
        //   tracks[j] += per_object_tracks;
        // }
        previous_features_ = tracked_features.merge(detected_features);

      } else {
        previous_features_ = tracked_features;
      }

      cv::Mat viz = drawBatchedFeatures(rgb, previous_features_);
      cv::imshow("batch Tracks", viz);

      prev_mono_ = current_mono;
    }
  }

  // cv::Mat drawBatchedFeatures(
  //     const cv::Mat& image,
  //     const gtsam::FastMap<ObjectId, CornerTracks>& batchedFeatures) {
  //   cv::Mat canvas;

  //   // Ensure we are drawing on a 3-channel color image
  //   if (image.channels() == 1) {
  //     cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);
  //   } else {
  //     canvas = image.clone();
  //   }

  //   // Initialize a random number generator for distinct colors
  //   cv::RNG rng(12345);

  //   for (const auto& [j, corner_tracks] : batchedFeatures) {
  //     cv::Scalar color = dyno::Color::uniqueObjectId(j).bgra();
  //     for (const auto& point : corner_tracks.keypoints) {
  //       // Draw a solid circle at the feature location
  //       cv::circle(canvas, point, 4, color, -1, cv::LINE_AA);

  //       // Optional: Draw a small outer ring to make it pop
  //       cv::circle(canvas, point, 6, cv::Scalar(255, 255, 255), 1,
  //       cv::LINE_AA);
  //     }
  //   }

  //   return canvas;
  // }

  cv::Mat drawBatchedFeatures(const cv::Mat& image,
                              const FeatureBlockContainer& batchedFeatures) {
    cv::Mat canvas;

    // Ensure we are drawing on a 3-channel color image
    if (image.channels() == 1) {
      cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);
    } else {
      canvas = image.clone();
    }

    // Initialize a random number generator for distinct colors
    cv::RNG rng(12345);

    for (size_t i = 0; i < batchedFeatures.size(); i++) {
      auto j = batchedFeatures.object_ids[i];
      auto point = batchedFeatures.points[i];
      cv::Scalar color = dyno::Color::uniqueObjectId(j).bgra();
      // Draw a solid circle at the feature location
      cv::circle(canvas, point, 4, color, -1, cv::LINE_AA);

      // Optional: Draw a small outer ring to make it pop
      cv::circle(canvas, point, 6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    return canvas;
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

  if (masks.empty()) {
    return {};
  }

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

  // calculate distance transform for each mask
  std::vector<cv::Mat> distanceTransformMasks(masks.size());
  for (size_t i = 0; i < masks.size(); i++) {
    cv::distanceTransform(masks[i], distanceTransformMasks[i], cv::DIST_L2, 3);
  }

  std::vector<std::vector<cv::Point2f>> batchedResults(masks.size());

  utils::ChronoTimingStats t3("masks_loop");
  // --- STEP 3: Process Each Mask Independently to Fulfill Specific Quotas ---
  cv::parallel_for_(
      cv::Range(0, static_cast<int>(masks.size())),
      [&](const cv::Range& range) {
        for (int m = range.start; m < range.end; ++m) {
          const cv::Mat& mask = masks[m];
          const cv::Mat& dist = distanceTransformMasks[m];
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

              // distance from edge of mask. Only include features that are
              // within some threshold from the mask edge (this is akin to
              // shrinking the object mask)
              const float d = dist.at<float>(y, x);

              const bool within_mask_edge = d > 10.0;

              if (val > threshold && val == max_ptr[x] &&
                  (!mask_ptr || mask_ptr[x]) && within_mask_edge) {
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
            const int x_cell = static_cast<int>(candidate.pt.x * inv_cell_size);
            const int y_cell = static_cast<int>(candidate.pt.y * inv_cell_size);

            // Enforce 3x3 search bounds around the center cell
            const int x1 = std::max(0, x_cell - 1);
            const int y1 = std::max(0, y_cell - 1);
            const int x2 = std::min(grid_width - 1, x_cell + 1);
            const int y2 = std::min(grid_height - 1, y_cell + 1);

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
          //             keep = fssalse;
          //             break;
          //         }
          //     }

          //     if (keep) {
          //         acceptedCorners.push_back(candidate.pt);
          //     }
          // }
        }
      });

  t3.stop();

  utils::ChronoTimingStats t4("sub_pixe_refine");
  const cv::Size window_size = cv::Size(5, 5);
  const cv::Size zero_zone = cv::Size(-1, -1);
  const cv::TermCriteria criteria = cv::TermCriteria(
      cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01);

  // flattern batched results (bit gross but should be fast)
  std::vector<cv::Point2f> points;
  std::vector<size_t> sizes;

  for (const auto& object : batchedResults) {
    sizes.push_back(object.size());
    points.insert(points.end(), object.begin(), object.end());
  }

  cv::cornerSubPix(src, points, window_size, zero_zone, criteria);

  // Split back into per-object vectors.
  size_t offset = 0;

  for (size_t i = 0; i < batchedResults.size(); ++i) {
    std::copy(points.begin() + offset, points.begin() + offset + sizes[i],
              batchedResults[i].begin());

    offset += sizes[i];
  }

  return batchedResults;
}

/**
 * @brief High-performance KLT tracker that flattens batched features into a
 * single OpenCV call to maximize SIMD efficiency, then unflattens the surviving
 * tracks.
 * * @param prev_mono_      Grayscale source frame (CV_8UC1)
 * @param mono      Grayscale target frame (CV_8UC1)
 * @param previous_tracks_  The tracked features from the previous frame,
 * grouped by mask/object ID
 * @return std::vector<std::vector<cv::Point2f>> Cleanly unflattened features
 * tracking into mono
 */
gtsam::FastMap<ObjectId, std::vector<cv::Point2f>> trackFeaturesUnified(
    const cv::Mat& prev_mono_, const cv::Mat& mono,
    const gtsam::FastMap<ObjectId, std::vector<cv::Point2f>>& previous_tracks_,
    const cv::Mat& currObjectMask) {
  CV_Assert(prev_mono_.type() == CV_8UC1 && mono.type() == CV_8UC1);

  // 1. --- FLATTENING PHASE ---
  // Count total points across all masks to make a single allocation
  size_t totalPoints = 0;
  for (const auto& [_, vec] : previous_tracks_) {
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

  for (const auto& [j, vec] : previous_tracks_) {
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
  // for (size_t m = 0; m < previous_tracks_.size(); ++m) {
  //   LOG(INFO) << "Prev object id " << prevObjectIds[m];
  //     for (const auto& pt : previous_tracks_[m]) {
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
  cv::calcOpticalFlowPyrLK(prev_mono_, mono, flatPrev, flatNext, status, err,
                           winSize, maxLevel, criteria);

  // 3. --- UNFLATTENING PHASE ---
  for (size_t i = 0; i < flatPrev.size(); ++i) {
    // Check if KLT tracking succeeded and point remains inside image boundaries
    // use 2i to check image boundaries
    cv::Point2i flextNextInt = static_cast<cv::Point2i>(flatNext[i]);
    if (status[i] && err[i] < 10.0f && flextNextInt.x >= 0 &&
        flextNextInt.x < mono.cols && flextNextInt.y >= 0 &&
        flextNextInt.y < mono.rows) {
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

/**
 * TODO:
 * retroactive tracking
 * reimplement re-detection features (ie survivial rate)
 * properly fill out FeatureData and ensure tracklet ids are correctly
 * generated/propogated check which properties we actually want in the
 * FeatureData move tests to use the new FeatureBLockContainer so any changes
 * are reflected in the tests test additional outlier rejection with opengv
 * 2DPnP solve (although this might just  be essential matrix calc!) do
 * technical writeup of changes! Implement full tracker in FeatureTrackerFast
 * class!
 */

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
  // OMDDataLoader loader(
  // "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/");

  // TartanAirShibuyaLoader
  // loader("/root/data/TartanAir_shibuya/RoadCrossing07/");
  ViodeLoader loader("/root/data/VIODE/city_day/mid");

  // auto detector = dyno::PyObjectDetectorWrapper::CreateYoloDetector();
  // CHECK_NOTNULL(detector);

  // DynopetsLoader loader("/root/data/dynopets_mocap/VAL10Seqs/4_laptop");

  FrontendParams fp;
  // fp.tracker_params.feature_detector_type =
  //   TrackerParams::FeatureDetectorType::GFFT_CUDA;
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

  FeatureTrackerBatch ftb;

  loader.registerImageContainerCallback(
      [&](ImageContainer::Ptr container) -> void {
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

        // auto frame = tracker->track(frame_id, timestamp, *container);

        ftb.track(container->rgb(), container->objectMotionMask());

        // // LOG(INFO) << "Batched extraction: " << time_ms << " [ms]";

        // // // cv::waitKey(0);
        // Frame::Ptr previous_frame = tracker->getPreviousFrame();
        // utils::ChronoTimingStats batch_all_t("batched.all");
        // const cv::Mat object_masks = container->objectMotionMask();
        // const cv::Mat current_mono =
        // ImageType::RGBMono::toMono(container->rgb());

        // utils::ChronoTimingStats labels_t("batched.get_labels");
        // ObjectIds object_ids = vision_tools::getObjectLabels(object_masks);
        // labels_t.stop();

        // utils::ChronoTimingStats bb_t("batched.get_bb");
        // std::vector<cv::Rect> bounding_boxes;
        // vision_tools::getObjectBoundingBoxes(object_masks, object_ids,
        // bounding_boxes); bb_t.stop();

        // std::map<ObjectId, size_t> object_index;
        // for(size_t i = 0; i < object_ids.size(); i++) {
        //   object_index[object_ids[i]] = i;
        // }

        // if (!previous_frame) {
        //   std::vector<cv::Mat> masks;
        //   std::vector<int> maxCorners;
        //   std::vector<float> minDistances;

        //   for (auto object_id : object_ids) {
        //     cv::Mat obj_mask = (object_masks == object_id);

        //     masks.push_back(obj_mask);
        //     maxCorners.push_back(300);
        //     minDistances.push_back(4);
        //   }

        //   cv::Mat object_masks_binary = object_masks > 0;
        //   cv::Mat static_mask;
        //   cv::bitwise_not(object_masks_binary, static_mask);

        //   masks.push_back(static_mask);
        //   maxCorners.push_back(1000);
        //   object_ids.push_back(0);
        //   minDistances.push_back(15);

        //   utils::ChronoTimingStats detection_t("batched.detection");
        //   auto detected_features = goodFeaturesToTrackBatched(
        //       current_mono, masks, maxCorners, minDistances, 0.01);
        //   auto time_ms = detection_t.stop();

        //   gtsam::FastMap<ObjectId, std::vector<cv::Point2f>>
        //   detected_feature_map; for (size_t i = 0; i <
        //   detected_features.size(); i++) {
        //     if (detected_features[i].size() > 0) {
        //       detected_feature_map[object_ids[i]] = detected_features[i];

        //       TrackingDetails details;
        //       details.object_id = object_ids[i];
        //       details.num_last_detected_features =
        //       detected_features[i].size();

        //       tracking_details[object_ids[i]] = details;
        //     }
        //   }
        //   previousBatchFeatures = detected_feature_map;

        //   cv::Mat viz = drawBatchedFeatures(container->rgb(),
        //   detected_feature_map); cv::imshow("batch Tracks", viz);
        // }

        // // LOG(INFO) << to_string(tracker->getTrackerInfo());

        // if (previous_frame) {
        //   ImageTracksParams track_viz_params(true);
        //   track_viz_params.show_intermediate_tracking = true;
        //   cv::Mat tracking = tracker->computeFeatureTracks(*previous_frame,
        //   *frame,
        //                                                    track_viz_params);

        //   cv::imshow("Tracks", tracking);

        //   // auto previous_mono =
        //   //
        //   ImageType::RGBMono::toMono(previous_frame->imageContainer().rgb());
        // }

        //   utils::ChronoTimingStats track_t("batched.track");
        //   auto tracked_features = trackFeaturesUnified(
        //       previous_mono, current_mono, previousBatchFeatures,
        //       object_masks);
        //   track_t.stop();

        //   // masks will be updated to include currently tracked points
        //   utils::ChronoTimingStats masks_t("batched.detection.masks");
        //   gtsam::FastMap<ObjectId, cv::Mat> mask_map;
        //   for (auto object_id : object_ids) {
        //     cv::Mat obj_mask = (object_masks == object_id);
        //     mask_map[object_id] = obj_mask;

        //     // cv::Rect detected_bb;
        //     // vision_tools::findObjectBoundingBox(object_masks, object_id,
        //     detected_bb);

        //     // LOG(INFO) << "Detected j=" << object_id << " " <<
        //     to_string(detected_bb);

        //     // masks.push_back(obj_mask);
        //     // maxCorners.push_back(300);
        //     // minDistances.push_back(4);
        //   }

        //   cv::Mat object_masks_binary = object_masks > 0;
        //   cv::Mat static_mask;
        //   cv::bitwise_not(object_masks_binary, static_mask);
        //   mask_map[0] = static_mask;

        //   std::set<ObjectId> object_needs_detection;
        //   for (auto& [j, per_object_tracks] : tracked_features) {
        //     LOG(INFO) << "Tracked features for j=" << j;
        //     int distance = j > 0 ? 8 : 15;
        //     for (const auto& kp : per_object_tracks) {
        //       // mark as location to ignore when doing feature detection
        //       cv::circle(mask_map[j], kp, distance, cv::Scalar(0),
        //       cv::FILLED);
        //     }

        //     auto num_tracked = per_object_tracks.size();
        //     auto num_previous =
        //     tracking_details[j].num_last_detected_features;
        //     // not since previous track but since the last detecion!
        //     const double survival_ratio =
        //         num_previous > 0 ? (double)num_tracked / (double)num_previous
        //         : 0.0;
        //     const bool poor_tracking = survival_ratio < 0.4;
        //     const bool too_few_tracks = static_cast<int>(num_tracked) < 30;

        //     bool needs_detection = poor_tracking || too_few_tracks;

        //     const bool is_object = j > 0;
        //     if(is_object) {
        //       const cv::Rect& detected_bounding_box =
        //       bounding_boxes[object_index[j]]; const cv::Rect
        //       tracked_bounding_box = cv::boundingRect(per_object_tracks);

        //       const double iou = utils::calculateIoU(detected_bounding_box,
        //       tracked_bounding_box); const auto min_iou = 0.4;

        //       const bool small_iou = iou < min_iou;

        //       needs_detection = needs_detection || small_iou;
        //     }

        //     if (needs_detection) {
        //       // hack for now
        //       LOG(INFO) << "Replacing tracks for poorly tracked object " <<
        //       j;
        //       // per_object_tracks = detected_feature_map[j];
        //       object_needs_detection.insert(j);

        //     } else {
        //       LOG(INFO) << "Good tracks for object " << j;
        //       // say previous object is well tracked
        //       // previous_objects.insert(j);
        //     }
        //   }

        //   std::vector<cv::Mat> masks;
        //   std::vector<int> maxCorners;
        //   std::vector<int> maxCornersAfterAnms;
        //   // min distance between detections
        //   std::vector<float> minDistances;
        //   ObjectIds objects_ids_for_detection;

        //   for (const auto& [j, masks_j] : mask_map) {
        //     // if does not exist in current tracking and we have detections
        //     // add as new object
        //     bool is_new = !tracked_features.exists(j);
        //     if (object_needs_detection.count(j) > 0 || is_new) {
        //       LOG(INFO) << "Object j " << j << " needs detection";
        //       masks.push_back(masks_j);
        //       objects_ids_for_detection.push_back(j);

        //       const int current_tracks = is_new ? 0 :
        //       tracked_features.at(j).size(); const int desired_tracks = j > 0
        //       ? 300 : 1000; const int tracked_needed =
        //           std::max(desired_tracks - current_tracks, 0);
        //       const int distance = j > 0 ? 8 : 15;

        //       maxCorners.push_back(tracked_needed);
        //       minDistances.push_back(distance);

        //       // // TODO: recompute maxCorners
        //       // if (j > 0) {
        //       //   maxCorners.push_back(300);
        //       //   minDistances.push_back(8);
        //       // } else {
        //       //   maxCorners.push_back(1000);
        //       //   minDistances.push_back(15);
        //       // }
        //     }
        //   }
        //   masks_t.stop();

        //   utils::ChronoTimingStats detection_t("batched.detection");

        //   AdaptiveNonMaximumSuppression non_maximum_supression(
        //         AnmsAlgorithmType::RangeTree);

        //   if (!masks.empty()) {
        //     auto detected_features = goodFeaturesToTrackBatched(
        //         current_mono, masks, maxCorners, minDistances, 0.01);

        //     gtsam::FastMap<ObjectId, std::vector<cv::Point2f>>
        //     detected_feature_map; for (size_t i = 0; i <
        //     detected_features.size(); i++) {
        //       if (detected_features[i].size() > 0) {
        //         const ObjectId object_id = objects_ids_for_detection[i];
        //         detected_feature_map[object_id] =
        //             detected_features[i];

        //         TrackingDetails details;
        //         details.object_id = object_id;
        //         details.num_last_detected_features =
        //         detected_features[i].size();
        //         tracking_details[details.object_id] = details;
        //       }
        //     }

        //     // for now just replace featues
        //     for (const auto& [j, per_object_tracks] : detected_feature_map) {
        //       // tracked_features[j] = per_object_tracks;
        //       // add tracks to existing tracks!

        //       tracked_features[j].insert(tracked_features[j].begin(),
        //                                  per_object_tracks.begin(),
        //                                  per_object_tracks.end());
        //     }
        //   }

        //   detection_t.stop();

        //   utils::ChronoTimingStats detection_viz_t("batched.detection_viz");
        //   cv::Mat tracked_viz =
        //       drawBatchedFeatures(container->rgb(), tracked_features);

        //   for(size_t i = 0; i < object_ids.size(); i++) {
        //     const auto object_id = object_ids[i];
        //     const auto bounding_box = bounding_boxes[i];

        //     utils::drawLabeledBoundingBox(tracked_viz,
        //     std::to_string(object_id),
        //     dyno::Color::uniqueObjectId(object_id).bgra(), bounding_box);
        //   }

        //   cv::imshow("batch Tracks", tracked_viz);
        //   previousBatchFeatures = tracked_features;
        // }
        // batch_all_t.stop();

        LOG(INFO) << utils::Statistics::Print();

        cv::waitKey(1);
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
