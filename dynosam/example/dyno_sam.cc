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

#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/OccupancyGrid2D.hpp"
#include "dynosam/pipeline/PipelineManager.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_nn/PyObjectDetector.hpp"
#include "dynosam_processing/datasets/ClusterSlamDataProvider.hpp"
#include "dynosam_processing/datasets/KittiDataProvider.hpp"
#include "dynosam_processing/datasets/TartanAirShibuya.hpp"
#include "dynosam_processing/datasets/ViodeDataProvider.hpp"
#include "dynosam_processing/datasets/VirtualKittiDataProvider.hpp"
#include "dynosam_sensors/Camera.hpp"
#include "dynosam_sensors/ImageContainer.hpp"

DEFINE_string(path_to_kitti, "/root/data/kitti", "Path to KITTI dataset");
// TODO: (jesse) many better ways to do this with ros - just for now
DEFINE_string(
    params_path, "dynosam/params",
    "Path to the folder containing the yaml files with the VIO parameters.");

#include "dynosam/frontend/vision/FeatureTrackerFast.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_processing/datasets/DynopetsDataProvider.hpp"
#include "dynosam_processing/datasets/KittiDataProvider.hpp"
#include "dynosam_processing/datasets/OMDDataProvider.hpp"

/**
 * TODO:
 * retroactive tracking
 * reimplement re-detection features (ie survivial rate)
 * properly fill out FeatureData and ensure tracklet ids are correctly
 * generated/propogated check which properties we actually want in the
 *
 * test additional outlier rejection with opengv
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
  FLAGS_v = 60;

  // KittiDataLoader::Params params;
  // KittiDataLoader loader("/root/data/vdo_slam/kitti/kitti/0020/", params);
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
  //   TrackerParams::FeatureDetectorType::GFFT_CUDA;
  fp.tracker_params.max_nr_keypoints_before_anms = 1000;
  fp.tracker_params.max_dynamic_features_per_frame = 200;
  fp.tracker_params.prefer_provided_optical_flow = false;
  fp.tracker_params.prefer_provided_object_detection = true;
  fp.tracker_params.max_dynamic_feature_age = 1000;
  fp.tracker_params.max_feature_track_age = 1000;
  // fp.tracker_params.feature_detector_type =
  // TrackerParams::FeatureDetectorType::ORB_SLAM_ORB;
  // fp.tracker_params.min_distance_btw_tracked_and_detected_static_features =
  // 15;
  // fp.tracker_params.min_distance_btw_tracked_and_detected_dynamic_features =
  // 8;

  auto camera = std::make_shared<Camera>(loader.getCameraParams());
  auto tracker = std::make_shared<FeatureTracker>(fp, camera);

  FeatureTrackerFast ftf(fp, camera);

  // // std::vector<std::vector<cv::Point2f>> previousBatchFeatures;
  // gtsam::FastMap<ObjectId, std::vector<cv::Point2f>> previousBatchFeatures;
  // gtsam::FastMap<ObjectId, TrackingDetails> tracking_details;
  // ObjectIds prevObjectIds;

  // FeatureTrackerBatch ftb;

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

        auto frame = tracker->track(frame_id, timestamp, *container);
        auto tracking_result = ftf.track(frame_id, timestamp, *container);
        // ftb.track(container->rgb(), container->objectMotionMask());

        // // LOG(INFO) << "Batched extraction: " << time_ms << " [ms]";
        LOG(INFO) << to_string(tracker->getTrackerInfo());

        const WrappedRGBMono wrapped_rgb = container->rgb();
        const cv::Mat rgb = wrapped_rgb.image();
        cv::imshow("Fast Track",
                   drawBatchedFeatures(rgb, tracking_result.featues, true));

        // // // cv::waitKey(0);
        Frame::Ptr previous_frame = tracker->getPreviousFrame();
        // utils::ChronoTimingStats batch_all_t("batched.all");
        // const cv::Mat object_masks = container->objectMotionMask();
        // const cv::Mat current_mono =
        // ImageType::RGBMono::toMono(container->rgb());

        // utils::ChronoTimingStats labels_t("batched.get_labels");
        // ObjectIds object_ids = getObjectLabels(object_masks);
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

        if (previous_frame) {
          ImageTracksParams track_viz_params(true);
          track_viz_params.show_intermediate_tracking = true;
          cv::Mat tracking = tracker->computeFeatureTracks(
              *previous_frame, *frame, track_viz_params);

          cv::imshow("Tracks", tracking);
        }

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
        //     // findObjectBoundingBox(object_masks, object_id,
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

  while (loader.spin()) {
  }
}
