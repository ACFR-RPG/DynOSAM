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

// LKWrapper::LKWrapper(const cv::Size& win_size,
//               int max_level,
//               const cv::TermCriteria& criteria,
//               int flags,
//               double min_eig_threshold,)
//         : win_size_(win_size),
//         max_level_(max_level),
//         criteria_(criteria),
//         flags_(flags),
//         min_eig_threshold_(min_eig_threshold)
// {

//   use_cuda_ = false;
//   #ifdef DYNO_CUDA_OPENCV_ENABLED
//     if(utils::opencvCudaAvailable()) {
//       lk_cuda_ = cv::cuda::OpticalFlowPyrLK::create(win_size_, max_level_,
//                                                           criteria_.maxCount,
//                                                           criteria_.epsilon,
//                                                           flags_,
//                                                           min_eig_threshold_);
//       use_cuda_ = true;
//     }
//   #endif

// }

void getCorrespondences(FeaturePairs& correspondences,
                        const FeatureFilterIterator& previous_features,
                        const FeatureFilterIterator& current_features) {
  correspondences.clear();

  const FeatureContainer& previous_feature_container =
      previous_features.getContainer();

  for (const auto& curr_feature : current_features) {
    // check if previous feature and is valid
    if (previous_feature_container.exists(curr_feature->trackletId())) {
      const auto prev_feature = previous_feature_container.getByTrackletId(
          curr_feature->trackletId());
      CHECK(prev_feature);

      // having checked that feature is in the previous set, also check that it
      // ahderes to the filter
      if (!previous_features(prev_feature)) {
        continue;
      }
      correspondences.push_back({prev_feature, curr_feature});
    }
  }
}

ObjectIds getObjectLabels(const cv::Mat& image) {
  // CHECK(!image.empty());
  // std::unordered_set<ObjectId> unique_labels;
  // for (auto it = image.begin<ObjectId>(); it != image.end<ObjectId>(); ++it)
  // {
  //   if (*it != background_label) {
  //     unique_labels.insert(*it);
  //   }
  // }
  // return ObjectIds(unique_labels.begin(), unique_labels.end());
  // std::vector<ObjectId> v(image.ptr<ObjectId>(), image.ptr<ObjectId>() +
  // image.total()); std::sort(v.begin(), v.end()); auto last =
  // std::unique(v.begin(), v.end()); v.erase(last, v.end());
  // v.erase(std::remove(v.begin(), v.end(), 0), v.end());
  // return v;

  // from testing in test_code_concepts.cc (CodeConcepts.uniqueLabelSpeed)
  // this implementation is up to 28x faster than a simple a std::set approach!!
  const int numThreads =
      std::min(std::thread::hardware_concurrency(), (unsigned int)image.rows);
  const int rowsPerThread = image.rows / numThreads;

  std::vector<std::future<std::unordered_set<int>>> futures;

  // Launch threads to process row chunks
  for (int t = 0; t < numThreads; ++t) {
    int startRow = t * rowsPerThread;
    int endRow = (t == numThreads - 1) ? image.rows : (t + 1) * rowsPerThread;

    futures.push_back(
        std::async(std::launch::async, [&image, startRow, endRow]() {
          std::unordered_set<int> localUnique;
          for (int row = startRow; row < endRow; ++row) {
            const int* rowPtr = image.ptr<int>(row);
            for (int col = 0; col < image.cols; ++col) {
              localUnique.insert(rowPtr[col]);
            }
          }
          return localUnique;
        }));
  }

  // Merge results
  std::unordered_set<int> globalUnique;
  for (auto& future : futures) {
    auto localSet = future.get();
    globalUnique.insert(localSet.begin(), localSet.end());
  }
  // dont include background label!!
  globalUnique.erase(background_label);

  // Convert to vector (NOTE: not sorted!!)
  std::vector<int> result(globalUnique.begin(), globalUnique.end());
  // std::sort(result.begin(), result.end());

  return result;
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
  cv::Mat mask_copy = mask.clone();

  cv::Mat obj_mask = (mask_copy == object_id);
  cv::Mat dilated_obj_mask;
  // dilate to fill any small holes in the mask to get a more complete set of
  // contours
  cv::Mat dilate_element = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(1, 11));  // a rectangle of 1*5
  cv::dilate(obj_mask, dilated_obj_mask, dilate_element, cv::Point(-1, -1));

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(dilated_obj_mask, contours, hierarchy, cv::RETR_TREE,
                   cv::CHAIN_APPROX_NONE);

  detected_contours = contours;

  if (contours.empty()) {
    detected_rect = cv::Rect();
    return false;
  } else if (contours.size() == 1u) {
    detected_rect = cv::boundingRect(contours.at(0));
  } else {
    std::vector<cv::Rect> rectangles;
    for (auto it : contours) {
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

void computeObjectMaskBoundaryMask(ObjectBoundaryMaskResult& result,
                                   const cv::Mat& mask, int thickness,
                                   bool use_as_feature_detection_mask) {
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

  result.objects_detected = vision_tools::getObjectLabels(mask);
  // this basically just creates a full mask over the existing masks using the
  // detected contours
  for (const auto object_id : result.objects_detected) {
    std::vector<std::vector<cv::Point>> detected_contours;
    cv::Rect detected_rect;
    vision_tools::findObjectBoundingBox(mask, object_id, detected_rect,
                                        detected_contours);

    // cv::drawContours(thicc_boarder, detected_contours, -1, 255, cv::FILLED);
    CHECK_LE(object_id, 255);  // works only with uint8 types...
    cv::drawContours(thicc_boarder, detected_contours, -1, object_id,
                     cv::FILLED);
    result.object_bounding_boxes.push_back(detected_rect);
  }
  // cv::Mat thicc_boarder_8u; //viz
  // cv::threshold(thicc_boarder, thicc_boarder_8u, 0, 255, cv::THRESH_BINARY);
  // cv::imshow("Thicc boarder", thicc_boarder_8u);
  // cv::waitKey(1);

  // Dilate the mask to expand outwards by 'thickness' pixels
  cv::Mat dilated_mask;
  cv::dilate(thicc_boarder, dilated_mask,
             cv::getStructuringElement(
                 cv::MORPH_ELLIPSE,
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
                cv::MORPH_ELLIPSE,
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

void relabelMasks(const cv::Mat& mask, cv::Mat& relabelled_mask,
                  const ObjectIds& old_labels, const ObjectIds& new_labels) {
  if (old_labels.size() != new_labels.size()) {
    throw std::invalid_argument(
        "Old labels and new labels must have the same size");
  }

  // Create a map from old labels to new labels
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

  // collect dynamic features
  for (const auto& [object_id, observations] : frame->getObjectObservations()) {
    Histogram hist(bh::make_histogram(bh::axis::variable<>(bins)));
    hist.name_ = "tacklet-length-" + std::to_string(object_id);

    for (auto tracklet_id : observations.object_features) {
      const Feature::Ptr feature = frame->at(tracklet_id);
      CHECK(feature);
      if (feature->usable()) {
        hist.histogram_(feature->age());
      }
    }

    histograms.insert2(object_id, hist);
  }

  // collect static features
  Histogram static_hist(bh::make_histogram(bh::axis::variable<>(bins)));
  static_hist.name_ = "tacklet-length-0";
  for (const auto& static_feature : frame->static_features_.beginUsable()) {
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
