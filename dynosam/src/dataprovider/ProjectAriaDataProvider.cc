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

#include "dynosam/dataprovider/ProjectAriaDataProvider.hpp"

#include <nlohmann/json.hpp>

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/utils/CsvParser.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {

class ProjectAriaAllLoader {
 public:
  DYNO_POINTER_TYPEDEFS(ProjectAriaAllLoader)

  ProjectAriaAllLoader(const std::string& file_path)
      : rgb_images_folder_path_(file_path + "/rgb_sync"),
        right_rgb_images_folder_path_(file_path + "/right"),
        depth_images_folder_path_(file_path + "/depth_sync"),
        optical_flow_folder_path_(file_path + "/optical_flow"),
        instance_masks_folder_(file_path + "/instance_masks"),
        intrinsics_file_path_(file_path + "/calibration_undistort.json"),
        rgb_timestamps_file_path_(file_path + "/sync_timestamp.csv") {
    // Initialize folders and file paths
    utils::throwExceptionIfPathInvalid(rgb_images_folder_path_);
    utils::throwExceptionIfPathInvalid(depth_images_folder_path_);
    utils::throwExceptionIfPathInvalid(optical_flow_folder_path_);
    utils::throwExceptionIfPathInvalid(instance_masks_folder_);
    // utils::throwExceptionIfPathInvalid(intrinsics_file_path_);
    // utils::throwExceptionIfPathInvalid(rgb_timestamps_file_path_);

    // first load images and size
    // the size will be used as a refernce for all other loaders
    // size is number of images
    // we use flow to set the size as for this dataset, there will be one less
    // optical flow as the flow ids index t to t+1 (which is gross, I know!!)
    // TODO: code and comments replicated in ClusterSlamDataProvider
    loadFlowImagesAndSize(optical_flow_image_paths_, dataset_size_);

    // load rgb, depth and instance masks - they should all have the same
    // length!!
    loadOtherImages();
    // loadTimestamps();
    loadCalibration();
  }

  size_t size() const { return dataset_size_; }

  cv::Mat getOpticalFlow(size_t idx) const {
    CHECK_LT(idx, optical_flow_image_paths_.size());

    cv::Mat flow;
    utils::loadFlow(optical_flow_image_paths_.at(idx), flow);
    CHECK(!flow.empty());
    return flow;
  }

  cv::Mat getRGB(size_t idx) const {
    CHECK_LT(idx, rgb_image_paths_.size());

    cv::Mat rgb;
    utils::loadRGB(rgb_image_paths_.at(idx), rgb);
    CHECK(!rgb.empty());

    return rgb;
  }

  cv::Mat getRightRGB(size_t idx) const {
    CHECK_LT(idx, right_rgb_image_paths_.size());

    cv::Mat rgb;
    utils::loadRGB(right_rgb_image_paths_.at(idx), rgb);
    CHECK(!rgb.empty());

    return rgb;
  }

  cv::Mat getInstanceMask(size_t idx) const {
    CHECK_LT(idx, rgb_image_paths_.size());
    CHECK_LT(idx, dataset_size_);

    // relabel masks from 1 to N
    static gtsam::FastMap<ObjectId, ObjectId> relabelled;
    static ObjectId current_max_label = 1u;

    cv::Mat mask;
    utils::loadMask(instance_masks_image_paths_.at(idx), mask);
    CHECK(!mask.empty());

    ObjectIds labels = vision_tools::getObjectLabels(mask);
    const ObjectIds& old_labels = labels;
    ObjectIds new_labels;
    for (ObjectId old_label : labels) {
      if (!relabelled.exists(old_label)) {
        relabelled[old_label] = current_max_label;
        current_max_label++;
      }
      ObjectId new_label = relabelled.at(old_label);
      new_labels.push_back(new_label);
    }

    cv::Mat relabelled_mask;
    vision_tools::relabelMasks(mask, relabelled_mask, old_labels, new_labels);

    return relabelled_mask;
  }

  cv::Mat getDepthImage(size_t idx) const {
    CHECK_LT(idx, depth_image_paths_.size());

    cv::Mat depth;
    utils::loadDepth(depth_image_paths_.at(idx), depth);
    CHECK(!depth.empty());

    return depth;
  }

  const CameraParams& getLeftCameraParams() const { return rgb_camera_params_; }

  double getTimestamp(size_t idx) {
    CHECK_LT(idx, times_.size());
    return times_.at(idx);
  }

 private:
  void loadFlowImagesAndSize(std::vector<std::string>& images_paths,
                             size_t& dataset_size) {
    std::vector<std::filesystem::path> files_in_directory =
        utils::getAllFilesInDir(optical_flow_folder_path_);
    dataset_size = files_in_directory.size();
    CHECK_GT(dataset_size, 0);

    for (const std::filesystem::path& path : files_in_directory) {
      const std::string file_path = path;
      utils::throwExceptionIfPathInvalid(file_path);
      images_paths.push_back(file_path);

      std::string timestamp_str = path.stem();
      double timestamp_nano = std::stod(timestamp_str);
      double timestamp_seconds = timestamp_nano / 1.0e+9;
      times_.push_back(timestamp_seconds);
    }
  }

  void loadOtherImages() {
    auto rgb_image_files = utils::getAllFilesInDir(rgb_images_folder_path_);
    auto right_rgb_images_files =
        utils::getAllFilesInDir(right_rgb_images_folder_path_);
    auto depth_image_files = utils::getAllFilesInDir(depth_images_folder_path_);
    auto instance_masks_image_files =
        utils::getAllFilesInDir(instance_masks_folder_);

    for (auto f : rgb_image_files) {
      rgb_image_paths_.push_back(f);
    }

    // left images are sync and right images are raw so we need one less
    for (auto f : right_rgb_images_files) {
      right_rgb_image_paths_.push_back(f);
    }
    right_rgb_image_paths_.pop_back();

    for (auto f : depth_image_files) {
      depth_image_paths_.push_back(f);
    }
    for (auto f : instance_masks_image_files) {
      instance_masks_image_paths_.push_back(f);
    }

    CHECK_EQ(rgb_image_paths_.size(), depth_image_paths_.size());
    CHECK_EQ(right_rgb_image_paths_.size(), depth_image_paths_.size());
    CHECK_EQ(depth_image_paths_.size(), instance_masks_image_paths_.size());

    // dataset size is number optical flow which should be one less!!
    // or equal too!!
    //  CHECK_EQ(instance_masks_image_paths_.size() -1u, dataset_size_);
  }

  // void loadTimestamps() {
  //     std::ifstream file(rgb_timestamps_file_path_);
  //     CsvReader csv_reader(file);
  //     auto it = csv_reader.begin();
  //     //skip header
  //     ++it;
  //     bool is_first_timestamp = true;
  //     Timestamp time_offset = 0; //first timestamp
  //     for(; it != csv_reader.end(); it++) {
  //         const auto row = *it;

  //         double time_ms = row.at<double>(0);
  //         Timestamp time_s = time_ms / 1000.0;

  //         if(is_first_timestamp) {
  //             time_offset = time_s;
  //             is_first_timestamp = false;
  //         }

  //         //start time at zero with first timestamp offset
  //         Timestamp offset_time_s = time_s - time_offset;
  //         times_.push_back(offset_time_s);

  //     }
  //     LOG(INFO) << "Loaded " << times_.size() << " timestamps from RGBD csv
  //     file: " << rgb_timestamps_file_path_;

  // }

  void loadCalibration() {
    // using json = nlohmann::json;

    // std::ifstream file(intrinsics_file_path_);
    // json calibration_json;
    // file >> calibration_json;

    // int rgb_width = calibration_json["rgb_width"].template get<int>();
    // int rgb_height = calibration_json["rgb_height"].template get<int>();

    // std::vector<double> K = calibration_json["rgb_intrinsics"].template
    // get<std::vector<double>>(); CameraParams::IntrinsicsCoeffs intrinsics({
    //     K.at(0), //fu
    //     K.at(4), //fv
    //     K.at(2), //cu
    //     K.at(5)  // cv
    //     });

    // CameraParams::IntrinsicsCoeffs intrinsics({
    //     535.288025, //fu
    //     535.288025, //fv
    //     623.312256, //cu
    //     348.522400  // cv
    //     });

    // acfr_1_moving_small
    CameraParams::IntrinsicsCoeffs intrinsics({
        267.644012,  // fu
        311.656128,  // fv
        267.644012,  // cu
        174.261200   // cv
    });

    // TODO: Assume Image is undistorted!!!!!!
    CameraParams::DistortionCoeffs distortion({0, 0, 0, 0});

    // distortion_model not given
    const auto distortion_model = "plumb_bob";
    const auto camera_model = "pinhole";
    auto model =
        CameraParams::stringToDistortion(distortion_model, camera_model);

    rgb_camera_params_ = CameraParams(intrinsics, distortion,
                                      // cv::Size(1280, 720),
                                      cv::Size(640, 360), model);
  }

 private:
  std::string rgb_images_folder_path_;
  std::string right_rgb_images_folder_path_;
  std::string depth_images_folder_path_;
  std::string optical_flow_folder_path_;
  std::string instance_masks_folder_;
  std::string intrinsics_file_path_;
  std::string rgb_timestamps_file_path_;

  std::vector<std::string> rgb_image_paths_;
  std::vector<std::string> right_rgb_image_paths_;
  std::vector<std::string> depth_image_paths_;
  std::vector<std::string> optical_flow_image_paths_;
  std::vector<std::string> instance_masks_image_paths_;

  std::vector<Timestamp> times_;

  size_t dataset_size_ = 0;
  CameraParams rgb_camera_params_;
};

struct ProjectAriaTimestampLoader : public TimestampBaseLoader {
  ProjectAriaAllLoader::Ptr loader_;

  ProjectAriaTimestampLoader(ProjectAriaAllLoader::Ptr loader)
      : loader_(CHECK_NOTNULL(loader)) {}
  std::string getFolderName() const override { return ""; }

  size_t size() const override { return loader_->size(); }

  double getItem(size_t idx) override { return loader_->getTimestamp(idx); }
};

ProjectARIADataLoader::ProjectARIADataLoader(const fs::path& dataset_path)
    : ProjectAriaDatasetProvider(dataset_path) {
  LOG(INFO) << "Starting ProjectARIADataLoader with path" << dataset_path;

  // this would go out of scope but we capture it in the functional loaders
  auto loader = std::make_shared<ProjectAriaAllLoader>(dataset_path);
  auto timestamp_loader = std::make_shared<ProjectAriaTimestampLoader>(loader);

  left_camera_params_ = loader->getLeftCameraParams();

  CHECK(getCameraParams());

  auto rgb_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getRGB(idx); });

  auto optical_flow_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getOpticalFlow(idx); });

  auto depth_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getDepthImage(idx); });

  auto instance_mask_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getInstanceMask(idx); });

  auto right_rgb =
      std::make_shared<FunctionalDataFolder<std::optional<cv::Mat>>>(
          [loader](size_t idx) { return loader->getRightRGB(idx); });

  // auto gt_loader =
  // std::make_shared<FunctionalDataFolder<GroundTruthInputPacket>>(
  //     [loader](size_t idx) {
  //         return loader->getGtPacket(idx);
  //     }
  // );

  this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader,
                   depth_loader, instance_mask_loader, right_rgb
                   // gt_loader
  );

  auto callback = [&](size_t frame_id, Timestamp timestamp, cv::Mat rgb,
                      cv::Mat optical_flow, cv::Mat depth,
                      cv::Mat instance_mask, std::optional<cv::Mat> right_rgb
                      /**GroundTruthInputPacket gt_object_pose_gt**/) -> bool {
    // CHECK_EQ(timestamp, gt_object_pose_gt.timestamp_);
    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
        .depth(depth)
        .opticalFlow(optical_flow)
        .objectMotionMask(instance_mask);

    // if (right_rgb) image_container.rightRgb(right_rgb.value());

    CHECK(image_container_callback_);
    if (image_container_callback_)
      image_container_callback_(
          std::make_shared<ImageContainer>(image_container));
    return true;
  };

  this->setCallback(callback);
}

}  // namespace dyno
