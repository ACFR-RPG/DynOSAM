/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/dataprovider/TartanAirShibuya.hpp"

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_cv/CameraParams.hpp"

namespace dyno {

class TartanAirShibuyaAllLoader {
 public:
  DYNO_POINTER_TYPEDEFS(TartanAirShibuyaAllLoader)

  TartanAirShibuyaAllLoader(const std::string& file_path) {
    // load flow images
    const auto flow_image_path = file_path + "/flow_0/";
    utils::throwExceptionIfPathInvalid(flow_image_path);
    loadFlowImagesAndSize(flow_0_paths_, dataset_size_, flow_image_path);

    // load images
    const auto rgb_image_path = file_path + "/image_0/";
    const auto depth_image_path = file_path + "/depth_0/";
    const auto mask_image_path = file_path + "/mask_0/";

    utils::throwExceptionIfPathInvalid(rgb_image_path);
    utils::throwExceptionIfPathInvalid(depth_image_path);
    utils::throwExceptionIfPathInvalid(mask_image_path);

    loadImages(image_0_paths_, rgb_image_path);
    loadImages(depth_0_paths_, depth_image_path);
    loadImages(masks_0_paths_, mask_image_path);

    camera_params_ =
        CameraParams(CameraParams::IntrinsicsCoeffs(
                         {772.5483399593904, 772.5483399593904, 320.0, 180.0}),
                     CameraParams::DistortionCoeffs({0, 0, 0, 0}),
                     cv::Size(640, 360), DistortionModel::RADTAN);

    const auto times_file_path = file_path + "/times.txt";
    utils::throwExceptionIfPathInvalid(times_file_path);
    loadTimes(times_file_path);

    const auto gt_file_path = file_path + "/gt_pose.txt";
    utils::throwExceptionIfPathInvalid(gt_file_path);
    loadGroundTruth(gt_file_path);
  }

  cv::Mat getOpticalFlow(size_t idx) const {
    CHECK_LT(idx, flow_0_paths_.size());

    cv::Mat flow;
    utils::loadFlow(flow_0_paths_.at(idx), flow);
    CHECK(!flow.empty());
    return flow;
  }

  cv::Mat getRGB(size_t idx) const {
    CHECK_LT(idx, image_0_paths_.size());
    cv::Mat rgb;
    utils::loadRGB(image_0_paths_.at(idx), rgb);
    CHECK(!rgb.empty());
    return rgb;
  }

  cv::Mat getInstanceMask(size_t idx) const {
    CHECK_LT(idx, masks_0_paths_.size());
    CHECK_LT(idx, dataset_size_);

    cv::Mat mask;
    utils::loadMask(masks_0_paths_.at(idx), mask);
    CHECK(!mask.empty());

    return mask;
  }

  cv::Mat getDepthImage(size_t idx) const {
    CHECK_LT(idx, depth_0_paths_.size());
    CHECK_LT(idx, dataset_size_);

    cv::Mat depth;
    utils::loadDepth(depth_0_paths_.at(idx), depth);
    CHECK(!depth.empty());

    return depth;
  }

  const GroundTruthInputPacket& getGtPacket(size_t idx) const {
    return ground_truth_packets_.at(idx);
  }

  const CameraParams& getLeftCameraParams() const { return camera_params_; }

  size_t size() const { return dataset_size_; }

  double getTimestamp(size_t idx) {
    CHECK_LT(idx, image_0_paths_.size());
    CHECK_LT(idx, dataset_size_);
    return times_.at(idx);
  }

 private:
  void loadFlowImagesAndSize(std::vector<std::string>& images_paths,
                             size_t& dataset_size,
                             const std::string& flow_image_path) {
    std::vector<std::filesystem::path> files_in_directory =
        utils::getAllFilesInDir(flow_image_path);
    dataset_size = files_in_directory.size();
    CHECK_GT(dataset_size, 0);

    for (const std::string file_path : files_in_directory) {
      utils::throwExceptionIfPathInvalid(file_path);
      images_paths.push_back(file_path);
    }
  }

  void loadImages(std::vector<std::string>& images_paths,
                  const std::string& image_folder) {
    std::vector<std::filesystem::path> files_in_directory =
        utils::getAllFilesInDir(image_folder);

    for (const std::filesystem::path& file_path : files_in_directory) {
      utils::throwExceptionIfPathInvalid(file_path);

      // only load png files
      auto ext = file_path.extension().string();
      if (ext == ".png") {
        images_paths.push_back((std::string)file_path);
      }
    }
  }

  void loadTimes(const std::string& times_file) {
    std::ifstream infile(times_file);
    if (!infile) {
      throw std::runtime_error("Could not open file " + times_file +
                               " when trying to load Tartan Air Shibuya!");
    }

    Timestamp value;
    while (infile >> value) {
      times_.push_back(value);
    }

    // NOTE: the timestamps (for some reason) are not (all) in ascending order
    // the images appear to be in order (there is no jump in motion visually the
    // place where the timestamps jump) we assume something is weird with the
    // timestamps only and therefore just re-order these!!!
    std::sort(times_.begin(), times_.end());
  }

  void loadGroundTruth(const std::string& gt_file) {
    const gtsam::Rot3 R_NED_CV((gtsam::Matrix3() << 0, 0,
                                1,  // X_cv (right)   = 0·x +1·y +0·z_NED
                                1, 0, 0,  // Y_cv (down)    = 0·x +0·y +1·z_NED
                                0, 1, 0)  // Z_cv (forward) = 1·x +0·y +0·z_NED
                                   .finished());

    std::ifstream fin(gt_file);
    if (!fin) throw std::runtime_error("Cannot open file: " + gt_file);

    gtsam::Pose3 initial_pose = gtsam::Pose3::Identity();
    bool initial_frame_set = false;

    std::string line;
    FrameId frame = 0;
    while (std::getline(fin, line)) {
      if (line.empty() || line[0] == '#') continue;  // skip blank / comment

      std::istringstream iss(line);
      double t, tx, ty, tz, qx, qy, qz, qw;
      if (!(iss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
        throw std::runtime_error("Malformed line: " + line);

      // Orientation as given: world_R_cam_NED (w first for GTSAM)
      gtsam::Rot3 world_R_camNED = gtsam::Rot3::Quaternion(qw, qx, qy, qz);

      // Convert to world_R_camCV
      gtsam::Rot3 world_R_camCV = world_R_camNED * R_NED_CV;

      gtsam::Pose3 gt_pose(world_R_camCV, gtsam::Point3(tx, ty, tz));

      if (!initial_frame_set) {
        // expect very first pose (first frame, and first pose within that
        // frame) to be the first camera pose
        initial_pose = gt_pose;
        initial_frame_set = true;
      }

      // offset initial pose so we start at "0, 0, 0"
      gt_pose = initial_pose.inverse() * gt_pose;

      GroundTruthInputPacket gt_packet;
      gt_packet.frame_id_ = frame;
      gt_packet.timestamp_ = t;
      gt_packet.X_world_ = gt_pose;

      ground_truth_packets_.insert2(frame, gt_packet);
      frame++;
    }
  }

 public:
  size_t dataset_size_;

  std::vector<std::string> image_0_paths_;
  std::vector<std::string> depth_0_paths_;
  std::vector<std::string> flow_0_paths_;
  std::vector<std::string> masks_0_paths_;

  std::vector<Timestamp> times_;

  GroundTruthPacketMap ground_truth_packets_;
  CameraParams camera_params_;
};

struct TartanAirShibuyaTimestampLoader : public TimestampBaseLoader {
  TartanAirShibuyaAllLoader::Ptr loader_;

  TartanAirShibuyaTimestampLoader(TartanAirShibuyaAllLoader::Ptr loader)
      : loader_(CHECK_NOTNULL(loader)) {}
  std::string getFolderName() const override { return ""; }

  size_t size() const override { return loader_->size(); }

  double getItem(size_t idx) override { return loader_->getTimestamp(idx); }
};

TartanAirShibuyaLoader::TartanAirShibuyaLoader(const fs::path& dataset_path)
    : TartanAirShibuyaProvider(dataset_path) {
  LOG(INFO) << "Starting TartanAirShibuyaLoader with path" << dataset_path;

  // this would go out of scope but we capture it in the functional loaders
  auto loader = std::make_shared<TartanAirShibuyaAllLoader>(dataset_path);
  auto timestamp_loader =
      std::make_shared<TartanAirShibuyaTimestampLoader>(loader);

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

  auto gt_loader =
      std::make_shared<FunctionalDataFolder<GroundTruthInputPacket>>(
          [loader](size_t idx) { return loader->getGtPacket(idx); });

  this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader,
                   depth_loader, instance_mask_loader, gt_loader);

  auto callback = [&](size_t frame_id, Timestamp timestamp, cv::Mat rgb,
                      cv::Mat optical_flow, cv::Mat depth,
                      cv::Mat instance_mask,
                      GroundTruthInputPacket gt_object_pose_gt) -> bool {
    // TODO: for now
    //  CHECK_EQ(timestamp, gt_object_pose_gt.timestamp_);

    CHECK(ground_truth_packet_callback_);
    if (ground_truth_packet_callback_)
      ground_truth_packet_callback_(gt_object_pose_gt);

    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
        .depth(depth)
        .opticalFlow(optical_flow)
        .objectMotionMask(instance_mask);
    CHECK(image_container_callback_);
    if (image_container_callback_)
      image_container_callback_(
          std::make_shared<ImageContainer>(image_container));
    return true;
  };

  this->setCallback(callback);
}

}  // namespace dyno
