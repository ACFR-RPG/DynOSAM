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

#include "dynosam/dataprovider/ViodeDataProvider.hpp"

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/frontend/imu/Imu-Definitions.hpp"
#include "dynosam/frontend/imu/ThreadSafeImuBuffer.hpp"
#include "dynosam/frontend/vision/StereoMatcher.hpp"
#include "dynosam/pipeline/ThreadSafeTemporalBuffer.hpp"
#include "dynosam_common/utils/CsvParser.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_vision_common/CameraParams.hpp"

namespace dyno {

class ViodeAllLoader {
 public:
  DYNO_POINTER_TYPEDEFS(ViodeAllLoader)

  ViodeAllLoader(const std::string& file_path) {
    syncFilePathsWithGroundTruth(file_path);
    setSensorParams();
  }

  cv::Mat getOpticalFlow(size_t idx) const {
    CHECK_LT(idx, flow_0_paths_.size());

    cv::Mat flow;
    utils::loadFlow(flow_0_paths_.at(idx), flow);
    CHECK(!flow.empty());
    return flow;
  }

  cv::Mat getRGB(size_t idx) const {
    CHECK_LT(idx, image_left_paths_.size());
    cv::Mat rgb;
    utils::loadRGB(image_left_paths_.at(idx), rgb);
    CHECK(!rgb.empty());
    return rgb;
  }

  cv::Mat getRightRGB(size_t idx) const {
    CHECK_LT(idx, image_right_paths_.size());
    cv::Mat rgb;
    utils::loadRGB(image_right_paths_.at(idx), rgb);
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
    cv::Mat rgb_left = getRGB(idx);
    cv::Mat rgb_right = getRightRGB(idx);

    // this set of images are loaded as 8UC4
    CHECK_EQ(rgb_right.type(), CV_8UC3)
        << "Somehow the image type has changed...";
    cv::Mat depth_image;
    CHECK_NOTNULL(stereo_matcher_)
        ->denseStereoReconstruction(rgb_left, rgb_right, depth_image);

    return depth_image;
  }

  ImuMeasurements getImuMeasurements(size_t idx) {
    return imu_measrements_.at(idx);
  }

  const GroundTruthInputPacket& getGtPacket(size_t idx) const {
    return ground_truth_packets_.at(idx);
  }

  const CameraParams& getLeftCameraParams() const { return camera_params_; }

  size_t size() const { return dataset_size_; }

  double getTimestamp(size_t idx) {
    CHECK_LT(idx, image_left_paths_.size());
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

  void syncFilePathsWithGroundTruth(const std::string& file_path) {
    const auto flow_image_path = file_path + "/cam0/flow_0/";
    utils::throwExceptionIfPathInvalid(flow_image_path);
    // load images
    const auto left_image_path = file_path + "/cam0/image_raw/";
    const auto right_image_path = file_path + "/cam1/image_raw/";
    const auto mask_image_path = file_path + "/cam0/mask_0/";

    utils::throwExceptionIfPathInvalid(left_image_path);
    utils::throwExceptionIfPathInvalid(right_image_path);
    utils::throwExceptionIfPathInvalid(mask_image_path);

    const auto gt_file_path = file_path + "/odometry_odom.csv";
    utils::throwExceptionIfPathInvalid(gt_file_path);
    LOG(INFO) << "GT Odometry File path checked at" << gt_file_path;

    std::ifstream infile(gt_file_path);
    if (!infile) {
      throw std::runtime_error("Could not open file " + gt_file_path +
                               " when trying to load VIODE!");
    }

    ThreadsafeTemporalBuffer<gtsam::Pose3> gt_odom_buffer;
    const CsvReader reader(infile);
    for (const auto& row : reader) {
      // time should be in seconds
      double stamp = row.at<double>(0);

      double tx = row.at<double>(1);
      double ty = row.at<double>(2);
      double tz = row.at<double>(3);

      double qx = row.at<double>(4);
      double qy = row.at<double>(5);
      double qz = row.at<double>(6);
      double qw = row.at<double>(7);

      gtsam::Pose3 gt_pose(gtsam::Rot3(qw, qx, qy, qz),
                           gtsam::Point3(tx, ty, tz));
      gt_odom_buffer.addValue(stamp, gt_pose);
    }
    LOG(INFO) << "Loaded gt odom";

    const auto imu_file_path = file_path + "/imu0_imu.csv";
    utils::throwExceptionIfPathInvalid(imu_file_path);
    LOG(INFO) << "IMU File path checked at" << imu_file_path;

    std::ifstream imu_infile(imu_file_path);
    if (!imu_infile) {
      throw std::runtime_error("Could not open file " + imu_file_path +
                               " when trying to load VIODE!");
    }

    static const gtsam::Rot3 VIODE_left_hand_transform(
        (gtsam::Matrix3() << 0, 0,
         1,        // X_cv (right)   = 0·x +1·y +0·z_NED
         1, 0, 0,  // Y_cv (down)    = 0·x +0·y +1·z_NED
         0, 1, 0)  // Z_cv (forward) = 1·x +0·y +0·z_NED
            .finished());

    static const gtsam::Rot3 R_cv_robotic(
        (gtsam::Matrix3() << 0, -1, 0, 0, 0, -1, 1, 0, 0).finished());

    ThreadsafeImuBuffer imu_buffer{-1};
    const CsvReader imu_reader(imu_infile);
    for (const auto& row : imu_reader) {
      // time should be in seconds
      double stamp = row.at<double>(0);

      double ax = row.at<double>(1);
      double ay = row.at<double>(2);
      double az = row.at<double>(3);

      double wx = row.at<double>(4);
      double wy = row.at<double>(5);
      double wz = row.at<double>(6);

      ImuAccGyr imu_data;
      imu_data << ax, ay, az, wx, wy, wz;
      // imu_data << 0, 0, 9.8, 0, 0, 0;
      cached_imu_measurements_.push_back(ImuMeasurement(stamp, imu_data));
      imu_buffer.addMeasurement(stamp, imu_data);
    }
    LOG(INFO) << "Loaded IMU data";

    gtsam::Pose3 initial_pose = gtsam::Pose3::Identity();
    bool initial_frame_set = false;

    // first load flow files
    std::vector<std::filesystem::path> flow_files_in_directory =
        utils::getAllFilesInDir(flow_image_path);
    FrameId frame = 0;
    Timestamp previous_timestamp = 0;

    for (size_t i = 0; i < flow_files_in_directory.size(); i++) {
      std::filesystem::path flow_path(flow_files_in_directory.at(i));
      auto image_name = flow_path.stem().string();
      double timestamp_sec;
      // try getting the file name as a double timestamp
      //  this will be in nano-seconds and then should be converted to seconds
      //  to compare with the odom
      // file
      try {
        timestamp_sec = std::stod(image_name) / 1e9;

      } catch (const std::runtime_error& e) {
        LOG(FATAL) << "Failed to extract timestamp from image name "
                   << flow_path;
      }

      // try to get gt pose at nearest timestamp
      gtsam::Pose3 gt_pose;
      Timestamp gt_stamp;
      if (gt_odom_buffer.getNearestValueToTime(timestamp_sec, 0.003, &gt_pose,
                                               &gt_stamp)) {
        // LOG(INFO) << "Synchronised image timestamp " << std::fixed
        //           << timestamp_sec << " with gt stamp " << gt_stamp
        //           << ". Diff = " << std::abs(timestamp_sec - gt_stamp);

        // we have a sync!
        // Now try and get the other files that match this one
        const std::string left_image_file =
            left_image_path + image_name + ".png";
        const std::string right_image_file =
            right_image_path + image_name + ".png";
        const std::string mask_image_file =
            mask_image_path + image_name + ".png";

        utils::throwExceptionIfPathInvalid(left_image_file);
        utils::throwExceptionIfPathInvalid(right_image_file);
        utils::throwExceptionIfPathInvalid(mask_image_file);

        // convert to CV coordiantes
        const auto world_R_cv = gt_pose.rotation() * VIODE_left_hand_transform;
        const auto& world_t_cv = gt_pose.translation();  // same origin
        gt_pose = gtsam::Pose3(world_R_cv, world_t_cv);

        if (!initial_frame_set) {
          // expect very first pose (first frame, and first pose within that
          // frame) to be the first camera pose
          initial_pose = gt_pose;
          initial_frame_set = true;

          // add no measurements at first time
          imu_measrements_.insert2(frame, ImuMeasurements{});
        } else {
          CHECK_GE(previous_timestamp, 0);

          Timestamps imu_timestamps;
          ImuAccGyrs imu_measurements;
          auto imu_query_result = imu_buffer.getImuDataInterpolatedBorders(
              previous_timestamp, timestamp_sec, &imu_timestamps,
              &imu_measurements);

          if (imu_query_result ==
              ThreadsafeImuBuffer::QueryResult::kDataAvailable) {
            // LOG(INFO) << "Gotten imu data " << imu_timestamps.size()
            //           << " between " << previous_timestamp << " and "
            //           << timestamp_sec;

            ImuMeasurements imu_data(imu_timestamps, imu_measurements);
            imu_data.synchronised_frame_id = frame;
            imu_measrements_.insert2(frame, imu_data);
          }
        }

        // offset initial pose so we start at "0, 0, 0"
        gt_pose = initial_pose.inverse() * gt_pose;

        GroundTruthInputPacket gt_packet;
        gt_packet.frame_id_ = frame;
        gt_packet.timestamp_ = timestamp_sec;
        gt_packet.X_world_ = gt_pose;

        ground_truth_packets_.insert2(frame, gt_packet);
        frame++;

        // update sync paths
        image_left_paths_.push_back(left_image_file);
        image_right_paths_.push_back(right_image_file);
        masks_0_paths_.push_back(mask_image_file);
        flow_0_paths_.push_back(flow_path);

        times_.push_back(timestamp_sec);
        previous_timestamp = timestamp_sec;

      } else {
        LOG(INFO) << "Skipping frame " << timestamp_sec;
      }
    }

    dataset_size_ = flow_0_paths_.size();
  }

  void setSensorParams() {
    CameraParams::IntrinsicsCoeffs K({376.0, 376.0, 376.0, 240.0});
    CameraParams::DistortionCoeffs D({0, 0, 0, 0});

    gtsam::Matrix33 rot;
    rot << 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;

    gtsam::Pose3 extrinsics_left(gtsam::Rot3(rot),
                                 gtsam::Point3(0.0, 0.0, 0.0));

    camera_params_ = CameraParams(K, D, cv::Size(752, 480),
                                  DistortionModel::RADTAN, extrinsics_left);
    camera_params_.setDepthParams(0.05);

    // the original dataset actually has some rotation in it but we ignore this
    // as this is just the cv-to-robotic convention
    gtsam::Pose3 extrinsics_right(gtsam::Rot3(rot),
                                  gtsam::Point3(0.0, 0.05, 0));
    CameraParams right_camera_params(K, D, cv::Size(752, 480),
                                     DistortionModel::RADTAN, extrinsics_right);

    stereo_camera_ =
        std::make_shared<StereoCamera>(camera_params_, right_camera_params);

    DenseStereoParams dense_stereo_params;
    dense_stereo_params.use_sgbm_ = true;
    dense_stereo_params.post_filter_disparity_ = false;
    dense_stereo_params.median_blur_disparity_ = false;

    dense_stereo_params.pre_filter_cap_ = 31;

    dense_stereo_params.sad_window_size_ = 5;

    dense_stereo_params.min_disparity_ = 0;
    dense_stereo_params.num_disparities_ = 16 * 8;  // 128 disparities

    dense_stereo_params.uniqueness_ratio_ = 15;

    dense_stereo_params.speckle_range_ = 2;
    dense_stereo_params.speckle_window_size_ = 50;

    // BM parameters (ignored if using SGBM)
    dense_stereo_params.texture_threshold_ = 0;
    dense_stereo_params.pre_filter_type_ = cv::StereoBM::PREFILTER_XSOBEL;
    dense_stereo_params.pre_filter_size_ = 9;

    // SGBM parameters (compute from sad_window_size_)
    dense_stereo_params.p1_ = 8 * 1 * dense_stereo_params.sad_window_size_ *
                              dense_stereo_params.sad_window_size_;
    dense_stereo_params.p2_ = 32 * 1 * dense_stereo_params.sad_window_size_ *
                              dense_stereo_params.sad_window_size_;

    dense_stereo_params.disp_12_max_diff_ = 1;

    // Recommended: use MODE_SGBM_3WAY, so set this to false to avoid MODE_HH
    dense_stereo_params.use_mode_HH_ = true;
    // matching_params.sad_window_size_ = 4;
    // matching_params.num_disparities_ = 16 * 2;
    stereo_matcher_ = std::make_shared<StereoMatcher>(
        stereo_camera_, StereoMatchingParams{}, dense_stereo_params);

    imu_params_.acc_noise_density = 0.2;
    imu_params_.gyro_noise_density = 0.05;

    imu_params_.gyro_random_walk = 4.0e-5;
    imu_params_.acc_random_walk = 0.02;

    static const gtsam::Rot3 R_body_camera(
        (gtsam::Matrix3() << 0, 0, 1, 1, 0, 0, 0, 1, 0).finished());
    imu_params_.body_P_sensor =
        gtsam::Pose3(R_body_camera.inverse(), gtsam::Point3(0, 0, 0));

    imu_params_.imu_integration_sigma = 1e-3;
    imu_params_.n_gravity = gtsam::Point3(0, 9.8, 0);
  }

 public:
  size_t dataset_size_;

  std::vector<std::string> image_left_paths_;
  std::vector<std::string> image_right_paths_;

  std::vector<std::string> flow_0_paths_;
  std::vector<std::string> masks_0_paths_;

  std::vector<Timestamp> times_;

  // frame k to imu measurements where the measurements should be from k-1 to k
  gtsam::FastMap<FrameId, ImuMeasurements> imu_measrements_;

  std::vector<ImuMeasurement> cached_imu_measurements_;
  bool imu_measurements_sent{false};

  GroundTruthPacketMap ground_truth_packets_;
  // left camera params
  CameraParams camera_params_;

  ImuParams imu_params_;

  StereoCamera::Ptr stereo_camera_;
  StereoMatcher::Ptr stereo_matcher_;
};

struct ViodeTimestampLoader : public TimestampBaseLoader {
  ViodeAllLoader::Ptr loader_;

  ViodeTimestampLoader(ViodeAllLoader::Ptr loader)
      : loader_(CHECK_NOTNULL(loader)) {}
  std::string getFolderName() const override { return ""; }

  size_t size() const override { return loader_->size(); }

  double getItem(size_t idx) override { return loader_->getTimestamp(idx); }
};

ViodeLoader::ViodeLoader(const fs::path& dataset_path)
    : ViodeProvider(dataset_path) {
  LOG(INFO) << "Starting VIODE Loader with path" << dataset_path;

  // this would go out of scope but we capture it in the functional loaders
  auto loader = std::make_shared<ViodeAllLoader>(dataset_path);
  auto timestamp_loader = std::make_shared<ViodeTimestampLoader>(loader);

  left_camera_params_ = loader->getLeftCameraParams();
  CHECK(getCameraParams());

  imu_params_ = loader->imu_params_;
  CHECK(getImuParams());

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

  auto imu_loader =
      std::make_shared<FunctionalDataFolder<std::optional<ImuMeasurements>>>(
          [loader](size_t idx) { return loader->getImuMeasurements(idx); });

  auto rgb_right_loader =
      std::make_shared<FunctionalDataFolder<std::optional<cv::Mat>>>(
          [loader](size_t idx) { return loader->getRightRGB(idx); });

  this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader,
                   depth_loader, instance_mask_loader, gt_loader, imu_loader,
                   rgb_right_loader);

  std::vector<ImuMeasurement>& cached_imu_measurements =
      loader->cached_imu_measurements_;
  auto& imu_measurements_sent = loader->imu_measurements_sent;

  auto callback = [&](size_t frame_id, Timestamp timestamp, cv::Mat rgb,
                      cv::Mat optical_flow, cv::Mat depth,
                      cv::Mat instance_mask,
                      GroundTruthInputPacket gt_object_pose_gt,
                      std::optional<ImuMeasurements> imu_measurements,
                      std::optional<cv::Mat> right_rgb) -> bool {
    CHECK_EQ(timestamp, gt_object_pose_gt.timestamp_);

    CHECK(ground_truth_packet_callback_);
    if (ground_truth_packet_callback_)
      ground_truth_packet_callback_(gt_object_pose_gt);

    if (cached_imu_measurements.size() > 0 && !imu_measurements_sent) {
      if (!imu_single_input_callback_) {
        LOG(WARNING) << "imu_single_input_callback_ has not been registered!! "
                        "Skipping IMU data...";

      } else {
        for (const auto& imu_measurement : cached_imu_measurements) {
          imu_single_input_callback_(imu_measurement);
        }
      }
      imu_measurements_sent = true;
    }

    // if (imu_multi_input_callback_ && imu_measurements)
    //   imu_multi_input_callback_(imu_measurements.value());

    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
        .depth(depth)
        .opticalFlow(optical_flow)
        .objectMotionMask(instance_mask);

    if (right_rgb) image_container.rightRgb(right_rgb.value());

    CHECK(image_container_callback_);
    if (image_container_callback_)
      image_container_callback_(
          std::make_shared<ImageContainer>(image_container));
    return true;
  };

  this->setCallback(callback);
}

}  // namespace dyno
