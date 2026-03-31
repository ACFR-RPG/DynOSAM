#include "dynosam/dataprovider/DynopetsDataProvider.hpp"

#include "dynosam_common/utils/CsvParser.hpp"

namespace dyno {

class Loader {
 private:
  size_t dataset_size_;
  std::vector<std::filesystem::path> rgb_image_paths_;
  std::vector<std::filesystem::path> depth_image_paths_;
  std::vector<std::filesystem::path> mask_image_paths_;
  std::vector<GroundTruthInputPacket> ground_truth_packets_;
  std::vector<Timestamp> times_;
  CameraParams camera_params_;

 public:
  DYNO_POINTER_TYPEDEFS(Loader)

  Loader(const fs::path& dataset_path) {
    loadFilePaths(dataset_path);
    loadIntrinsics(dataset_path);
    loadGroundTruth(dataset_path);
  };

  cv::Mat getRGB(size_t idx) const {
    CHECK_LT(idx, rgb_image_paths_.size());
    cv::Mat rgb;
    utils::loadRGB((std::string)rgb_image_paths_.at(idx), rgb);
    CHECK(!rgb.empty());
    return rgb;
  }

  cv::Mat getDepth(size_t idx) const {
    CHECK_LT(idx, depth_image_paths_.size());
    CHECK_LT(idx, dataset_size_);

    cv::Mat depth;
    utils::loadDepth(depth_image_paths_.at(idx), depth);
    CHECK(!depth.empty());

    depth = depth / 1000.0;  // convert to meters immediately

    return depth;
  }

  cv::Mat getMask(size_t idx) const {
    CHECK_LT(idx, mask_image_paths_.size());
    cv::Mat mask = cv::imread((std::string)mask_image_paths_.at(idx),
                              cv::IMREAD_GRAYSCALE);

    CHECK(!mask.empty());

    // Ensure binary (optional but safer)
    cv::Mat binary;
    cv::threshold(mask, binary, 127, 255, cv::THRESH_BINARY);

    // Invert + normalize:
    // Original: background=255, object=0
    // After:    background=0,   object=1
    cv::Mat inverted;
    cv::bitwise_not(binary, inverted);  // 255->0, 0->255

    cv::Mat mask01;
    inverted /= 255;  // now: 0 or 1 (still uint8)

    // Convert to CV_32SC1
    cv::Mat mask32s;
    inverted.convertTo(mask32s, CV_32SC1);

    return mask32s;
  }

  const GroundTruthInputPacket& getGtPacket(size_t idx) const {
    CHECK_LT(idx, dataset_size_);
    return ground_truth_packets_.at(idx);
  }

  const CameraParams& getLeftCameraParams() const { return camera_params_; }

  size_t size() const { return dataset_size_; }

  double getTimestamp(size_t idx) {
    CHECK_LT(idx, dataset_size_);
    return times_.at(idx);
  }

 private:
  void loadFilePaths(const std::string& file_path) {
    const auto rgb_image_path = file_path + "/color";
    const auto depth_image_path = file_path + "/depth";
    const auto mask_image_path = file_path + "/mask";

    utils::throwExceptionIfPathInvalid(rgb_image_path);
    utils::throwExceptionIfPathInvalid(depth_image_path);
    utils::throwExceptionIfPathInvalid(mask_image_path);

    rgb_image_paths_ = utils::getAllFilesInDir(rgb_image_path);

    depth_image_paths_ = utils::getAllFilesInDir(depth_image_path);

    mask_image_paths_ = utils::getAllFilesInDir(mask_image_path);

    CHECK_EQ(rgb_image_paths_.size(), depth_image_paths_.size());
    CHECK_EQ(mask_image_paths_.size(), depth_image_paths_.size());

    dataset_size_ = rgb_image_paths_.size();
  }

  void loadIntrinsics(const std::string& file_path) {
    const auto intrinsics_path = file_path + "/intrinsics.txt";
    utils::throwExceptionIfPathInvalid(intrinsics_path);

    std::ifstream file(intrinsics_path);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + intrinsics_path);
    }

    std::vector<double> vals;
    double v;
    while (file >> v) {  // read space-separated numbers
      vals.push_back(v);
    }

    if (vals.size() < 8) {
      throw std::runtime_error("File does not contain enough values: " +
                               container_to_string(vals));
    }

    double fx = vals[0];
    double fy = vals[1];
    double cx = vals[2];
    double cy = vals[3];
    double skew = vals[4];

    int H = static_cast<int>(vals[6]);
    int W = static_cast<int>(vals[7]);

    CameraParams::IntrinsicsCoeffs K({fx, fy, cx, cy});
    CameraParams::DistortionCoeffs D({0, 0, 0, 0});

    camera_params_ =
        CameraParams(K, D, cv::Size(W, H), DistortionModel::RADTAN);
  }

  void loadGroundTruth(const std::string& file_path) {
    const auto camera_poses_file_path = file_path + "/camera_poses.csv";
    utils::throwExceptionIfPathInvalid(camera_poses_file_path);

    const auto object_poses_file_path = file_path + "/object_poses.csv";
    utils::throwExceptionIfPathInvalid(object_poses_file_path);

    std::ifstream cam_pose_file(camera_poses_file_path);
    if (!cam_pose_file.is_open()) {
      throw std::runtime_error("Cannot open file: " + camera_poses_file_path);
    }

    std::ifstream object_pose_file(object_poses_file_path);
    if (!object_pose_file.is_open()) {
      throw std::runtime_error("Cannot open file: " + object_poses_file_path);
    }

    using PoseTimestampVector = std::vector<std::pair<Timestamp, gtsam::Pose3>>;

    auto load_from_csv = [](CsvReader& csv) -> PoseTimestampVector {
      PoseTimestampVector values;

      // skip header row
      auto it = csv.begin();
      it++;

      for (; it != csv.end(); it++) {
        const CsvReader::Row& row = *it;
        CHECK_EQ(row.size(), 8);

        Timestamp timestamp = row.at<Timestamp>(0);

        double tx = row.at<double>(1);
        double ty = row.at<double>(2);
        double tz = row.at<double>(3);

        double qx = row.at<double>(4);
        double qy = row.at<double>(5);
        double qz = row.at<double>(6);
        double qw = row.at<double>(7);

        gtsam::Point3 t(tx, ty, tz);
        gtsam::Rot3 rot(qw, qx, qy, qz);
        values.push_back(std::make_pair(timestamp, gtsam::Pose3(rot, t)));
      }
      return values;
    };

    CsvReader camera_pose_csv(cam_pose_file);
    CsvReader object_pose_csv(object_pose_file);

    auto camera_poses = load_from_csv(camera_pose_csv);
    auto object_poses = load_from_csv(object_pose_csv);

    // dataset should already be set so call function after loadFilePaths
    CHECK_EQ(camera_poses.size(), dataset_size_);
    CHECK_EQ(object_poses.size(), dataset_size_);

    gtsam::Pose3 initial_camera_pose;
    bool initial_pose_set = false;

    ground_truth_packets_.reserve(dataset_size_);
    times_.reserve(dataset_size_);
    for (size_t frame_id = 0; frame_id < camera_poses.size(); frame_id++) {
      auto [timestamp_c, X_W_k_original] = camera_poses.at(frame_id);
      auto [timestamp_o, L_W_k_original] = object_poses.at(frame_id);

      CHECK_EQ(timestamp_c, timestamp_o);

      if (!initial_pose_set) {
        initial_camera_pose = X_W_k_original;
        initial_pose_set = true;
      }

      gtsam::Pose3 X_W_k = initial_camera_pose.inverse() * X_W_k_original;
      // L in camera frame
      gtsam::Pose3 L_X_k = X_W_k_original.inverse() * L_W_k_original;
      // ground truth L in world accounting for camera offset
      gtsam::Pose3 L_W_k = X_W_k * L_X_k;

      GroundTruthInputPacket gt_packet;
      gt_packet.frame_id_ = frame_id;
      gt_packet.timestamp_ = timestamp_c;
      gt_packet.X_world_ = X_W_k;

      ObjectPoseGT object_gt;
      object_gt.frame_id_ = frame_id;
      object_gt.object_id_ = 1;
      object_gt.L_camera_ = L_X_k;
      object_gt.L_world_ = L_W_k;

      gt_packet.object_poses_.push_back(std::move(object_gt));

      // set motions
      if (frame_id > 0) {
        const GroundTruthInputPacket& previous_gt_packet =
            ground_truth_packets_.at(frame_id - 1);
        gt_packet.calculateAndSetMotions(previous_gt_packet);
      }

      ground_truth_packets_.push_back(gt_packet);
      times_.push_back(timestamp_c);
    }
  }
};

struct TimestampLoader : public TimestampBaseLoader {
  Loader::Ptr loader_;

  TimestampLoader(Loader::Ptr loader) : loader_(CHECK_NOTNULL(loader)) {}
  std::string getFolderName() const override { return ""; }

  size_t size() const override { return loader_->size(); }

  double getItem(size_t idx) override { return loader_->getTimestamp(idx); }
};

DynopetsLoader::DynopetsLoader(const fs::path& dataset_path)
    : DynoeptsProvider(dataset_path) {
  LOG(INFO) << "Starting DynopetsLoader Loader with path" << dataset_path;

  // this would go out of scope but we capture it in the functional loaders
  auto loader = std::make_shared<Loader>(dataset_path);
  auto timestamp_loader = std::make_shared<TimestampLoader>(loader);

  left_camera_params_ = loader->getLeftCameraParams();
  CHECK(getCameraParams());

  auto rgb_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getRGB(idx); });

  auto optical_flow_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t) { return cv::Mat(); });

  auto depth_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getDepth(idx); });

  auto instance_mask_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getMask(idx); });

  auto gt_loader =
      std::make_shared<FunctionalDataFolder<GroundTruthInputPacket>>(
          [loader](size_t idx) { return loader->getGtPacket(idx); });

  this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader,
                   depth_loader, instance_mask_loader, gt_loader);

  auto callback = [&](size_t frame_id, Timestamp timestamp, cv::Mat rgb,
                      cv::Mat, cv::Mat depth, cv::Mat mask,
                      GroundTruthInputPacket gt_object_pose_gt) -> bool {
    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb).depth(depth).objectMotionMask(mask);

    if (ground_truth_packet_callback_)
      ground_truth_packet_callback_(gt_object_pose_gt);

    if (image_container_callback_)
      image_container_callback_(
          std::make_shared<ImageContainer>(image_container));
    return true;
  };

  this->setCallback(callback);
}

}  // namespace dyno
