#include "dynosam/dataprovider/DynoeptsDataProvider.hpp"

namespace dyno {

class Loader {
 private:
  size_t dataset_size_;
  std::vector<std::filesystem::path> rgb_image_paths_;
  std::vector<std::filesystem::path> depth_image_paths_;
  std::vector<std::filesystem::path> mask_image_paths_;
  GroundTruthPacketMap ground_truth_packets_;
  std::vector<Timestamp> times_;
  CameraParams camera_params_;

 public:
  DYNO_POINTER_TYPEDEFS(Loader)

  Loader(const fs::path& dataset_path) {
    loadFilePaths(dataset_path);
    loadIntrinsics(dataset_path);
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
    return ground_truth_packets_.at(idx);
  }

  const CameraParams& getLeftCameraParams() const { return camera_params_; }

  size_t size() const { return dataset_size_; }

  double getTimestamp(size_t idx) {
    // for now just return idx
    return idx;
    // CHECK_LT(idx, rgb_image_paths_.size());
    // CHECK_LT(idx, dataset_size_);
    // return times_.at(idx);
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
};

struct TimestampLoader : public TimestampBaseLoader {
  Loader::Ptr loader_;

  TimestampLoader(Loader::Ptr loader) : loader_(CHECK_NOTNULL(loader)) {}
  std::string getFolderName() const override { return ""; }

  size_t size() const override { return loader_->size(); }

  double getItem(size_t idx) override { return loader_->getTimestamp(idx); }
};

DynoeptsLoader::DynoeptsLoader(const fs::path& dataset_path)
    : DynoeptsProvider(dataset_path) {
  LOG(INFO) << "Starting DynoeptsLoader Loader with path" << dataset_path;

  // this would go out of scope but we capture it in the functional loaders
  auto loader = std::make_shared<Loader>(dataset_path);
  auto timestamp_loader = std::make_shared<TimestampLoader>(loader);

  left_camera_params_ = loader->getLeftCameraParams();
  CHECK(getCameraParams());

  auto rgb_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getRGB(idx); });

  auto optical_flow_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return cv::Mat(); });

  auto depth_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getDepth(idx); });

  auto instance_mask_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getMask(idx); });

  auto gt_loader =
      std::make_shared<FunctionalDataFolder<GroundTruthInputPacket>>(
          [loader](size_t idx) { return GroundTruthInputPacket{}; });

  this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader,
                   depth_loader, instance_mask_loader, gt_loader);

  auto callback = [&](size_t frame_id, Timestamp timestamp, cv::Mat rgb,
                      cv::Mat, cv::Mat depth, cv::Mat mask,
                      GroundTruthInputPacket gt_object_pose_gt) -> bool {
    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb).depth(depth).objectMotionMask(mask);

    if (image_container_callback_)
      image_container_callback_(
          std::make_shared<ImageContainer>(image_container));
    return true;
  };

  this->setCallback(callback);
}

}  // namespace dyno
