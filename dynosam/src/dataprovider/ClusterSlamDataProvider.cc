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

#include "dynosam/dataprovider/ClusterSlamDataProvider.hpp"

#include <glog/logging.h>

#include <fstream>
#include <png++/png.hpp>  //libpng-dev

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/frontend/vision/StereoMatcher.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"  //for getObjectLabels
#include "dynosam_common/Algorithms.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_cv/StereoCamera.hpp"

namespace dyno {

/**
 * @brief As there is so much interdependancy between the folders in the
 * dataset, we just have the one loader so all the data can be accessed in the
 * same class. This class then provides interfaces for each
 * dyno::DataFolder<....> so that the Dynodataset can use it!
 *
 */
class ClusterSlamAllLoader {
 public:
  DYNO_POINTER_TYPEDEFS(ClusterSlamAllLoader)

  /**
   * @brief Construct a new Cluster Slam Gt Loader object.
   *
   * File path is the full file path to a clusterslam folder, containing
   * /images
   * /pose....
   *
   * @param file_path const std::string&
   */
  ClusterSlamAllLoader(const std::string& file_path)
      : left_images_folder_path_(file_path + "/images/left"),
        right_images_folder_path_(file_path + "/images/right"),
        optical_flow_folder_path_(file_path + "/optical_flow"),
        pose_folder_path_(file_path + "/pose"),
        instance_masks_folder_(file_path + "/instance_masks"),
        left_landmarks_folder_(file_path + "/landmarks/left"),
        landmark_mapping_file_path_(file_path + "/landmark_mapping.txt"),
        intrinsics_file_path_(file_path + "/intrinsic.txt") {
    utils::throwExceptionIfPathInvalid(left_images_folder_path_);
    utils::throwExceptionIfPathInvalid(right_images_folder_path_);
    utils::throwExceptionIfPathInvalid(optical_flow_folder_path_);
    utils::throwExceptionIfPathInvalid(pose_folder_path_);
    utils::throwExceptionIfPathInvalid(instance_masks_folder_);
    utils::throwExceptionIfPathInvalid(left_landmarks_folder_);
    utils::throwExceptionIfPathInvalid(landmark_mapping_file_path_);
    utils::throwExceptionIfPathInvalid(intrinsics_file_path_);

    // first load images and size
    // the size will be used as a refernce for all other loaders
    // size is number of images
    // we use flow to set the size as for this dataset, there will be one less
    // optical flow as the flow ids index t to t+1 (which is gross, I know!!)
    loadFlowImagesAndSize(optical_flow_image_paths_, dataset_size_);
    // left
    loadImages(left_rgb_image_paths_, left_images_folder_path_);
    // right
    loadImages(right_rgb_image_paths_, right_images_folder_path_);
    CHECK_EQ(right_rgb_image_paths_.size(), left_rgb_image_paths_.size());

    loadInstanceMasksImages(instance_masks_image_paths_);
    loadLeftLandmarks(left_landmarks_map_);

    loadLandMarkMapping(landmark_mapping_);

    CHECK_EQ(optical_flow_image_paths_.size(),
             left_rgb_image_paths_.size() - 1);
    CHECK_EQ(instance_masks_image_paths_.size(),
             optical_flow_image_paths_.size());
    CHECK_EQ(size(), optical_flow_image_paths_.size());
    setGroundTruthPacket(ground_truth_packets_);

    setIntrisics();
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
    CHECK_LT(idx, left_rgb_image_paths_.size());

    cv::Mat rgb;
    utils::loadRGB(left_rgb_image_paths_.at(idx), rgb);
    CHECK(!rgb.empty());

    return rgb;
  }

  cv::Mat getRightRGB(size_t idx) const {
    cv::Mat rgb_right;
    CHECK_LT(idx, right_rgb_image_paths_.size());

    utils::loadRGB(right_rgb_image_paths_.at(idx), rgb_right);
    CHECK(!rgb_right.empty());

    return rgb_right;
  }

  cv::Mat getInstanceMask(size_t idx) const {
    CHECK_LT(idx, left_rgb_image_paths_.size());
    CHECK_LT(idx, dataset_size_);

    cv::Mat mask, relabelled_mask;
    utils::loadMask(instance_masks_image_paths_.at(idx), mask);
    CHECK(!mask.empty());

    associateDetectedBBWithObject(mask, idx, relabelled_mask);

    return relabelled_mask;
  }

  cv::Mat getDepthImage(size_t idx) const {
    return denseStereoReconstruction(idx);
  }

  const GroundTruthInputPacket& getGtPacket(size_t idx) const {
    return ground_truth_packets_.at(idx);
  }

  const CameraParams& getLeftCameraParams() const {
    return left_camera_params_;
  }

  double getTimestamp(size_t idx) {
    // we dont have a time? for now just make idx
    return static_cast<double>(idx);
  }

 private:
  void loadFlowImagesAndSize(std::vector<std::string>& images_paths,
                             size_t& dataset_size) {
    std::vector<std::filesystem::path> files_in_directory =
        utils::getAllFilesInDir(optical_flow_folder_path_);
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

    for (const std::string file_path : files_in_directory) {
      utils::throwExceptionIfPathInvalid(file_path);
      images_paths.push_back(file_path);
    }
  }

  void loadInstanceMasksImages(std::vector<std::string>& images_paths) {
    std::vector<std::filesystem::path> files_in_directory =
        utils::getAllFilesInDir(instance_masks_folder_);

    for (const std::string file_path : files_in_directory) {
      utils::throwExceptionIfPathInvalid(file_path);
      images_paths.push_back(file_path);
    }
  }

  // set all intrinsics/camera parameters related variables include stereo
  // camera/matcher etc...
  void setIntrisics() {
    std::ifstream fstream;
    fstream.open(intrinsics_file_path_, std::ios::in);
    if (!fstream) {
      throw std::runtime_error("Could not open file " + intrinsics_file_path_ +
                               " when trying to load Cluster Slam intrinsics!");
    }

    auto load_projection_matrix =
        [](std::ifstream& fstream, gtsam::Matrix34& projection_matrix) -> void {
      // read the first 3 liens of the projection matrix and populate camera
      // matrix
      for (size_t i = 0; i < 3; i++) {
        // std::vector<std::string> split_line;
        // hmmm my getline function does not seem to work here (maybe because
        // doubles so we'll do it manually!)
        // CHECK(getLine(fstream, split_line));
        // CHECK_EQ(split_line.size(), 4) << "Line was: " <<
        // container_to_string(split_line, ", ");

        std::string s;
        std::getline(fstream, s);

        std::stringstream ss;
        ss << s;

        LOG(INFO) << ss.str();
        for (size_t j = 0; j < 4; j++) {
          double tmp;
          ss >> tmp;
          projection_matrix(i, j) = tmp;
        }
      }
    };  // load_projection_matrix

    // intrinscis file is
    // 3 x 4 projection matrix, over 3 lines
    //  new line
    // 3 x 4 projection matrix, over 3 lines
    // populate camera 1 matrix
    load_projection_matrix(fstream, projection_matrix_cam1_);

    // read the line line
    std::string s;
    std::getline(fstream, s);
    // populate camera 2 matrix
    load_projection_matrix(fstream, projection_matrix_cam2_);

    // set K matrices
    K_cam_1 = projection_matrix_cam1_.topLeftCorner(3, 3);
    K_cam_2 = projection_matrix_cam2_.topLeftCorner(3, 3);

    double fx_left = K_cam_1(0, 0);
    double fy_left = K_cam_1(1, 1);
    double cu_left = K_cam_1(0, 2);
    double cv_left = K_cam_1(1, 2);

    double fx_right = K_cam_2(0, 0);
    double fy_right = K_cam_2(1, 1);
    double cu_right = K_cam_2(0, 2);
    double cv_right = K_cam_2(1, 2);

    // Projection_matrix = K * extrinsics
    gtsam::Pose3 extrinsics_left =
        gtsam::Pose3(K_cam_1.inverse() * projection_matrix_cam1_);
    gtsam::Pose3 extrinsics_right =
        gtsam::Pose3(K_cam_2.inverse() * projection_matrix_cam2_);

    // I think this dataset has the projection matrix in the local left frame
    // (i.e projection of 2 into 1) becuase the translation component of
    // extrinsics_right is negative we want extrinsice to be in the world frame
    // because this is what the camera params expect OR the dataset is in OpenCV
    // convention...
    extrinsics_right = extrinsics_right.inverse();

    LOG(INFO) << "left e " << extrinsics_left;
    LOG(INFO) << "right e " << extrinsics_right;

    left_camera_params_ = CameraParams(
        CameraParams::IntrinsicsCoeffs({fx_left, fy_left, cu_left, cv_left}),
        CameraParams::DistortionCoeffs({0, 0, 0, 0}), getRGB(0).size(),
        DistortionModel::RADTAN, extrinsics_left);

    CameraParams right_camera_params(
        CameraParams::IntrinsicsCoeffs(
            {fx_right, fy_right, cu_right, cv_right}),
        CameraParams::DistortionCoeffs({0, 0, 0, 0}), getRGB(0).size(),
        DistortionModel::RADTAN, extrinsics_right);

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

    stereo_camera_ = std::make_shared<StereoCamera>(left_camera_params_,
                                                    right_camera_params);
    stereo_matcher_ = std::make_shared<StereoMatcher>(
        stereo_camera_, StereoMatchingParams{}, dense_stereo_params);
  }

  cv::Mat denseStereoReconstruction(size_t frame) const {
    cv::Mat rgb_left = getRGB(frame);
    cv::Mat rgb_right = getRightRGB(frame);

    // this set of images are loaded as 8UC4
    CHECK_EQ(rgb_right.type(), CV_8UC4)
        << "Somehow the image type has changed...";
    cv::Mat depth_image;
    CHECK_NOTNULL(stereo_matcher_)
        ->denseStereoReconstruction(rgb_left, rgb_right, depth_image);

    return depth_image;
  }

  // mapping of landmark id -> u, v coordiante
  using LandmarksMap = gtsam::FastMap<int, Keypoint>;
  using LandmarksMapPerFrame = gtsam::FastMap<FrameId, LandmarksMap>;

  void loadLeftLandmarks(LandmarksMapPerFrame& detected_landmarks) {
    std::vector<std::filesystem::path> files_in_directory =
        utils::getAllFilesInDir(left_landmarks_folder_);

    for (const std::filesystem::path& file_path : files_in_directory) {
      // this should be the file name which we will then decompose into the
      // frame id e.g. .../pose/0000.txt -> filename = 0000 -> frame id = int(0)
      const std::string file_name = file_path.filename();
      const int frame = std::stoi(file_name);

      std::ifstream fstream;
      fstream.open((std::string)file_path, std::ios::in);
      if (!fstream) {
        throw std::runtime_error(
            "Could not open file " + (std::string)file_path +
            " when trying to load Cluster Slam landmarks information!");
      }

      // check we've never processed this frame before
      CHECK(!detected_landmarks.exists(frame));

      LandmarksMap detected_lmks_this_frame;

      // expected size of each line (landamrk_id u v)
      constexpr static size_t kExpectedSize = 3u;
      while (!fstream.eof()) {
        std::vector<std::string> split_line;
        if (!utils::getLine(fstream, split_line)) continue;

        CHECK_EQ(split_line.size(), kExpectedSize)
            << "Line present in landmarks file does not have the right length "
               "- "
            << container_to_string(split_line);

        const int landmark_id = std::stoi(split_line.at(0));
        const double u = std::stod(split_line.at(1));
        const double v = std::stod(split_line.at(2));

        detected_lmks_this_frame.insert2(landmark_id, Keypoint(u, v));
      }

      detected_landmarks.insert2(frame, detected_lmks_this_frame);
    }

    VLOG(20) << "Loaded " << detected_landmarks.size()
             << " detected landmarks...";
  }

  // Landmark to cluster id mapping.
  // the landmark id is the key in LandmarksMap
  void loadLandMarkMapping(gtsam::FastMap<int, int>& landmark_mapping) {
    std::ifstream fstream;
    fstream.open(landmark_mapping_file_path_, std::ios::in);
    if (!fstream) {
      throw std::runtime_error(
          "Could not open file " + landmark_mapping_file_path_ +
          " when trying to load Cluster Slam landmarks mapping!");
    }

    // expected size of each line (landamrk_id cluster_id)
    constexpr static size_t kExpectedSize = 2u;
    while (!fstream.eof()) {
      std::vector<std::string> split_line;
      if (!utils::getLine(fstream, split_line)) continue;

      CHECK_EQ(split_line.size(), kExpectedSize)
          << "Line present in landmark mapping file does not have the right "
             "length - "
          << container_to_string(split_line);

      const int landmark_id = std::stoi(split_line.at(0));
      // clusters start at 0 here, but that includes background label (which is
      // 0) since the  line ordering of the trajectories files determines the
      // label the camera pose is always first (line = 0), so any cluster that
      // is associated to an object and not the camera pose, should have idx > 0
      const int cluster_id = std::stoi(split_line.at(1));

      landmark_mapping.insert2(landmark_id, cluster_id);
    }
  }

  // we're going to have to associate the detected bounding boxes
  // with the actual landmark ids so that the gt labels match with the
  // tracking labels....
  // we will have to do this by taking all the landmarks of detected features
  // and check which objects lies within the bounding box....
  // this is almost the definitions of heinous but like, whatever....
  // this function updates
  // returns the new instance mask
  // from the original mask, relabel the pixel values using the gt tracks
  void associateDetectedBBWithObject(const cv::Mat& instance_mask,
                                     FrameId frame_id,
                                     cv::Mat& relabelled_mask) const {
    const GroundTruthInputPacket& gt_packet = getGtPacket(frame_id);
    ObjectIds object_ids = vision_tools::getObjectLabels(instance_mask);
    const size_t n = object_ids.size();

    instance_mask.copyTo(relabelled_mask);

    if (n == 0) {
      return;
    }

    // sort kps map into a map of clusters -> [kps....]
    const LandmarksMap& kps_map = left_landmarks_map_.at(frame_id);
    // cluster id (track id -> all keypoints for that cluster)
    std::set<int> cluster_id_set;
    gtsam::FastMap<int, Keypoints> clustered_kps;
    for (const auto& [landmark_id, kp] : kps_map) {
      const auto cluster_id = landmark_mapping_.at(landmark_id);
      cluster_id_set.insert(cluster_id);
      if (!clustered_kps.exists(cluster_id)) {
        clustered_kps.insert2(cluster_id, Keypoints());
      }
      clustered_kps.at(cluster_id).push_back(kp);
    }

    std::vector<int> cluster_ids(cluster_id_set.begin(), cluster_id_set.end());

    // want to assign object ids to clusters
    const size_t m = cluster_ids.size();
    CHECK_EQ(m, clustered_kps.size());

    LOG(INFO) << n << " " << m;

    Eigen::MatrixXd cost;
    cost.resize(n, m);

    for (size_t rows = 0; rows < n; rows++) {
      ObjectId object_id = object_ids.at(rows);

      // get the binary mask just for this object, we will use this too find a
      // bounding box
      // and count how many kp's fall into this mask
      const cv::Mat object_mask = relabelled_mask == object_id;

      cv::Rect bb;
      if (!vision_tools::findObjectBoundingBox(instance_mask, object_id, bb)) {
        VLOG(10) << "No bb could be found for detected object " << object_id
                 << " at frame " << frame_id
                 << ". Skipping association and removing object from mask";
        relabelled_mask.setTo(cv::Scalar(0), object_mask);
        continue;
      }

      for (size_t cols = 0; cols < m; cols++) {
        int cluster_id = cluster_ids.at(cols);
        const Keypoints& kps = clustered_kps.at(cluster_id);

        //(inverse) cost for the object id with this cluster id
        // hack - make really small number so we divide by zero!!
        double object_count = 0.000001;
        for (const auto& kp : kps) {
          cv::Point2f pt(utils::gtsamPointToCv(kp));
          if (bb.contains(pt)) {
            object_count++;
          }
        }
        cost(rows, cols) = object_count;
      }
    }

    // apply scaling to the costs to turn the cost function from an argmax to an
    // argmin problem which the hungrian problem sovles ie we want the
    // assignment that maximuses the number of kp's in the object insancec
    Eigen::MatrixXd loged_costs =
        cost.unaryExpr([](double x) { return 1.0 / x * 10; });
    Eigen::VectorXi assignment;
    internal::HungarianAlgorithm().solve(loged_costs, assignment);

    // for(size_t i = 0; i < assignment.size(); i++) {
    //     int j = assignment[i];

    //     //the i-th object is assignment to the j-th cluster
    //     ObjectId object_id = object_ids.at(i);
    //     int assigned_custer = cluster_ids.at(j);

    //     const cv::Mat object_mask = relabelled_mask == object_id;
    //     relabelled_mask.setTo(cv::Scalar(assigned_custer), object_mask);
    // }
    ObjectIds old_labels = object_ids;
    ObjectIds new_labels;
    for (size_t i = 0; i < assignment.size(); i++) {
      int j = assignment[i];

      // //the i-th object is assignment to the j-th cluster
      // ObjectId instance_id = instance_ids.at(i);
      int assigned_custer = cluster_ids.at(j);
      new_labels.push_back(assigned_custer);

      // LOG(INFO) << "Relabelled " << instance_id << " to " <<
      // assigned_object_id;

      // const cv::Mat object_mask = relabelled_mask == instance_id;
      // relabelled_mask.setTo(cv::Scalar(assigned_object_id), object_mask);
    }

    vision_tools::relabelMasks(instance_mask, relabelled_mask, old_labels,
                               new_labels);
  }

  void setGroundTruthPacket(GroundTruthPacketMap& ground_truth_packets) {
    // load the poses
    // first line contains camera pose
    // consequative lines contain object ids which are indexed from 0, but we
    // will index from 1 expect the file name to give us the frame we are
    // working with!!
    std::vector<std::filesystem::path> files_in_directory =
        utils::getAllFilesInDir(pose_folder_path_);

    gtsam::Pose3 initial_pose = gtsam::Pose3::Identity();
    bool initial_frame_set = false;

    Eigen::Matrix3d R_carla_to_opencv;
    R_carla_to_opencv << 0, 0, 1, 1, 0, 0, 0, -1, 0;
    gtsam::Rot3 rot_carla_to_opencv(R_carla_to_opencv);
    gtsam::Pose3 T_carla_to_opencv(rot_carla_to_opencv, gtsam::Point3(0, 0, 0));

    for (const std::filesystem::path& pose_file_path : files_in_directory) {
      // this should be the file name which we will then decompose into the
      // frame id e.g. .../pose/0000.txt -> filename = 0000 -> frame id = int(0)
      const std::string file_name = pose_file_path.filename();
      const int frame = std::stoi(file_name);
      const gtsam::Pose3Vector poses =
          processPoseFile(pose_file_path, ground_truth_packets);

      // check we've never processed this frame before
      CHECK(!ground_truth_packets.exists(frame));

      if (frame > 0) {
        // check we have the previous frame!! (ie.e we process in order!)
        CHECK(ground_truth_packets.exists(frame - 1));
      }

      GroundTruthInputPacket gt_packet;
      gt_packet.frame_id_ = frame;
      gt_packet.timestamp_ = getTimestamp(frame);
      // Quoting the dataset home page:
      // For each trajectory file under pose/ directory, each line in each file
      // represents the pose of each cluster except for the first line - that
      // line represents the camera pose.
      gtsam::Pose3 original_camera_pose;
      gtsam::Pose3 aligned_camera_pose;
      for (size_t i = 0; i < poses.size(); i++) {
        // pose is in world coordinate convention so we put into camera
        // convention (the camera_to_world.inverse())
        //  gtsam::Pose3 pose = camera_to_world.inverse() * poses.at(i);
        gtsam::Pose3 pose = poses.at(i);
        // pose = T_carla_to_opencv.inverse() * pose * T_carla_to_opencv;
        // pose = T_carla_to_opencv *
        // gtsam::Pose3(toRightHandedRotation(pose.rotation()),
        // toRightHandedVector(pose.translation())); pose =
        // gtsam::Pose3(toRightHandedRotation(pose.rotation()),
        // toRightHandedVector(pose.translation())); pose =
        // gtsam::Pose3(rot_carla_to_opencv * pose.rotation(),
        // pose.translation()); translation(1) = -translation(1); pose =
        // gtsam::Pose3(rot_carla_to_opencv * pose.rotation(),
        // rot_carla_to_opencv * translation);

        // LOG(INFO) << pose;

        // pose = camera_to_world.inverse() * pose;

        // LOG(INFO) << pose;

        if (i == 0) {
          if (!initial_frame_set) {
            // expect very first pose (first frame, and first pose within that
            // frame) to be the first camera pose
            initial_pose = pose;
            initial_frame_set = true;
            LOG(INFO) << pose;
          }
          original_camera_pose = pose;
          // offset initial pose so we start at "0, 0, 0"
          aligned_camera_pose = initial_pose.inverse() * pose;
          gt_packet.X_world_ = aligned_camera_pose;
        } else {
          ObjectPoseGT object_pose_gt;
          object_pose_gt.frame_id_ = frame;
          object_pose_gt.object_id_ = i;

          Eigen::Matrix3d R_carla_to_opencv;
          R_carla_to_opencv << 1, 0, 0, 0, 0, 1, 0, -1, 0;
          gtsam::Rot3 rot_carla_to_opencv(R_carla_to_opencv);
          gtsam::Pose3 T_carla_to_opencv(rot_carla_to_opencv,
                                         gtsam::Point3(0, 0, 0));
          gtsam::Pose3 object_pose(rot_carla_to_opencv * pose.rotation(),
                                   pose.translation());
          // gtsam::Pose3 object_pose = pose;

          LOG(INFO) << "Processing object " << i;

          // construct relative object pose with original camera pose
          auto relative_object_pose =
              original_camera_pose.inverse() * object_pose;
          auto aligned_object_pose = gt_packet.X_world_ * relative_object_pose;
          object_pose_gt.L_camera_ = relative_object_pose;
          object_pose_gt.L_world_ = aligned_object_pose;

          gt_packet.object_poses_.push_back(object_pose_gt);
        }
      }

      if (frame > 0) {
        const GroundTruthInputPacket& previous_gt_packet =
            ground_truth_packets.at(frame - 1);
        gt_packet.calculateAndSetMotions(previous_gt_packet);
      }

      ground_truth_packets.insert2(frame, gt_packet);
    }
  }
  gtsam::Pose3Vector processPoseFile(
      const std::string& pose_file,
      GroundTruthPacketMap& ground_truth_packets) {
    std::ifstream fstream;
    fstream.open(pose_file, std::ios::in);
    if (!fstream) {
      throw std::runtime_error(
          "Could not open file " + pose_file +
          " when trying to load Cluster Slam pose information!");
    }

    // expected size of each line (3 translation components + 4 for quaternion)
    constexpr static size_t kExpectedSize = 7u;

    /**
     * @brief NOTE: the dataset claims that the format is x,y,z, qx, qy, qz, qw
     * however this is not true!!!
     * actual format is x,y,z, QW, qx, qy, qz
     *
     */
    auto construct_pose =
        [](const std::vector<std::string>& pose_string) -> gtsam::Pose3 {
      const double x = std::stod(pose_string.at(0));
      const double y = std::stod(pose_string.at(1));
      const double z = std::stod(pose_string.at(2));

      const double qw = std::stod(pose_string.at(3));
      const double qx = std::stod(pose_string.at(4));
      const double qy = std::stod(pose_string.at(5));
      const double qz = std::stod(pose_string.at(6));

      return gtsam::Pose3(gtsam::Rot3(qw, qx, qy, qz), gtsam::Point3(x, y, z));
    };

    gtsam::Pose3Vector poses;
    while (!fstream.eof()) {
      std::vector<std::string> split_line;
      if (!utils::getLine(fstream, split_line)) continue;

      CHECK_EQ(split_line.size(), kExpectedSize)
          << "Line present in pose file does not have the right length - "
          << container_to_string(split_line);
      poses.push_back(construct_pose(split_line));
    }
    return poses;
  }

 private:
  const std::string left_images_folder_path_;
  const std::string optical_flow_folder_path_;
  const std::string pose_folder_path_;
  const std::string left_landmarks_folder_;
  const std::string right_images_folder_path_;
  const std::string instance_masks_folder_;
  const std::string landmark_mapping_file_path_;
  const std::string intrinsics_file_path_;

  std::vector<std::string>
      left_rgb_image_paths_;  // index from 0 to match the naming convention of
                              // the dataset
  std::vector<std::string> right_rgb_image_paths_;
  std::vector<std::string> optical_flow_image_paths_;
  std::vector<std::string> instance_masks_image_paths_;
  LandmarksMapPerFrame left_landmarks_map_;

  // instance masks per frame - we need a chace of them as we're going to change
  // the index of each object to match the gt labels!!
  gtsam::FastMap<FrameId, cv::Mat> instance_masks_;

  // Landmark to cluster id mapping
  // cluster should be the actual track
  // the landmark id is the key in LandmarksMap
  gtsam::FastMap<int, int> landmark_mapping_;

  // set by setIntrinsics()
  gtsam::Matrix33 K_cam_1;
  gtsam::Matrix33 K_cam_2;
  gtsam::Matrix34 projection_matrix_cam1_;
  gtsam::Matrix34 projection_matrix_cam2_;
  CameraParams left_camera_params_;

  StereoCamera::Ptr stereo_camera_;
  StereoMatcher::Ptr stereo_matcher_;

  GroundTruthPacketMap ground_truth_packets_;
  size_t dataset_size_;  // set in setGroundTruthPacket. Reflects the number of
                         // files in the /optical_flow folder which is one per
                         // frame
};

struct ClusterSlamTimestampLoader : public TimestampBaseLoader {
  ClusterSlamAllLoader::Ptr loader_;

  ClusterSlamTimestampLoader(ClusterSlamAllLoader::Ptr loader)
      : loader_(CHECK_NOTNULL(loader)) {}
  std::string getFolderName() const override { return ""; }

  size_t size() const override { return loader_->size(); }

  double getItem(size_t idx) override { return loader_->getTimestamp(idx); }
};

ClusterSlamDataLoader::ClusterSlamDataLoader(const fs::path& dataset_path)
    : ClusterSlamDatasetProvider(dataset_path) {
  LOG(INFO) << "Starting ClusterSlamDataLoader with path" << dataset_path;

  // this would go out of scope but we capture it in the functional loaders
  auto loader = std::make_shared<ClusterSlamAllLoader>(dataset_path);
  auto timestamp_loader = std::make_shared<ClusterSlamTimestampLoader>(loader);

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

  auto rgb_right_loader =
      std::make_shared<FunctionalDataFolder<std::optional<cv::Mat>>>(
          [loader](size_t idx) { return loader->getRightRGB(idx); });

  this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader,
                   depth_loader, instance_mask_loader, gt_loader,
                   rgb_right_loader);

  auto callback = [&](size_t frame_id, Timestamp timestamp, cv::Mat rgb,
                      cv::Mat optical_flow, cv::Mat depth,
                      cv::Mat instance_mask,
                      GroundTruthInputPacket gt_object_pose_gt,
                      std::optional<cv::Mat> right_rgb) -> bool {
    CHECK_EQ(timestamp, gt_object_pose_gt.timestamp_);

    CHECK(ground_truth_packet_callback_);
    if (ground_truth_packet_callback_)
      ground_truth_packet_callback_(gt_object_pose_gt);

    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
        .depth(depth)
        .opticalFlow(optical_flow)
        .objectMotionMask(instance_mask);

    // TODO: currently cannot use this as frontend does not take RGBDCamera as
    // input (ie no baseline!!)
    //  if (right_rgb) image_container.rightRgb(right_rgb.value());

    CHECK(image_container_callback_);
    if (image_container_callback_)
      image_container_callback_(
          std::make_shared<ImageContainer>(image_container));
    return true;
  };

  this->setCallback(callback);
}

}  // namespace dyno
