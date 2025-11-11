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

#include "dynosam/dataprovider/VirtualKittiDataProvider.hpp"

#include <glog/logging.h>

#include <fstream>
#include <png++/png.hpp>  //libpng-dev

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"  //for getObjectLabels

namespace dyno {

/**
 * @brief NOTE!!: In vitual kitti, the track id's start at 0 and then the ground
 * truth instances in the semantic instances are indexed from trackId+1 such
 * that a pixel value of zero means not a vehicle. In order to reduce confusion
 * when we work with the semantic images and then associate with the ground
 * truth labels, we will index all tracks as trackID = virtualKittiTrackID + 1
 * so all tracks start from 1!
 *
 * @param split_string
 * @param idx
 * @return int
 */
inline int getTrackID(const std::vector<std::string>& split_string, int idx) {
  return std::stoi(split_string.at(idx)) + 1;
}

/**
 * @brief Loading class for the Virtual Kitti 2 dataset:
 * https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
 *
 */

class GenericVirtualKittiImageLoader : public dyno::DataFolder<cv::Mat> {
 public:
  GenericVirtualKittiImageLoader(const std::string& full_path)
      : full_path_(full_path) {}

  std::string getFolderName() const override { return ""; }
  const std::string
      full_path_;  //! Full path as set by the full path constructor argument
};

class VirtualKittiRGBDataFolder : public GenericVirtualKittiImageLoader {
 public:
  VirtualKittiRGBDataFolder(const std::string& path)
      : GenericVirtualKittiImageLoader(path) {}
  cv::Mat getItem(size_t idx) override {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(5) << idx;
    const std::string file_path = full_path_ + "/rgb_" + ss.str() + ".jpg";
    utils::throwExceptionIfPathInvalid(file_path);

    cv::Mat rgb;
    utils::loadRGB(file_path, rgb);

    return rgb;
  }
};

class VirtualKittiFlowDataFolder : public GenericVirtualKittiImageLoader {
 public:
  VirtualKittiFlowDataFolder(const std::string& path)
      : GenericVirtualKittiImageLoader(path) {}

  cv::Mat getItem(size_t idx) override {
    CHECK(idx > 0);
    std::stringstream ss;
    // index at frame -1 so that we get the flow from t-1 to t (where the idx is
    // t)
    ss << std::setfill('0') << std::setw(5) << idx - 1;
    const std::string file_path = full_path_ + "/flow_" + ss.str() + ".png";
    // const std::string file_path = full_path_  + "/backwardFlow_" + ss.str() +
    // ".png"; //specific to backwards flow
    utils::throwExceptionIfPathInvalid(file_path);
    return vKittiPngToFlow(file_path);
  }

 private:
  // as per example code in
  // https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
  //  In R, flow along x-axis normalized by image width and quantized to [0;2^16
  //  – 1] In G, flow along x-axis normalized by image width and quantized to
  //  [0;2^16 – 1] B = 0 for invalid flow (e.g., sky pixels)
  cv::Mat vKittiPngToFlow(const std::string& path) const {
    // cv::Mat bgr = cv::imread(path, cv::IMREAD_ANYCOLOR |
    // cv::IMREAD_ANYDEPTH);
    cv::Mat bgr;
    utils::loadRGB(path, bgr);

    CHECK(!bgr.empty());

    int h = bgr.rows;
    int w = bgr.cols;
    int c = bgr.channels();

    CHECK_EQ(bgr.type(), CV_16UC3) << "type was " << utils::cvTypeToString(bgr);
    CHECK_EQ(c, 3);

    cv::Mat bgr_channels[3];
    cv::split(bgr, bgr_channels);
    cv::Mat b_channel = bgr_channels[0];

    cv::Mat bgr_float;
    // float 32
    bgr.convertTo(bgr_float, CV_32F);

    // scaled to [0;2**16 – 1]
    cv::Mat unscaled_out_flow = 2.0 / (std::pow(2, 16) - 1.0) * bgr_float - 1;

    cv::Mat channels[2];
    // now only take g and r channels
    cv::split(unscaled_out_flow, channels);
    // g,r == flow_y,x normalized by height,width
    cv::Mat& flow_y = channels[1];
    cv::Mat& flow_x = channels[2];
    flow_x *= w - 1.0;
    flow_y *= h - 1.0;

    // re-order channels so x is first and then y
    cv::Mat x_y_ordered_channels[] = {flow_x, flow_y};
    cv::Mat out_flow;
    cv::merge(x_y_ordered_channels, 2, out_flow);

    // b == invalid flow flag == 0 for sky or other invalid flow
    cv::Mat invalid = (b_channel == 0);
    CHECK(!invalid.empty());
    out_flow.setTo(0, invalid);

    out_flow.convertTo(out_flow, CV_32F);

    return out_flow;
  }
};

class VirtualKittiDepthDataFolder : public GenericVirtualKittiImageLoader {
 public:
  VirtualKittiDepthDataFolder(const std::string& path)
      : GenericVirtualKittiImageLoader(path) {}

  cv::Mat getItem(size_t idx) override {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(5) << idx;
    const std::string file_path = full_path_ + "/depth_" + ss.str() + ".png";
    utils::throwExceptionIfPathInvalid(file_path);

    // TODO: for now? When to apply preprocessing?
    cv::Mat depth;
    utils::loadRGB(file_path, depth);
    // first convert to double as utils::loadRGB provides 16UC1
    depth.convertTo(depth, CV_64F);
    // apply depth factor to get information into meters (from cm)
    depth /= 100.0;

    return depth;
  }
};

struct VirtualKittiTimestampLoader : public TimestampBaseLoader {
  VirtualKittiTimestampLoader(size_t total_num_frames)
      : total_num_frames_(total_num_frames) {}

  std::string getFolderName() const override { return ""; }
  double getItem(size_t idx) override {
    // we dont have a time? for now just make idx
    return static_cast<double>(idx);
  }

  size_t size() const override {
    // only valid after loading
    // we go up to -1 becuase the optical flow has ONE LESS image
    return total_num_frames_ - 1u;
  }

  const size_t total_num_frames_;
};

class VirtualKittiTextGtLoader {
 public:
  DYNO_POINTER_TYPEDEFS(VirtualKittiTextGtLoader)

  /**
   * @brief Construct a new Virtual Kitti Text Gt Laoder object
   *
   * File path is full file path to the textgt folder containing bbox.txt,
   * colors.txt, ... pose.txt
   *
   * @param file_path
   */
  VirtualKittiTextGtLoader(const std::string& file_path)
      : bbox_path(file_path + "/bbox.txt"),
        colors_path(file_path + "/colors.txt"),
        extrinsic_path(file_path + "/extrinsic.txt"),
        info_path(file_path + "/info.txt"),
        intrinsic_path(file_path + "/intrinsic.txt"),
        pose_path(file_path + "/pose.txt") {
    utils::throwExceptionIfPathInvalid(bbox_path);
    utils::throwExceptionIfPathInvalid(colors_path);
    utils::throwExceptionIfPathInvalid(extrinsic_path);
    utils::throwExceptionIfPathInvalid(info_path);
    utils::throwExceptionIfPathInvalid(intrinsic_path);
    utils::throwExceptionIfPathInvalid(pose_path);

    loadBBoxMetaData(bbox_data_);
    loadPoseTxt(object_poses_, bbox_data_);
    loadCameraPoses(camera_poses_);

    gt_packets_ = constructGTPackets(object_poses_, camera_poses_);
  }

  const GroundTruthInputPacket& getGTPacket(size_t frame) const {
    return gt_packets_.at(frame);
  }

  size_t size() const { return gt_packets_.size(); }

  /**
   * @brief Queries if an object is moving in the current frame
   *
   * Unlike in Virtual kitti which indicates if movement is between frame t and
   * t+1, we need to know if the object moved between t-1 and t (as we estimate
   * for t-1 to t)
   *
   * The frame argument is considered to be frame t and we requery the isMoving
   * variable of the BBoxMetaData at t-1.
   *
   * If object does not exist at frame return false?
   *
   * @param frame
   * @param object_id
   * @return true
   * @return false
   */
  bool isMoving(FrameId frame, ObjectId object_id) const {
    // check current frame and prev frame
    if (checkExists(frame, object_id, bbox_data_) &&
        checkExists(frame - 1, object_id, bbox_data_)) {
      return bbox_data_.at(frame - 1).at(object_id).is_moving;
    }
    return false;
  }

 private:
  /**
   * @brief Nested map mapping frame id to a map of object ids to a type T
   * Allows fast access of object per frame
   *
   * @tparam T
   */
  template <typename T>
  using FrameObjectIDMap = std::map<FrameId, std::map<ObjectId, T>>;

  template <typename T>
  bool checkExists(FrameId frame_id, ObjectId object_id,
                   const FrameObjectIDMap<T>& map) const {
    auto it = map.find(frame_id);
    if (it == map.end()) return false;

    const auto& inner_map = it->second;
    auto it_inner = inner_map.find(object_id);

    if (it_inner == inner_map.end()) return false;

    return true;
  }

  template <typename T>
  void addToFrameObjectMap(FrameId frame_id, ObjectId object_id, T t,
                           FrameObjectIDMap<T>& map) {
    if (map.find(frame_id) == map.end()) {
      map[frame_id] = std::map<ObjectId, T>();
    }

    map[frame_id][object_id] = t;
  }

  /// @brief Storing metadata from the bbox.txt
  struct BBoxMetaData {
    FrameId frame;
    ObjectId track_id;  //! to be treated as object id
    cv::Rect bbox;
    bool is_moving;  //! flag to indicate whether the object is really moving
                     //! between this frame and the next one
  };

 private:
  void loadBBoxMetaData(FrameObjectIDMap<BBoxMetaData>& bbox_metadata) {
    std::ifstream fstream;
    fstream.open(bbox_path, std::ios::in);
    if (!fstream) {
      throw std::runtime_error(
          "Could not open file " + bbox_path +
          " when trying to load BBox metadata for Virtual Kitti dataset!");
    }

    LOG(INFO) << "Loading BBox metadta for virtual kitti datasetat path: "
              << bbox_path;

    static const std::map<std::string, int> header_map = {
        {"frame", 0},
        {"cameraID", 1},
        {"trackID", 2},
        {"left", 3},
        {"right", 4},
        {"top", 5},
        {"bottom", 6},
        {"number_pixels", 7},
        {"truncation_ratio", 8},
        {"occupancy_ratio", 9},
        {"isMoving", 10},
    };

    auto convert_string_bool = [](const std::string& string_bool) -> bool {
      if (string_bool == "True")
        return true;
      else if (string_bool == "False")
        return false;
      else
        throw std::runtime_error("Unknown string bool " + string_bool +
                                 " when attempting to covnert to bool type");
    };

    bool is_first = true;  // to access header
    while (!fstream.eof()) {
      std::vector<std::string> split_line;
      if (!utils::getLine(fstream, split_line)) continue;

      CHECK_EQ(split_line.size(), header_map.size())
          << "Header map and the line present in the bbox.txt file do not have "
             "the same length - "
          << container_to_string(split_line);

      if (is_first) {
        // check header and validate
        for (const auto& [header_value, index] : header_map) {
          const std::string value = split_line.at(index);
          if (value != header_value) {
            throw std::runtime_error("Header value mismatch - " + value +
                                     " != " + header_value);
          }
        }
        is_first = false;
      } else {
        const int frame = std::stoi(split_line.at(header_map.at("frame")));
        const int camera_id =
            std::stoi(split_line.at(header_map.at("cameraID")));
        const int track_id = getTrackID(split_line, header_map.at("trackID"));
        const int left = std::stoi(split_line.at(header_map.at("left")));
        const int right = std::stoi(split_line.at(header_map.at("right")));
        const int top = std::stoi(split_line.at(header_map.at("top")));
        const int bottom = std::stoi(split_line.at(header_map.at("bottom")));
        // const int number_pixels =
        // std::stoi(split_line.at(header_map.at("number_pixels"))); const
        // double truncation_ratio =
        // std::stod(split_line.at(header_map.at("truncation_ratio"))); const
        // double occupancy_ratio =
        // std::stod(split_line.at(header_map.at("occupancy_ratio")));
        const bool is_moving =
            convert_string_bool(split_line.at(header_map.at("isMoving")));

        if (camera_id != 0) {
          continue;  // hardcoded for only one camera atm
        }

        BBoxMetaData bbox_data;
        bbox_data.frame = frame;
        bbox_data.track_id = track_id;
        bbox_data.is_moving = is_moving;

        // convert to cv::Rect where image starts top right
        cv::Point tl(left, top);
        cv::Point br(right, bottom);
        bbox_data.bbox = cv::Rect(tl, br);

        addToFrameObjectMap(frame, track_id, bbox_data, bbox_metadata);
      }
    }
  }

  void loadPoseTxt(FrameObjectIDMap<ObjectPoseGT>& object_poses,
                   const FrameObjectIDMap<BBoxMetaData>& bbox_metadata) {
    std::ifstream fstream;
    fstream.open(pose_path, std::ios::in);
    if (!fstream) {
      throw std::runtime_error("Could not open file " + pose_path +
                               " when trying to load Object pose metadata for "
                               "Virtual Kitti dataset!");
    }

    LOG(INFO)
        << "Loading Object pose metadta for virtual kitti datasetat path: "
        << pose_path;

    static const std::map<std::string, int> header_map = {
        {"frame", 0},
        {"cameraID", 1},
        {"trackID", 2},
        {"alpha", 3},
        {"width", 4},
        {"height", 5},
        {"length", 6},
        {"world_space_X", 7},
        {"world_space_Y", 8},
        {"world_space_Z", 9},
        {"rotation_world_space_y", 10},
        {"rotation_world_space_x", 11},
        {"rotation_world_space_z", 12},
        {"camera_space_X", 13},
        {"camera_space_Y", 14},
        {"camera_space_Z", 15},
        {"rotation_camera_space_y", 16},
        {"rotation_camera_space_x", 17},
        {"rotation_camera_space_z", 18},
    };

    bool is_first = true;  // to access header
    while (!fstream.eof()) {
      std::vector<std::string> split_line;
      if (!utils::getLine(fstream, split_line)) continue;

      CHECK_EQ(split_line.size(), header_map.size())
          << "Header map and the line present in the pose.txt file do not have "
             "the same length";

      if (is_first) {
        // check header and validate
        for (const auto& [header_value, index] : header_map) {
          CHECK_EQ(split_line.at(index), header_value);
        }
        is_first = false;
      } else {
        const int frame = std::stoi(split_line.at(header_map.at("frame")));
        const int camera_id =
            std::stoi(split_line.at(header_map.at("cameraID")));
        const int track_id = getTrackID(split_line, header_map.at("trackID"));

        if (camera_id != 0) {
          continue;  // hardcoded for only one camera atm
        }

        // const double alpha =
        // std::stod(split_line.at(header_map.at("alpha")));
        const double width = std::stod(split_line.at(header_map.at("width")));
        const double height = std::stod(split_line.at(header_map.at("height")));
        const double length = std::stod(split_line.at(header_map.at("length")));
        // const double world_space_X =
        // std::stod(split_line.at(header_map.at("world_space_X"))); const
        // double world_space_Y =
        // std::stod(split_line.at(header_map.at("world_space_Y"))); const
        // double world_space_Z =
        // std::stod(split_line.at(header_map.at("world_space_Z"))); const
        // double rotation_world_space_y =
        // std::stod(split_line.at(header_map.at("rotation_world_space_y")));
        // const double rotation_world_space_x =
        // std::stod(split_line.at(header_map.at("rotation_world_space_x")));
        // const double rotation_world_space_z =
        // std::stod(split_line.at(header_map.at("rotation_world_space_z")));
        const double camera_space_X =
            std::stod(split_line.at(header_map.at("camera_space_X")));
        const double camera_space_Y =
            std::stod(split_line.at(header_map.at("camera_space_Y")));
        const double camera_space_Z =
            std::stod(split_line.at(header_map.at("camera_space_Z")));
        const double rotation_camera_space_y =
            std::stod(split_line.at(header_map.at("rotation_camera_space_y")));
        const double rotation_camera_space_x =
            std::stod(split_line.at(header_map.at("rotation_camera_space_x")));
        const double rotation_camera_space_z =
            std::stod(split_line.at(header_map.at("rotation_camera_space_z")));

        ObjectPoseGT object_pose_gt;
        object_pose_gt.frame_id_ = frame;
        object_pose_gt.object_id_ = track_id;
        object_pose_gt.object_dimensions_ =
            gtsam::Vector3(width, height, length);

        {
          // assign r vector
          // double y = rotation_camera_space_y + (3.1415926 / 2);  //
          // +(3.1415926/2)
          double y = rotation_camera_space_y;  // +(3.1415926/2)
          double x = rotation_camera_space_x;
          double z = rotation_camera_space_z;

          // the angles are in radians.
          double cy = cos(y);
          double sy = sin(y);
          double cx = cos(x);
          double sx = sin(x);
          double cz = cos(z);
          double sz = sin(z);

          double m00, m01, m02, m10, m11, m12, m20, m21, m22;

          m00 = cy * cz + sy * sx * sz;
          m01 = -cy * sz + sy * sx * cz;
          m02 = sy * cx;
          m10 = cx * sz;
          m11 = cx * cz;
          m12 = -sx;
          m20 = -sy * cz + cy * sx * sz;
          m21 = sy * sz + cy * sx * cz;
          m22 = cy * cx;

          gtsam::Matrix33 rot;
          rot(0, 0) = m00;
          rot(0, 1) = m01;
          rot(0, 2) = m02;
          rot(1, 0) = m10;
          rot(1, 1) = m11;
          rot(1, 2) = m12;
          rot(2, 0) = m20;
          rot(2, 1) = m21;
          rot(2, 2) = m22;

          gtsam::Vector3 translation(camera_space_X, camera_space_Y,
                                     camera_space_Z);

          object_pose_gt.L_camera_ =
              gtsam::Pose3(gtsam::Rot3(rot), translation);
        }

        CHECK(checkExists(frame, track_id, bbox_metadata));

        // check asociated bboxdata has the right info for this object
        const BBoxMetaData& bbox_data = bbox_metadata.at(frame).at(track_id);
        CHECK_EQ(bbox_data.frame, frame);
        CHECK_EQ(bbox_data.track_id, track_id);

        object_pose_gt.bounding_box_ = bbox_data.bbox;
        addToFrameObjectMap(frame, track_id, object_pose_gt, object_poses);
      }
    }
  }

  void loadCameraPoses(std::vector<gtsam::Pose3>& camera_poses) {
    std::ifstream fstream;
    fstream.open(extrinsic_path, std::ios::in);
    if (!fstream) {
      throw std::runtime_error(
          "Could not open file " + extrinsic_path +
          " when trying to load camera exterinsics for Virtual Kitti dataset!");
    }

    LOG(INFO) << "Camera pose data for virtual kitti datasetat path: "
              << extrinsic_path;

    bool is_first = true;

    gtsam::Pose3 initial_pose = gtsam::Pose3::Identity();
    bool set_initial_pose = false;

    while (!fstream.eof()) {
      std::vector<std::string> split_line;
      if (!utils::getLine(fstream, split_line)) continue;

      // frame camera_id, 3x4 camera pose (r11, r12, r13, t1....)
      // header: frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2
      // r3,3 t3 0 0 0 1
      constexpr static size_t kExtrinsicFileLength = 18u;
      CHECK_EQ(split_line.size(), kExtrinsicFileLength);

      if (is_first) {
        is_first = false;
      } else {
        const int frame = std::stoi(split_line.at(0));
        const int camera_id = std::stoi(split_line.at(1));

        if (camera_id != 0) {
          continue;  // hardcoded for only one camera atm
        }

        // sanity check to ensure things are added in order and at the right
        // index (frame is zero indexed) check only after we check the camera id
        CHECK_EQ(frame, camera_poses.size());

        const double r11 = std::stod(split_line.at(2));
        const double r12 = std::stod(split_line.at(3));
        const double r13 = std::stod(split_line.at(4));
        const double t1 = std::stod(split_line.at(5));
        const double r21 = std::stod(split_line.at(6));
        const double r22 = std::stod(split_line.at(7));
        const double r23 = std::stod(split_line.at(8));
        const double t2 = std::stod(split_line.at(9));
        const double r31 = std::stod(split_line.at(10));
        const double r32 = std::stod(split_line.at(11));
        const double r33 = std::stod(split_line.at(12));
        const double t3 = std::stod(split_line.at(13));

        gtsam::Matrix33 rot;
        rot(0, 0) = r11;
        rot(0, 1) = r12;
        rot(0, 2) = r13;
        rot(1, 0) = r21;
        rot(1, 1) = r22;
        rot(1, 2) = r23;
        rot(2, 0) = r31;
        rot(2, 1) = r32;
        rot(2, 2) = r33;

        // last 4 values should be [0 0 0 1] for homogenous matrix form so we
        // just check
        CHECK_EQ(std::stod(split_line.at(14)), 0);
        CHECK_EQ(std::stod(split_line.at(15)), 0);
        CHECK_EQ(std::stod(split_line.at(16)), 0);
        CHECK_EQ(std::stod(split_line.at(17)), 1);

        gtsam::Pose3 pose(gtsam::Rot3(rot), gtsam::Point3(t1, t2, t3));

        const static gtsam::Pose3 camera_to_world(
            gtsam::Rot3::RzRyRx(M_PI_2, 0, M_PI_2),
            gtsam::traits<gtsam::Point3>::Identity());
        // pose is in world coordinate convention so we put into camera
        // convention (the camera_to_world.inverse()) the pose provided is
        // actually world -> camera but we want camera -> world eg
        // T_world_camera, so we apply the inverse transform
        pose = camera_to_world.inverse() * pose.inverse();

        if (!set_initial_pose) {
          initial_pose = pose;
          set_initial_pose = true;
        }

        // offset initial pose so we start at "0, 0, 0"
        pose = initial_pose.inverse() * pose;
        camera_poses.push_back(pose);
      }
    }
  }

  // only include objects that are moving - we will then use the returned vector
  // as the input to the mask loader to remove masks for stationary objects
  std::vector<GroundTruthInputPacket> constructGTPackets(
      const FrameObjectIDMap<ObjectPoseGT>& object_poses,
      const std::vector<gtsam::Pose3>& camera_poses) const {
    std::vector<GroundTruthInputPacket> gt_packets;
    for (size_t frame_id = 0; frame_id < camera_poses.size(); frame_id++) {
      GroundTruthInputPacket gt_packet;
      gt_packet.frame_id_ = frame_id;
      gt_packet.X_world_ = camera_poses.at(frame_id);

      const auto& it = object_poses.find(frame_id);
      if (it != object_poses.end()) {
        auto& object_id_pose_map = it->second;
        for (auto& [object_id, object_pose_gt] : object_id_pose_map) {
          // query for moving
          //  if(isMoving(frame_id, object_id)) {
          //  LOG(INFO) << "Added moving object " << frame_id << " " <<
          //  object_id;
          auto object_pose = object_pose_gt;
          object_pose.L_world_ = gt_packet.X_world_ * object_pose.L_camera_;
          gt_packet.object_poses_.push_back(object_pose);
          // }
        }
      }

      // set motions
      if (frame_id > 0) {
        const GroundTruthInputPacket& previous_gt_packet =
            gt_packets.at(frame_id - 1);
        gt_packet.calculateAndSetMotions(previous_gt_packet);
      }

      gt_packets.push_back(gt_packet);
    }

    return gt_packets;
  }

 private:
  const std::string bbox_path;
  const std::string colors_path;
  const std::string extrinsic_path;
  const std::string info_path;
  const std::string intrinsic_path;
  const std::string pose_path;

  FrameObjectIDMap<ObjectPoseGT> object_poses_;  //! all object poses
  FrameObjectIDMap<BBoxMetaData> bbox_data_;     //! all object poses

  std::vector<gtsam::Pose3> camera_poses_;  //! camera poses in the world frame
  std::vector<GroundTruthInputPacket> gt_packets_;  //!
};

// to be inherited either for MOTION segmentation or normal instance seg
class VirtualKittiGenericInstanceSegFolder
    : public GenericVirtualKittiImageLoader {
 public:
  DYNO_POINTER_TYPEDEFS(VirtualKittiGenericInstanceSegFolder)

  VirtualKittiGenericInstanceSegFolder(const std::string& path,
                                       VirtualKittiTextGtLoader::Ptr gt_loader)
      : GenericVirtualKittiImageLoader(path),
        gt_loader_(CHECK_NOTNULL(gt_loader)) {}
  virtual ~VirtualKittiGenericInstanceSegFolder() = default;

  cv::Mat getItem(size_t idx) override {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(5) << idx;
    const std::string file_path =
        full_path_ + "/instancegt_" + ss.str() + ".png";
    utils::throwExceptionIfPathInvalid(file_path);

    const GroundTruthInputPacket& gt_packet = gt_loader_->getGTPacket(idx);
    return loadImage(file_path, idx, gt_packet);
  }

 protected:
  virtual cv::Mat loadImage(const std::string& file_path, FrameId frame,
                            const GroundTruthInputPacket& gt_packet) = 0;

  /**
   * @brief Loads instance semantic mask from the file path.
   *
   * This loads the image using libpng++ since we need to load the image as an
   * 8bit indexed PNG file (which opencv does not have functionality for).
   * Keeping consistent with the ground truth indexing we use for VirtualKitti
   * we do not need to modify the trackID's within the image since we want the
   * trackID's to be indexed from 1
   *
   * @param file_path
   * @return cv::Mat
   */
  cv::Mat loadSemanticInstanceUnchanged(const std::string& file_path) const {
    png::image<png::index_pixel> index_image(file_path);

    cv::Size size(index_image.get_width(), index_image.get_height());
    cv::Mat semantic_image(size, ImageType::MotionMask::OpenCVType);

    for (size_t y = 0; y < index_image.get_height(); ++y) {
      for (size_t x = 0; x < index_image.get_width(); ++x) {
        const auto& pixel = index_image.get_pixel(x, y);

        // the semantic label
        int i_byte = static_cast<int>(pixel);
        semantic_image.at<int>(y, x) = i_byte;
      }
    }
    return semantic_image;
  }

 protected:
  VirtualKittiTextGtLoader::Ptr gt_loader_;
};

// this one we will have to remove semgentation mask depending on whats in the
// gt folder
class VirtualKittiMotionSegFolder
    : public VirtualKittiGenericInstanceSegFolder {
 public:
  VirtualKittiMotionSegFolder(const std::string& path,
                              VirtualKittiTextGtLoader::Ptr gt_loader)
      : VirtualKittiGenericInstanceSegFolder(path, gt_loader) {}

  cv::Mat loadImage(const std::string& file_path, FrameId frame,
                    const GroundTruthInputPacket& gt_packet) override {
    CHECK_EQ(frame, gt_packet.frame_id_);

    // remove object in semantic mask
    cv::Mat instance_semantic_mask = loadSemanticInstanceUnchanged(file_path);
    cv::Mat motion_mask;

    // we used to use gt_loader->isMoving function which seems to be back
    // indexed (see the function comment) now we use the motion_info information
    // in the ground truth packet (see removeStaticObjectFromMask()) not sure if
    // this actually makes a difference...
    removeStaticObjectFromMask(instance_semantic_mask, motion_mask, gt_packet);
    return motion_mask;
  }
};

class VirtualKittiClassSegmentationDataFolder
    : public GenericVirtualKittiImageLoader {
 public:
  VirtualKittiClassSegmentationDataFolder(const std::string& path)
      : GenericVirtualKittiImageLoader(path) {}
  cv::Mat getItem(size_t idx) override {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(5) << idx;
    const std::string file_path = full_path_ + "/classgt_" + ss.str() + ".png";
    utils::throwExceptionIfPathInvalid(file_path);
    return loadClassSegmentation(file_path);
  }

 private:
  // TODO: comments
  cv::Mat loadClassSegmentation(const std::string& file_path) const {
    // TODO: for now just hardcode in road!!!
    png::image<png::rgb_pixel> rgb_image(file_path);

    cv::Size size(rgb_image.get_width(), rgb_image.get_height());
    cv::Mat class_segmentation(size, ImageType::ClassSegmentation::OpenCVType);

    for (size_t y = 0; y < rgb_image.get_height(); ++y) {
      for (size_t x = 0; x < rgb_image.get_width(); ++x) {
        const auto& pixel = rgb_image.get_pixel(x, y);
        const unsigned char red = (unsigned char)pixel.red;
        const unsigned char green = (unsigned char)pixel.green;
        const unsigned char blue = (unsigned char)pixel.blue;

        // class_segmentation_rgb.at<cv::Vec3b>(y, x)[0] = blue;
        // class_segmentation_rgb.at<cv::Vec3b>(y, x)[1] = green;
        // class_segmentation_rgb.at<cv::Vec3b>(y, x)[2] = red;

        // TODO: hardcoded road values from VKITTI2
        if (red == 100 && green == 60 && blue == 100) {
          class_segmentation.at<int>(y, x) =
              ImageType::ClassSegmentation::Labels::Road;
        } else {
          class_segmentation.at<int>(y, x) =
              ImageType::ClassSegmentation::Labels::Undefined;
        }
      }
    }

    return class_segmentation;
  }
};

// instance segmentation for all objects regardless of motion
class VirtualKittiInstanceSegFolder
    : public VirtualKittiGenericInstanceSegFolder {
 public:
  VirtualKittiInstanceSegFolder(const std::string& path,
                                VirtualKittiTextGtLoader::Ptr gt_loader)
      : VirtualKittiGenericInstanceSegFolder(path, gt_loader) {}

  cv::Mat loadImage(const std::string& file_path, FrameId,
                    const GroundTruthInputPacket&) override {
    return loadSemanticInstanceUnchanged(file_path);
  }
};

class VirtualKittiGTFolder : public DataFolder<GroundTruthInputPacket> {
 public:
  VirtualKittiGTFolder(VirtualKittiTextGtLoader::Ptr gt_loader,
                       VirtualKittiTimestampLoader::Ptr timestamp_loader)
      : gt_loader_(CHECK_NOTNULL(gt_loader)),
        timestamp_loader_(CHECK_NOTNULL(timestamp_loader)) {}

  std::string getFolderName() const override { return ""; }

  GroundTruthInputPacket getItem(size_t idx) override {
    auto packet = gt_loader_->getGTPacket(idx);
    packet.timestamp_ = timestamp_loader_->getItem(idx);
    return packet;
  }

  VirtualKittiTextGtLoader::Ptr gt_loader_;
  VirtualKittiTimestampLoader::Ptr timestamp_loader_;
};

// everything current assumes camera0!! This includes loading things like
// extrinsics...
VirtualKittiDataLoader::VirtualKittiDataLoader(const fs::path& dataset_path,
                                               const Params& params)
    : VirtualKittiDatasetProvider(dataset_path), params_(params) {
  const std::string& scene = params.scene;
  const std::string& scene_type = params.scene_type;
  const auto& mask_type = params.mask_type;
  const std::string path = dataset_path;

  LOG(INFO) << "Starting VirtualKittiDataLoader with path " << path
            << " requested scene " << scene << " and scene type " << scene_type;
  const std::string depth_folder = path + "/" + v_depth_folder + "/" + scene +
                                   "/" + scene_type + "/frames/depth/Camera_0";
  const std::string forward_flow_folder = path + "/" + v_forward_flow_folder +
                                          "/" + scene + "/" + scene_type +
                                          "/frames/forwardFlow/Camera_0";
  const std::string backward_flow_folder = path + "/" + v_backward_flow_folder +
                                           "/" + scene + "/" + scene_type +
                                           "/frames/backwardFlow/Camera_0";
  const std::string forward_scene_flow_folder =
      path + "/" + v_forward_scene_flow_folder + "/" + scene + "/" +
      scene_type + "/frames/forwardsceneFlow/Camera_0";
  const std::string class_semantics_folder =
      path + "/" + v_class_semantics_folder + "/" + scene + "/" + scene_type +
      "/frames/classSegmentation/Camera_0";
  const std::string instance_segmentation_folder =
      path + "/" + v_instance_segmentation_folder + "/" + scene + "/" +
      scene_type + "/frames/instanceSegmentation/Camera_0";
  const std::string rgb_folder = path + "/" + v_rgb_folder + "/" + scene + "/" +
                                 scene_type + "/frames/rgb/Camera_0";
  const std::string text_gt_folder =
      path + "/" + v_text_gt_folder + "/" + scene + "/" + scene_type;

  utils::throwExceptionIfPathInvalid(depth_folder);
  utils::throwExceptionIfPathInvalid(forward_flow_folder);
  utils::throwExceptionIfPathInvalid(forward_scene_flow_folder);
  utils::throwExceptionIfPathInvalid(class_semantics_folder);
  utils::throwExceptionIfPathInvalid(instance_segmentation_folder);
  utils::throwExceptionIfPathInvalid(rgb_folder);
  utils::throwExceptionIfPathInvalid(text_gt_folder);

  auto gt_loader = std::make_shared<VirtualKittiTextGtLoader>(text_gt_folder);

  auto timestamp_folder =
      std::make_shared<VirtualKittiTimestampLoader>(gt_loader->size());
  auto rgb_loader = std::make_shared<VirtualKittiRGBDataFolder>(rgb_folder);
  auto optical_flow_loader =
      std::make_shared<VirtualKittiFlowDataFolder>(forward_flow_folder);
  auto depth_loader =
      std::make_shared<VirtualKittiDepthDataFolder>(depth_folder);

  VirtualKittiGenericInstanceSegFolder::Ptr motion_mask_loader = nullptr;
  if (mask_type == MaskType::MOTION) {
    LOG(INFO) << "Using MaskType::MOTION for loading mask";
    motion_mask_loader = std::make_shared<VirtualKittiMotionSegFolder>(
        instance_segmentation_folder, gt_loader);
  } else if (mask_type == MaskType::SEMANTIC_INSTANCE) {
    LOG(INFO) << "Using MaskType::SEMANTIC_INSTANCE for loading mask";
    motion_mask_loader = std::make_shared<VirtualKittiInstanceSegFolder>(
        instance_segmentation_folder, gt_loader);
  } else {
    LOG(FATAL) << "Unknown MaskType for KittiDataLoader";
  }
  CHECK_NOTNULL(motion_mask_loader);

  auto gt_loader_folder =
      std::make_shared<VirtualKittiGTFolder>(gt_loader, timestamp_folder);
  auto class_semantics_loader =
      std::make_shared<VirtualKittiClassSegmentationDataFolder>(
          class_semantics_folder);

  this->setLoaders(timestamp_folder, rgb_loader, optical_flow_loader,
                   depth_loader, motion_mask_loader, class_semantics_loader,
                   gt_loader_folder);

  auto callback = [&](size_t frame_id, Timestamp timestamp, cv::Mat rgb,
                      cv::Mat optical_flow, cv::Mat depth,
                      cv::Mat instance_mask, cv::Mat class_semantics,
                      GroundTruthInputPacket gt_object_pose_gt) -> bool {
    CHECK(timestamp == gt_object_pose_gt.timestamp_);

    CHECK(ground_truth_packet_callback_);
    if (ground_truth_packet_callback_)
      ground_truth_packet_callback_(gt_object_pose_gt);

    // ImageContainer::Ptr image_container = nullptr;
    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
        .depth(depth)
        .opticalFlow(optical_flow)
        .objectMotionMask(instance_mask);

    // if(params_.mask_type == MaskType::MOTION) {
    //     image_container = ImageContainer::Create(
    //         timestamp,
    //         frame_id,
    //         ImageWrapper<ImageType::RGBMono>(rgb),
    //         ImageWrapper<ImageType::Depth>(depth),
    //         ImageWrapper<ImageType::OpticalFlow>(optical_flow),
    //         ImageWrapper<ImageType::MotionMask>(instance_mask),
    //         ImageWrapper<ImageType::ClassSegmentation>(class_semantics));
    // }
    // else {
    //     image_container = ImageContainer::Create(
    //         timestamp,
    //         frame_id,
    //         ImageWrapper<ImageType::RGBMono>(rgb),
    //         ImageWrapper<ImageType::Depth>(depth),
    //         ImageWrapper<ImageType::OpticalFlow>(optical_flow),
    //         ImageWrapper<ImageType::SemanticMask>(instance_mask));
    // }

    // CHECK(image_container);
    CHECK(image_container_callback_);
    if (image_container_callback_)
      image_container_callback_(
          std::make_shared<ImageContainer>(image_container));
    return true;
  };

  this->setCallback(callback);

  // TODO: virtual function in base class to validate starting/end frames as it
  // may be different between datasets?
  setStartingFrame(1u);  // have to start at at least 1 so we can index
                         // backwards with optical flow
}

void VirtualKittiDataLoader::setCameraParams() {
  CameraParams::IntrinsicsCoeffs intrinsics({725.0087, 725.0087, 620.5, 187});
  CameraParams::DistortionCoeffs distortion({0, 0, 0, 0});

  cv::Size image_size(1242, 375);
  camera_params_ =
      CameraParams(intrinsics, distortion, image_size, "radial-tangential");
}

VirtualKittiDataLoader::Params VirtualKittiDataLoader::Params::fromYaml(
    const std::string& params_folder) {
  YamlParser yaml_parser(params_folder + "DatasetParams.yaml");

  Params params;

  std::string mask_type;
  yaml_parser.getYamlParam("mask_type", &mask_type);
  params.mask_type = maskTypeFromString(mask_type);

  yaml_parser.getYamlParam("scene", &params.scene);
  yaml_parser.getYamlParam("scene_type", &params.scene_type);
  return params;
}

// bool VirtualKittiDataLoader::spin() {
//     return false;
// }

}  // namespace dyno
