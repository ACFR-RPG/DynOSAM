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

#include "dynosam_ros/displays/inbuilt_displays/FrontendInbuiltDisplayRos.hpp"

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/memory.h>
#include <pcl_conversions/pcl_conversions.h>

#include <dynosam/frontend/RGBDInstance-Definitions.hpp>
#include <dynosam/frontend/vision/Feature.hpp>  //for functional_keypoint
#include <dynosam/utils/SafeCast.hpp>
#include <dynosam/visualizer/ColourMap.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>

#include "dynosam_ros/RosUtils.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/qos.hpp"

namespace dyno {

FrontendInbuiltDisplayRos::FrontendInbuiltDisplayRos(
    const ReferenceFrames params, rclcpp::Node::SharedPtr node)
    : InbuiltDisplayCommon(params, node) {
  const rclcpp::QoS& sensor_data_qos = rclcpp::SensorDataQoS();
  tracking_image_pub_ =
      image_transport::create_publisher(node.get(), "tracking_image");
  static_tracked_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("static_cloud", 1);
  dynamic_tracked_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_cloud", 1);
  odometry_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  object_pose_pub_ =
      node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "composed_object_poses", 1);
  object_pose_path_pub_ =
      node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "composed_object_paths", 1);
  odometry_path_pub_ =
      node->create_publisher<nav_msgs::msg::Path>("odom_path", 2);
  object_motion_pub_ =
      node->create_publisher<nav_msgs::msg::Path>("object_motions", 1);
  object_bbx_line_pub_ =
      node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "object_bbx_viz", 1);
  object_bbx_pub_ =
      node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "object_bounding_boxes", 1);

  // TODO: fix up gt publisher namespacing -? currently
  // dynosam/dynosam/ground_trut....
  gt_odometry_pub_ =
      node->create_publisher<nav_msgs::msg::Odometry>("~/ground_truth/odom", 1);
  gt_object_pose_pub_ =
      node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "~/ground_truth/object_poses", 1);
  gt_object_path_pub_ =
      node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "~/ground_truth/object_paths", 1);
  gt_odom_path_pub_ = node->create_publisher<nav_msgs::msg::Path>(
      "~/ground_truth/odom_path", 1);
  gt_bounding_box_pub_ = image_transport::create_publisher(
      node.get(), "~/ground_truth/bounding_boxes");

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);
}

FrontendInbuiltDisplayRos::~FrontendInbuiltDisplayRos() {
  if (video_writer_) video_writer_->release();
}

void FrontendInbuiltDisplayRos::spinOnce(
    const FrontendOutputPacketBase::ConstPtr& frontend_output) {
  // TODO: does frontend or backend publish tf transform?
  if (frontend_output->debug_imagery_)
    publishDebugImage(*frontend_output->debug_imagery_);

  if (frontend_output->gt_packet_ && frontend_output->debug_imagery_) {
    const auto& debug_imagery = *frontend_output->debug_imagery_;
    // TODO: put tracking images back into frontend output
    const cv::Mat& rgb_image = debug_imagery.rgb_viz;
    publishGroundTruthInfo(frontend_output->getTimestamp(),
                           frontend_output->gt_packet_.value(), rgb_image);
  }

  RGBDInstanceOutputPacket::ConstPtr rgbd_output =
      safeCast<FrontendOutputPacketBase, RGBDInstanceOutputPacket>(
          frontend_output);
  if (rgbd_output) {
    processRGBDOutputpacket(rgbd_output);
    // TODO:currently only with RGBDInstanceOutputPacket becuase camera poses is
    // in RGBDInstanceOutputPacket but should be in base
    // (FrontendOutputPacketBase)
    publishOdometryPath(odometry_path_pub_, rgbd_output->camera_poses_,
                        frontend_output->getTimestamp());
    publishOdometry(frontend_output->T_world_camera_,
                    frontend_output->getTimestamp());
    // publishCameraMarker(
    //     frontend_output->T_world_camera_,
    //     frontend_output->getTimestamp(),
    //     "frontend_camera_pose",
    //     cv::viz::Color::yellow()
    // );
  }
}

void FrontendInbuiltDisplayRos::processRGBDOutputpacket(
    const RGBDInstanceOutputPacket::ConstPtr& rgbd_frontend_output) {
  CHECK(rgbd_frontend_output);
  publishPointCloud(static_tracked_points_pub_,
                    rgbd_frontend_output->static_landmarks_,
                    rgbd_frontend_output->T_world_camera_);

  // TODO: there is a bunch of repeated code in this function and in
  // groupObjectCloud we leave this as is becuase this function ALSO creates the
  // coloured point cloud (which groupObjectCloud does not) eventually, refactor
  // into one function or calcualte the coloured cloud in the
  // RGBDInstanceOutputPacket
  CloudPerObject clouds_per_obj = publishPointCloud(
      dynamic_tracked_points_pub_, rgbd_frontend_output->dynamic_landmarks_,
      rgbd_frontend_output->T_world_camera_);

  publishObjectPositions(object_pose_pub_,
                         rgbd_frontend_output->propogated_object_poses_,
                         rgbd_frontend_output->getFrameId(),
                         rgbd_frontend_output->getTimestamp(), "frontend");

  publishObjectPaths(object_pose_path_pub_,
                     rgbd_frontend_output->propogated_object_poses_,
                     rgbd_frontend_output->getFrameId(),
                     rgbd_frontend_output->getTimestamp(), "frontend", 60);

  // object bounding box using linelist and id text in rviz
  {
    visualization_msgs::msg::MarkerArray
        object_bbx_linelist_marker_array;  // using linelist
    visualization_msgs::msg::MarkerArray object_bbx_marker_array;  // using cube
    static visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    object_bbx_linelist_marker_array.markers.push_back(delete_marker);
    object_bbx_marker_array.markers.push_back(delete_marker);

    for (const auto& [object_id, obj_cloud] : clouds_per_obj) {
      pcl::PointXYZ centroid;
      pcl::computeCentroid(obj_cloud, centroid);

      const cv::Scalar colour = Color::uniqueId(object_id);

      visualization_msgs::msg::Marker txt_marker;
      txt_marker.header.frame_id = "world";
      txt_marker.ns = "object_id";
      txt_marker.id = object_id;
      txt_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      txt_marker.action = visualization_msgs::msg::Marker::ADD;
      txt_marker.header.stamp = node_->now();
      txt_marker.scale.z = 2.0;
      txt_marker.color.r = colour(0) / 255.0;
      txt_marker.color.g = colour(1) / 255.0;
      txt_marker.color.b = colour(2) / 255.0;
      txt_marker.color.a = 1;
      txt_marker.text = "obj " + std::to_string(object_id);
      txt_marker.pose.position.x = centroid.x;
      txt_marker.pose.position.y = centroid.y - 2.0;
      txt_marker.pose.position.z = centroid.z - 1.0;
      object_bbx_linelist_marker_array.markers.push_back(txt_marker);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_cloud_ptr =
          pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >(obj_cloud);
      ObjectBBX aabb = findAABBFromCloud<pcl::PointXYZRGB>(obj_cloud_ptr);
      ObjectBBX obb = findOBBFromCloud<pcl::PointXYZRGB>(obj_cloud_ptr);

      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "world";
      marker.ns = "object_bbx_linelist";
      marker.id = object_id;
      marker.type = visualization_msgs::msg::Marker::LINE_LIST;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.header.stamp = node_->now();
      marker.scale.x = 0.1;

      marker.pose.orientation.x = 0;
      marker.pose.orientation.y = 0;
      marker.pose.orientation.z = 0;
      marker.pose.orientation.w = 1;

      marker.color.r = colour(0) / 255.0;
      marker.color.g = colour(1) / 255.0;
      marker.color.b = colour(2) / 255.0;
      marker.color.a = 1;

      for (const pcl::PointXYZ& this_line_list_point :
           findLineListPointsFromAABBMinMax(aabb.min_bbx_point_,
                                            aabb.max_bbx_point_)) {
        geometry_msgs::msg::Point p;
        p.x = this_line_list_point.x;
        p.y = this_line_list_point.y;
        p.z = this_line_list_point.z;
        marker.points.push_back(p);
      }
      object_bbx_linelist_marker_array.markers.push_back(marker);

      visualization_msgs::msg::Marker bbx_marker;
      bbx_marker.header.frame_id = "world";
      bbx_marker.ns = "object_bbx";
      bbx_marker.id = object_id;
      bbx_marker.type = visualization_msgs::msg::Marker::CUBE;
      bbx_marker.action = visualization_msgs::msg::Marker::ADD;
      bbx_marker.header.stamp = node_->now();
      bbx_marker.scale.x = obb.max_bbx_point_.x() - obb.min_bbx_point_.x();
      bbx_marker.scale.y = obb.max_bbx_point_.y() - obb.min_bbx_point_.y();
      bbx_marker.scale.z = obb.max_bbx_point_.z() - obb.min_bbx_point_.z();

      bbx_marker.pose.position.x = obb.bbx_position_.x();
      bbx_marker.pose.position.y = obb.bbx_position_.y();
      bbx_marker.pose.position.z = obb.bbx_position_.z();

      const gtsam::Quaternion& q = obb.orientation_.toQuaternion();
      bbx_marker.pose.orientation.x = q.x();
      bbx_marker.pose.orientation.y = q.y();
      bbx_marker.pose.orientation.z = q.z();
      bbx_marker.pose.orientation.w = q.w();

      bbx_marker.color.r = colour(0) / 255.0;
      bbx_marker.color.g = colour(1) / 255.0;
      bbx_marker.color.b = colour(2) / 255.0;
      bbx_marker.color.a = 0.2;

      object_bbx_marker_array.markers.push_back(bbx_marker);
    }
    object_bbx_line_pub_->publish(object_bbx_linelist_marker_array);
    object_bbx_pub_->publish(object_bbx_marker_array);
  }

  // Predict 3D object poses
  gtsam::FastMap<ObjectId, std::vector<gtsam::Pose3> > obj_predicted_poses;
  {
    int prediction_length = 3;
    const FrameId current_frame_id = rgbd_frontend_output->getFrameId();
    const ObjectPoseMap& obj_poses =
        rgbd_frontend_output->propogated_object_poses_;
    const MotionEstimateMap& obj_motions =
        rgbd_frontend_output->estimated_motions_;
    for (const auto& [object_id, this_obj_traj] : obj_poses) {
      if (this_obj_traj.exists(current_frame_id)) {
        gtsam::Pose3 current_obj_pose = this_obj_traj.at(current_frame_id);
        if (obj_motions.exists(object_id)) {
          gtsam::Pose3 current_obj_motion = obj_motions.at(object_id);

          gtsam::Pose3 last_object_pose = current_obj_pose;
          std::vector<gtsam::Pose3> predicted_poses;

          for (int i_prediction = 0; i_prediction < prediction_length;
               i_prediction++) {
            predicted_poses.push_back(last_object_pose);
            last_object_pose = current_obj_motion * last_object_pose;
            last_object_pose = current_obj_motion * last_object_pose;
            // last_object_pose = current_obj_motion * last_object_pose;
          }

          obj_predicted_poses.insert2(object_id, predicted_poses);
        }
      }
    }
  }

  // Write object motion to nav_msgs/Path and publish
  nav_msgs::msg::Path object_motions_msg;
  {
    const FrameId current_frame_id = rgbd_frontend_output->getFrameId();
    const MotionEstimateMap& obj_motions =
        rgbd_frontend_output->estimated_motions_;

    // object_motions_msg.header.seq = current_frame_id;
    object_motions_msg.header.stamp = node_->now();
    object_motions_msg.header.frame_id = params_.odom_frame;

    for (const auto& [object_id, current_obj_motion] : obj_motions) {
      geometry_msgs::msg::PoseStamped current_obj_motion_msg;
      current_obj_motion_msg.header.stamp = node_->now();
      current_obj_motion_msg.header.frame_id = std::to_string(object_id);
      current_obj_motion_msg.pose.position.x =
          current_obj_motion.estimate_.translation().x();
      current_obj_motion_msg.pose.position.y =
          current_obj_motion.estimate_.translation().y();
      current_obj_motion_msg.pose.position.z =
          current_obj_motion.estimate_.translation().z();
      current_obj_motion_msg.pose.orientation.w =
          current_obj_motion.estimate_.rotation()
              .toQuaternion()
              .w();  // in w x y z form
      current_obj_motion_msg.pose.orientation.x =
          current_obj_motion.estimate_.rotation()
              .toQuaternion()
              .x();  // in w x y z form
      current_obj_motion_msg.pose.orientation.y =
          current_obj_motion.estimate_.rotation()
              .toQuaternion()
              .y();  // in w x y z form
      current_obj_motion_msg.pose.orientation.z =
          current_obj_motion.estimate_.rotation()
              .toQuaternion()
              .z();  // in w x y z form

      object_motions_msg.poses.push_back(current_obj_motion_msg);
    }

    object_motion_pub_->publish(object_motions_msg);
  }

  // 2D image visualisation using opencv (TODO: and check that camera is
  // given!!)
  if (rgbd_frontend_output->debug_imagery_) {
    cv::Mat rgb_with_bbx;
    rgbd_frontend_output->debug_imagery_->tracking_image.copyTo(rgb_with_bbx);
    const cv::Size rgb_size = rgb_with_bbx.size();

    const StatusKeypointVector& obj_px =
        rgbd_frontend_output->dynamic_keypoint_measurements_;
    for (const KeypointStatus& this_px : obj_px) {
      ObjectId object_id = this_px.objectId();
      const gtsam::Point2& px_coordinate = this_px.value();

      const cv::Scalar colour = Color::uniqueId(object_id).bgra();
      // const cv::Scalar colour_bgr(colour[2], colour[1], colour[0]);
      cv::Point centre(functional_keypoint::u(px_coordinate),
                       functional_keypoint::v(px_coordinate));
      int radius = 1;
      int thickness = 1;
      cv::circle(rgb_with_bbx, centre, radius, colour, thickness);
    }

    // object history and prediction
    const ObjectPoseMap& obj_poses =
        rgbd_frontend_output->propogated_object_poses_;
    const gtsam::Pose3& cam_pose = rgbd_frontend_output->T_world_camera_;
    int line_thinkness = 3;
    for (const auto& [object_id, this_obj_traj] : obj_poses) {
      cv::Scalar colour = Color::uniqueId(object_id).bgra();
      // const cv::Scalar colour_bgr(colour[2], colour[1], colour[0]);
      std::vector<cv::Point> cv_line;
      for (const auto& [frame_id, this_obj_pose] : this_obj_traj) {
        gtsam::Pose3 this_obj_pose_in_cam = cam_pose.inverse() * this_obj_pose;
        Landmark this_obj_position_in_cam = this_obj_pose_in_cam.translation();
        Keypoint this_obj_px_in_cam;
        bool is_lmk_contained =
            rgbd_frontend_output->camera_->isLandmarkContained(
                this_obj_position_in_cam, &this_obj_px_in_cam);
        if (is_lmk_contained) {
          cv_line.push_back(
              cv::Point(functional_keypoint::u(this_obj_px_in_cam),
                        functional_keypoint::v(this_obj_px_in_cam)));
        }
      }
      int line_length = cv_line.size();
      for (int i_line = 0; i_line < line_length - 1; i_line++) {
        cv::line(rgb_with_bbx, cv_line[i_line], cv_line[i_line + 1], colour,
                 line_thinkness, cv::LINE_AA);
      }

      if (!obj_predicted_poses.exists(object_id)) {
        continue;
      }

      cv_line.clear();
      std::vector<gtsam::Pose3> this_obj_predicted_poses =
          obj_predicted_poses.at(object_id);
      for (const auto& this_obj_pose : this_obj_predicted_poses) {
        gtsam::Pose3 this_obj_pose_in_cam = cam_pose.inverse() * this_obj_pose;
        Landmark this_obj_position_in_cam = this_obj_pose_in_cam.translation();
        Keypoint this_obj_px_in_cam;
        bool is_lmk_contained =
            rgbd_frontend_output->camera_->isLandmarkContained(
                this_obj_position_in_cam, &this_obj_px_in_cam);
        if (is_lmk_contained) {
          cv_line.push_back(
              cv::Point(functional_keypoint::u(this_obj_px_in_cam),
                        functional_keypoint::v(this_obj_px_in_cam)));
        }
      }
      line_length = cv_line.size();
      // TODO: removed prediction arrow for now!!
      //  for (int i_line = 0; i_line < line_length-1; i_line++){
      //      double alpha = (double) i_line / ((double) line_length);
      //      cv::Scalar grad_colour = colour;
      //      grad_colour[0] = (255.0 - colour[0]) * alpha + colour[0];
      //      grad_colour[1] = (255.0 - colour[0]) * alpha + colour[1];
      //      grad_colour[2] = (255.0 - colour[0]) * alpha + colour[2];
      //      cv::arrowedLine(rgb_with_bbx, cv_line[i_line], cv_line[i_line+1],
      //      grad_colour, line_thinkness-1, cv::LINE_AA, 0, 0.3);
      //  }
    }

    // cv::resize(rgb_with_bbx, rgb_with_bbx, cv::Size(1280, 720) , 0, 0,
    // CV_INTER_LINEAR); if(video_writer_)  {
    //     CHECK(video_writer_->isOpened());
    //     video_writer_->write(rgb_with_bbx);
    // }

    cv::imshow("RGB with object bounding boxes", rgb_with_bbx);

    int frame_id = rgbd_frontend_output->getFrameId();
    // cv::Mat rgb_objects;
    // rgbd_frontend_output->frame_.tracking_images_.cloneImage<ImageType::RGBMono>(rgb_objects);

    // cv::imwrite("/root/results/RSS/"+std::to_string(frame_id)+".png",
    // rgb_objects);
  }

  {
    visualization_msgs::msg::MarkerArray object_pred_marker_array;

    for (const auto& [object_id, this_obj_predicted_poses] :
         obj_predicted_poses) {
      const cv::Scalar colour = Color::uniqueId(object_id);
      int pose_length = this_obj_predicted_poses.size();
      for (int i_pose = 0; i_pose < pose_length - 1; i_pose++) {
        gtsam::Point3 point_0 = this_obj_predicted_poses[i_pose].translation();
        gtsam::Point3 point_1 =
            this_obj_predicted_poses[i_pose + 1].translation();

        visualization_msgs::msg::Marker arrow_marker;
        arrow_marker.header.frame_id = "world";
        arrow_marker.ns = "object_pred";
        arrow_marker.id = object_id * 10 + i_pose;
        arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
        arrow_marker.action = visualization_msgs::msg::Marker::ADD;
        arrow_marker.header.stamp = node_->now();
        arrow_marker.scale.x = 0.3;
        arrow_marker.scale.y = 0.5;
        arrow_marker.scale.z = 0.8;

        double alpha = (double)i_pose / ((double)pose_length);
        arrow_marker.color.r =
            ((255.0 - colour[0]) * alpha + colour[0]) / 255.0;
        arrow_marker.color.g =
            ((255.0 - colour[1]) * alpha + colour[1]) / 255.0;
        arrow_marker.color.b =
            ((255.0 - colour[2]) * alpha + colour[2]) / 255.0;
        arrow_marker.color.a = 1;

        geometry_msgs::msg::Point p0, p1;
        p0.x = point_0.x();
        p0.y = point_0.y();
        p0.z = point_0.z();
        p1.x = point_1.x();
        p1.y = point_1.y();
        p1.z = point_1.z();
        arrow_marker.points.push_back(p0);
        arrow_marker.points.push_back(p1);
        object_pred_marker_array.markers.push_back(arrow_marker);
      }
    }
    object_bbx_line_pub_->publish(object_pred_marker_array);
  }
}

void FrontendInbuiltDisplayRos::publishOdometry(
    const gtsam::Pose3& T_world_camera, Timestamp timestamp) {
  InbuiltDisplayCommon::publishOdometry(odometry_pub_, T_world_camera,
                                        timestamp);
  geometry_msgs::msg::TransformStamped t;
  // utils::convertWithHeader(T_world_camera, t, timestamp,
  // params_.odom_frame, params_.camera_frame_); Send the transformation
  dyno::convert<gtsam::Pose3, geometry_msgs::msg::TransformStamped>(
      T_world_camera, t);

  t.header.stamp = node_->now();
  t.header.frame_id = params_.odom_frame;
  t.child_frame_id = params_.camera_frame;

  tf_broadcaster_->sendTransform(t);
}

// void FrontendInbuiltDisplayRos::publishOdometryPath(const gtsam::Pose3&
// T_world_camera, Timestamp timestamp) {
//     geometry_msgs::msg::PoseStamped pose_stamped;
//     utils::convertWithHeader(T_world_camera, pose_stamped, timestamp,
//     "world");

//     static std_msgs::msg::Header header;
//     header.stamp = utils::toRosTime(timestamp);
//     header.frame_id = "world";
//     odom_path_msg_.header = header;

//     odom_path_msg_.poses.push_back(pose_stamped);
//     odometry_path_pub_->publish(odom_path_msg_);

// }

void FrontendInbuiltDisplayRos::publishDebugImage(
    const DebugImagery& debug_imagery) {
  if (debug_imagery.tracking_image.empty()) return;

  // cv::Mat resized_image;
  // cv::resize(debug_image, resized_image, cv::Size(640, 480));

  std_msgs::msg::Header hdr;
  sensor_msgs::msg::Image::SharedPtr msg =
      cv_bridge::CvImage(hdr, "bgr8", debug_imagery.tracking_image)
          .toImageMsg();
  tracking_image_pub_.publish(msg);
}

// TODO: lots of repeated code with this funyction and publishObjectPositions -
// need to functionalise
void FrontendInbuiltDisplayRos::publishGroundTruthInfo(
    Timestamp timestamp, const GroundTruthInputPacket& gt_packet,
    const cv::Mat& rgb) {
  // odometry gt
  const gtsam::Pose3& T_world_camera = gt_packet.X_world_;
  nav_msgs::msg::Odometry odom_msg;
  utils::convertWithHeader(T_world_camera, odom_msg, timestamp, "world",
                           "camera");
  gt_odometry_pub_->publish(odom_msg);

  const auto frame_id = gt_packet.frame_id_;

  // odom path gt
  geometry_msgs::msg::PoseStamped pose_stamped;
  utils::convertWithHeader(T_world_camera, pose_stamped, timestamp, "world");
  static std_msgs::msg::Header header;
  header.stamp = utils::toRosTime(timestamp);
  header.frame_id = "world";
  gt_odom_path_msg_.header = header;
  gt_odom_path_msg_.poses.push_back(pose_stamped);

  gt_odom_path_pub_->publish(gt_odom_path_msg_);

  // prepare display image
  cv::Mat disp_image;
  rgb.copyTo(disp_image);

  static ObjectPoseMap gt_object_poses;
  for (const auto& object_pose_gt : gt_packet.object_poses_) {
    gt_object_poses.insert22(object_pose_gt.object_id_, gt_packet.frame_id_,
                             object_pose_gt.L_world_);
  }

  publishObjectPositions(gt_object_pose_pub_, gt_object_poses, frame_id,
                         timestamp, "ground_truth");

  publishObjectPaths(gt_object_path_pub_, gt_object_poses, frame_id, timestamp,
                     "ground_truth", 60);

  // static std::map<ObjectId, gtsam::Pose3Vector> gt_object_trajectories;
  // //Used for gt path updating static std::map<ObjectId, FrameId>
  // gt_object_trajectories_update; // The last frame id that the object was
  // seen in

  // std::set<ObjectId> seen_objects;

  // //prepare gt object pose markers
  // visualization_msgs::msg::MarkerArray object_pose_marker_array;
  // visualization_msgs::msg::MarkerArray object_path_marker_array;
  // static visualization_msgs::msg::Marker delete_marker;
  // delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

  // object_pose_marker_array.markers.push_back(delete_marker);
  // object_path_marker_array.markers.push_back(delete_marker);

  // for(const auto& object_pose_gt : gt_packet.object_poses_) {
  //     // const gtsam::Pose3 L_world = T_world_camera *
  //     object_pose_gt.L_camera_; const gtsam::Pose3 L_world =
  //     object_pose_gt.L_world_; const ObjectId object_id =
  //     object_pose_gt.object_id_;

  //     seen_objects.insert(object_id);

  //     visualization_msgs::msg::Marker marker;
  //     marker.header.frame_id = "world";
  //     marker.ns = "ground_truth_object_poses";
  //     marker.id = object_id;
  //     marker.type = visualization_msgs::msg::Marker::CUBE;
  //     marker.action = visualization_msgs::msg::Marker::ADD;
  //     marker.header.stamp = node_->now();
  //     marker.pose.position.x = L_world.x();
  //     marker.pose.position.y = L_world.y();
  //     marker.pose.position.z = L_world.z();
  //     marker.pose.orientation.x = L_world.rotation().toQuaternion().x();
  //     marker.pose.orientation.y = L_world.rotation().toQuaternion().y();
  //     marker.pose.orientation.z = L_world.rotation().toQuaternion().z();
  //     marker.pose.orientation.w = L_world.rotation().toQuaternion().w();
  //     marker.scale.x = 0.5;
  //     marker.scale.y = 0.5;
  //     marker.scale.z = 0.5;
  //     marker.color.a = 1.0; // Don't forget to set the alpha!

  //     const cv::Scalar colour = Color::uniqueId(object_id);
  //     marker.color.r = colour(0)/255.0;
  //     marker.color.g = colour(1)/255.0;
  //     marker.color.b = colour(2)/255.0;

  //     visualization_msgs::msg::Marker text_marker = marker;
  //     text_marker.ns = "ground_truth_object_labels";
  //     text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  //     text_marker.text = std::to_string(object_id);
  //     text_marker.pose.position.z += 1.0; //make it higher than the pose
  //     marker text_marker.scale.z = 0.7;

  //     object_pose_marker_array.markers.push_back(marker);
  //     object_pose_marker_array.markers.push_back(text_marker);

  //     //draw on bbox
  //     object_pose_gt.drawBoundingBox(disp_image);

  //     //update past trajectotries of objects
  //     auto it = gt_object_trajectories.find(object_id);
  //     if(it == gt_object_trajectories.end()) {
  //         gt_object_trajectories[object_id] = gtsam::Pose3Vector();
  //     }

  //     gt_object_trajectories_update[object_id] = frame_id;
  //     gt_object_trajectories[object_id].push_back(L_world);

  // }

  // //repeated code from publishObjectPositions function
  // //iterate over object trajectories and display the ones with enough poses
  // and the ones weve seen recently for(const auto& [object_id, poses] :
  // gt_object_trajectories) {
  //     const FrameId last_seen_frame =
  //     gt_object_trajectories_update.at(object_id);

  //     //if weve seen the object in the last 30 frames and the length is at
  //     least 2 if(poses.size() < 2u) {
  //         continue;
  //     }

  //     //draw a line list for viz
  //     visualization_msgs::msg::Marker line_list_marker;
  //     line_list_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  //     line_list_marker.header.frame_id = "world";
  //     line_list_marker.ns = "gt_frontend_composed_object_path";
  //     line_list_marker.id = object_id;
  //     line_list_marker.header.stamp = node_->now();
  //     line_list_marker.scale.x = 0.5;

  //     line_list_marker.pose.orientation.x = 0;
  //     line_list_marker.pose.orientation.y = 0;
  //     line_list_marker.pose.orientation.z = 0;
  //     line_list_marker.pose.orientation.w = 1;

  //     const cv::Scalar colour = Color::uniqueId(object_id);
  //     line_list_marker.color.r = colour(0)/255.0;
  //     line_list_marker.color.g = colour(1)/255.0;
  //     line_list_marker.color.b = colour(2)/255.0;
  //     line_list_marker.color.a = 1;

  //     //only draw the last 60 poses
  //     const size_t traj_size = std::min(60, static_cast<int>(poses.size()));
  //     // const size_t traj_size = poses.size();
  //     //have to duplicate the first in each drawn pair so that we construct a
  //     complete line for(size_t i = poses.size() - traj_size + 1; i <
  //     poses.size(); i++) {
  //         const gtsam::Pose3& prev_pose = poses.at(i-1);
  //         const gtsam::Pose3& curr_pose = poses.at(i);

  //         {
  //             geometry_msgs::msg::Point p;
  //             p.x = prev_pose.x();
  //             p.y = prev_pose.y();
  //             p.z = prev_pose.z();

  //             line_list_marker.points.push_back(p);
  //         }

  //         {
  //             geometry_msgs::msg::Point p;
  //             p.x = curr_pose.x();
  //             p.y = curr_pose.y();
  //             p.z = curr_pose.z();

  //             line_list_marker.points.push_back(p);
  //         }
  //     }

  //     object_path_marker_array.markers.push_back(line_list_marker);
  // }

  // Publish centroids of composed object poses
  // gt_object_pose_pub_->publish(object_pose_marker_array);

  // // Publish composed object path
  // gt_object_path_pub_->publish(object_path_marker_array);

  cv::Mat resized_image;
  cv::resize(disp_image, resized_image, cv::Size(640, 480));

  std_msgs::msg::Header hdr;
  sensor_msgs::msg::Image::SharedPtr msg =
      cv_bridge::CvImage(hdr, "bgr8", resized_image).toImageMsg();
  gt_bounding_box_pub_.publish(msg);
}

}  // namespace dyno
