/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#pragma once

#include <dynosam/visualizer/FrontendDisplay.hpp>
#include <dynosam/common/GroundTruthPacket.hpp>
#include <dynosam/frontend/RGBDInstance-Definitions.hpp>

#include "image_transport/image_transport.hpp"


#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"


namespace dyno {

class FrontendDisplayRos : public FrontendDisplay {
public:
    FrontendDisplayRos(rclcpp::Node::SharedPtr node);

    void spinOnce(const FrontendOutputPacketBase::ConstPtr& frontend_output) override;

private:
    void processRGBDOutputpacket(const RGBDInstanceOutputPacket::ConstPtr& rgbd_frontend_output);

    void publishStaticCloud(const Landmarks& static_landmarks);
    void publishObjectCloud(const StatusKeypointMeasurements& dynamic_measurements, const Landmarks& dynamic_landmarks);
    void publishObjectPositions(const std::map<ObjectId, gtsam::Pose3>& propogated_object_poses);

    // void publishVisibleCloud(const FrontendOutputPacketBase& frontend_output);
    void publishOdometry(const gtsam::Pose3& T_world_camera);
    void publishDebugImage(const cv::Mat& debug_image);

    void publishGroundTruthInfo(const GroundTruthInputPacket& gt_packet, const cv::Mat& rgb);

private:
    rclcpp::Node::SharedPtr node_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr static_tracked_points_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_tracked_points_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;

    //sort of just for now
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr object_pose_pub_;
    image_transport::Publisher tracking_image_pub_;

    //ground truth publishers
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gt_object_pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr gt_odometry_pub_;
    image_transport::Publisher gt_bounding_box_pub_;

};

}
