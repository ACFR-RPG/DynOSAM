from dynosam_ros_common.dynosam_node import DynosamNode
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch_ros.actions import Node
from dynosam_ros_common.launch_utils import get_default_dynosam_params_path
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    # pkg_dir = get_package_share_directory('realsense2_description')
    # xacro_file = os.path.join(pkg_dir, 'urdf', '_d435.urdf.xacro')

    # Process xacro to string
    # robot_description_raw = xacro.process_file(xacro_file).toxml()

    # 2. Configure robot_state_publisher
    # robot_state_publisher_node = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='screen',
    #     parameters=[{'robot_description': robot_description_raw}]
    # )


    return LaunchDescription([
        DeclareLaunchArgument("params_path", default_value=get_default_dynosam_params_path("dynosam2")),
        DeclareLaunchArgument("v", default_value="10"),
        DeclareLaunchArgument("output_path", default_value="/root/results/DynoSAM/"),
        DeclareLaunchArgument("camera_info_topic", default_value="/d455/color/camera_info"),
        DeclareLaunchArgument("rgb_cam_topic", default_value="/d455/color/image_raw"),
        DeclareLaunchArgument("depth_cam_topic", default_value="/d455/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("imu_topic", default_value="/d455/imu"),
        DeclareLaunchArgument("depth_scale", default_value="0.001", description="RGB-Depth scale value"),
        DeclareLaunchArgument("rescale_width", default_value="640", description="Image width to rescale to"),
        DeclareLaunchArgument("rescale_height", default_value="480", description="Image height to rescale to"),
        DeclareLaunchArgument("odom_frame", default_value="odom", description="ROS odom frame"),
        DeclareLaunchArgument("base_frame", default_value="camera_link", description="ROS base frame (the estimation frame)"),
        # robot_state_publisher_node,
        DynosamNode(
                package="dynosam2_ros",
                executable="dynosam2_node",
                output="screen",
                parameters=[
                    {"params_path": LaunchConfiguration("params_path")},
                    {"rescale_width": LaunchConfiguration("rescale_width")},
                    {"rescale_height": LaunchConfiguration("rescale_height")},
                    {"online": True},
                    {"input_image_mode": "rgb+aligned_depth"},
                    {"base_frame": LaunchConfiguration("base_frame")},
                    {"odom_frame": LaunchConfiguration("odom_frame")},
                    {"depth_scale": LaunchConfiguration("depth_scale")},
                    {"v": LaunchConfiguration("v")}
                ],
                remappings=[
                    ("rgb/camera_info", LaunchConfiguration("camera_info_topic")),
                    ("rgb/image_raw",   LaunchConfiguration("rgb_cam_topic")),
                    ("depth/image_raw", LaunchConfiguration("depth_cam_topic")),
                    ("imu", LaunchConfiguration("imu_topic"))
                ]
            ),
        ])
