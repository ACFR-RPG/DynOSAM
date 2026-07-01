from dynosam_ros.dynosam_node import DynosamNode
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch_ros.actions import Node
from dynosam_ros.launch_utils import get_default_dynosam_params_path
from ament_index_python.packages import get_package_share_directory
import xacro

import os


def generate_launch_description():
    # pkg_dir = get_package_share_directory('realsense2_description')
    # xacro_file = os.path.join(pkg_dir, 'urdf', 'test_d435i_camera.urdf.xacro')

    # # Process xacro to string
    # robot_description_raw = xacro.process_file(
    #     xacro_file,
    #     mappings={'use_nominal_extrinsics': 'true'}
    # ).toxml()
    # print(robot_description_raw)

    # # 2. Configure robot_state_publisher
    # robot_state_publisher_node = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='screen',
    #     parameters=[{'robot_description': robot_description_raw}]
    # )

    return LaunchDescription([
        DeclareLaunchArgument("params_path", default_value=get_default_dynosam_params_path()),
        DeclareLaunchArgument("v", default_value="30"),
        DeclareLaunchArgument("output_path", default_value="/root/results/DynoSAM/"),
        DeclareLaunchArgument("camera_info_topic", default_value="/d455/color/camera_info"),
        DeclareLaunchArgument("rgb_cam_topic", default_value="/d455/color/image_raw"),
        DeclareLaunchArgument("depth_cam_topic", default_value="/d455/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("mask_cam_topic", default_value="/d455/color/mask"),
        DeclareLaunchArgument("rescale_width", default_value="640", description="Image width to rescale to"),
        DeclareLaunchArgument("rescale_height", default_value="480", description="Image height to rescale to"),
        DeclareLaunchArgument("base_frame", default_value="camera_link",
                              description="Parent of camera optical frame in TF tree (Z-up, robotics convention)"),
        DeclareLaunchArgument("odom_frame", default_value="odom",
                              description="Odometry/world frame for DynoSAM output topics and TF"),
        # DeclareLaunchArgument("depth_scale", default_value="0.001",
        #                       description="Depth image scale factor: 1.0 for metre sources (Gazebo, ZED), "
        #                                   "0.001 for millimetre sources (some RealSense configs)."),
        # robot_state_publisher_node,
        DynosamNode(
                package="dynosam_ros",
                executable="dynosam_node",
                output="screen",
                parameters=[
                    {"params_path": LaunchConfiguration("params_path")},
                    {"rescale_width": LaunchConfiguration("rescale_width")},
                    {"rescale_height": LaunchConfiguration("rescale_height")},
                    {"online": True},
                    {"input_image_mode": "rgb+aligned_depth+aligned_mask"},
                    {"base_frame":  LaunchConfiguration("base_frame")},
                    {"odom_frame":  LaunchConfiguration("odom_frame")},
                    {"rgb_optical_frame": "camera_color_optical_frame"},
                    # {"depth_scale": LaunchConfiguration("depth_scale")},
                    # {"baseline": 0.05},
                    {"v": LaunchConfiguration("v")}
                ],
                remappings=[
                    ("rgb/camera_info", LaunchConfiguration("camera_info_topic")),
                    ("rgb/image_raw",   LaunchConfiguration("rgb_cam_topic")),
                    ("depth/image_raw", LaunchConfiguration("depth_cam_topic")),
                    ("mask/image_raw",  LaunchConfiguration("mask_cam_topic")),
                ]
            ),
        ])
