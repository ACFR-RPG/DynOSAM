from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument


from dynosam_ros_common.dynosam_node import DynosamNode

import os
import yaml

def generate_launch_description():
    rs_config_file = os.path.join(
        get_package_share_directory('dynosam2_ros'),
        'config',
        'realsense_depth.yaml'
    )

    camera_name = 'd455'
    camera_namespace = ''
    camera_prefix = '/' + '/'.join([segment for segment in [camera_namespace, camera_name] if segment])
    rs_node_name = camera_prefix



    with open(rs_config_file, 'r', encoding='utf-8') as config_handle:
        rs_config = yaml.safe_load(config_handle) or {}
    rs_node = Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name=camera_name,
            namespace=camera_namespace,
            output='screen',
            parameters=[rs_config],
            arguments=['--ros-args', '--log-level', 'info'],
        )

    dynosam_params = os.path.join(
        get_package_share_directory('dynosam_ros'),
        'config',
        camera_name
    )

    dynosam2_node = DynosamNode(
        package="dynosam2_ros",
            executable="dynosam2_node",
            output="screen",
            parameters=[
                {"params_path": dynosam_params},
                {"online": True},
                {"input_image_mode": "rgb+aligned_depth"},
                {"base_frame": "camera_link"},
                {"odom_frame": "odom"},
                {"enable_groundtruth_sub": False},

            ],
            remappings=[
                ('rgb/image_raw', f'{camera_prefix}/color/image_raw'),
                ('depth/image_raw', f'{camera_prefix}/aligned_depth_to_color/image_raw'),
                ('rgb/camera_info', f'{camera_prefix}/color/camera_info'),
                ("imu", f'{camera_prefix}/imu'),
                # ("ground_truth", "/mrl_vicon/ground_truth")
            ],
        )

    return LaunchDescription([
        DeclareLaunchArgument("output_path", default_value="/root/results/misc/"),
        DeclareLaunchArgument("v", default_value="50"),
        rs_node,
        dynosam2_node
    ])
