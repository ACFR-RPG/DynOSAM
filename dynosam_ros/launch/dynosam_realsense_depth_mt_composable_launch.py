from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, Node, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration


from dynosam_ros.dynosam_node import DynosamNode

import os
import yaml

def generate_launch_description():
    rs_config_file = os.path.join(
        get_package_share_directory('dynosam_ros'),
        'config',
        'realsense_depth.yaml'
    )

    camera_name = 'd455'
    camera_namespace = ''
    camera_prefix = '/' + '/'.join([segment for segment in [camera_namespace, camera_name] if segment])
    rs_node_name = camera_prefix



    with open(rs_config_file, 'r', encoding='utf-8') as config_handle:
        rs_config = yaml.safe_load(config_handle) or {}
    rs_node = ComposableNode(
            name=camera_name,
            package='realsense2_camera',
            namespace='',
            plugin='realsense2_camera::RealSenseNodeFactory',
            parameters=[rs_config],
            extra_arguments=[{'use_intra_process_comms': True}]
    )

    dynosam_params = os.path.join(
        get_package_share_directory('dynosam_ros'),
        'config',
        camera_name
    )

    container = ComposableNodeContainer(
        name='dynosam_launch_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='dynosam_ros',
                plugin='dyno::DynosamComposableNode',
                name='dynosam',
                parameters=[
                    {"params_path": dynosam_params},
                    {"online": True},
                    {"input_image_mode": "rgb+aligned_depth"},
                    {"base_frame": "camera_link"},
                    {"odom_frame": "odom"}
                ],
                remappings=[
                    ('rgb/image_raw', f'{camera_prefix}/color/image_raw'),
                    ('depth/image_raw', f'{camera_prefix}/aligned_depth_to_color/image_raw'),
                    ('rgb/camera_info', f'{camera_prefix}/color/camera_info'),
                    ("imu", f'{camera_prefix}/imu')
                ],
            ),

        ],
        output='screen',
    )

    realsense_container = LoadComposableNodes(
        target_container='dynosam_launch_container',
        composable_node_descriptions=[rs_node]
    )


    return LaunchDescription([
        DeclareLaunchArgument("output_path", default_value="/root/results/misc/"),
        DeclareLaunchArgument("v", default_value="30"),
        realsense_container,
        container

    ])
