from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration


from dynosam_ros.dynosam_node import DynosamNode

import os
import yaml

def generate_launch_description():
    rs_config_file = os.path.join(
        get_package_share_directory('dynosam_ros'),
        'config',
        'realsense_stereo.yaml'
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

    # dynosam_node = DynosamNode(
    #     package="dynosam_ros",
    #         executable="dynosam_node",
    #         output="screen",
    #         parameters=[
    #             {"params_path": dynosam_params},
    #             {"online": True},
    #             {"input_image_mode": 2}, # Corresponds with InputImageMode::Stereo}
    #             # {"baseline": 0.095},
    #             {"base_frame": "camera_link"},
    #             {"odom_frame": "odom"}
    #         ],
    #         remappings=[
    #             ('cam0/image_raw', f'{camera_prefix}/infra1/image_rect_raw'),
    #             ('cam1/image_raw', f'{camera_prefix}/infra2/image_rect_raw'),
    #             ('dataprovider/cam0/camera_info', f'{camera_prefix}/infra1/camera_info'),
    #             ('dataprovider/cam1/camera_info', f'{camera_prefix}/infra2/camera_info'),
    #         ],
    #     )

    dynosam_node = DynosamNode(
    package="dynosam_ros",
        executable="dynosam_node",
        output="screen",
        parameters=[
            {"params_path": dynosam_params},
            {"online": True},
            {"input_image_mode": "stereo"},
            # {"baseline": 0.095},
            {"base_frame": "camera_link"},
            {"odom_frame": "odom"},
            {"image_0_optical_frame": "camera_infra1_optical_frame"},
            {"image_1_optical_frame": "camera_infra2_optical_frame"}

        ],
        remappings=[
            ('image_0/image_raw', f'{camera_prefix}/infra1/image_rect_raw'),
            ('image_1/image_raw', f'{camera_prefix}/infra2/image_rect_raw'),
            ('dataprovider/image_0/camera_info', f'{camera_prefix}/infra1/camera_info'),
            ('dataprovider/image_1/camera_info', f'{camera_prefix}/infra2/camera_info'),
        ],
    )


    # wait a few seconds for the realsense to launch properly
    delay_node = TimerAction(
        period=2.0,
        actions=[dynosam_node]
    )

    return LaunchDescription([
        DeclareLaunchArgument("output_path", default_value="/root/results/misc/"),
        DeclareLaunchArgument("v", default_value="0"),
        rs_node,
        delay_node
    ])
