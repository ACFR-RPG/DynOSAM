from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():


    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('dynosam2_ros'),
                    'launch',
                    'dynosam2_online_rgb_launch.py',
                ])
            ),
            launch_arguments={
                # 'base_frame': "zed_camera_link",
                'base_frame': "zed_camera_link",
                'depth_scale': "1.0",
                'imu_topic': "/zed/zed_node/imu/data",
                'camera_info_topic': "/zed/zed_node/rgb/color/rect/camera_info",
                'rgb_cam_topic': "/zed/zed_node/rgb/color/rect/image",
                'depth_cam_topic': "/zed/zed_node/depth/depth_registered"
            }.items(),
        ),
    ])
