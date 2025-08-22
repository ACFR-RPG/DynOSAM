from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Define arguments
    #TODO: (jesse) dont actually need these here as they are already defined in dynosam_launch_utils
    declared_arguments = [
        DeclareLaunchArgument('camera_info', default_value='/dyno/camera/camera_info', description='Camera info topic'),
        DeclareLaunchArgument('rgb_cam_topic', default_value='/dyno/camera/rgb', description='RGB camera topic'),
        DeclareLaunchArgument('depth_cam_topic', default_value='/dyno/camera/depth', description='Depth camera topic'),
        DeclareLaunchArgument('motion_mask_cam_topic', default_value='/dyno/camera/motion_mask', description='Motion mask camera topic'),
        DeclareLaunchArgument('optical_flow_cam_topic', default_value='/dyno/camera/optical_flow', description='Optical flow camera topic'),
        DeclareLaunchArgument('output_path', default_value='/output_path', description='Output path directory')
    ]

    # Get the child launch file
    dynosam_launch_file = os.path.join(
            get_package_share_directory('dynosam_ros'), 'launch', 'dyno_sam_launch.py')
    # Include the child launch file and pass all arguments
    child_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(dynosam_launch_file),
        launch_arguments={
            'camera_info': LaunchConfiguration('camera_info'),
            'rgb_cam_topic': LaunchConfiguration('rgb_cam_topic'),
            'depth_cam_topic': LaunchConfiguration('depth_cam_topic'),
            'motion_mask_cam_topic': LaunchConfiguration('motion_mask_cam_topic'),
            'optical_flow_cam_topic': LaunchConfiguration('optical_flow_cam_topic'),
            'online': 'True',
            'output_path': LaunchConfiguration('output_path')
        }.items()
    )

    return LaunchDescription(declared_arguments + [child_launch])
