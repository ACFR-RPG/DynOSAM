
from dynosam_ros.dynosam_node import DynosamNode
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch_ros.actions import Node


from dynosam_ros.launch_utils import get_default_dynosam_params_path

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("params_path", default_value=get_default_dynosam_params_path()),
        DeclareLaunchArgument("v", default_value="30"),
        DeclareLaunchArgument("output_path", default_value="/root/results/DynoSAM/"),
        DeclareLaunchArgument("dataset_path", default_value="/root/data/VDO/kitti/0004"),
        DynosamNode(
            package="dynosam_ros",
            executable="dynosam_node",
            output="screen",
            parameters=[
                {"dataset_path": LaunchConfiguration("dataset_path")},
                {"params_folder_path": LaunchConfiguration("params_path")},
                {"online": False},
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=["0.0", "0.0", "0.0", "1.57", "-1.57", "0.0", "world", "robot"]
        )
    ])
