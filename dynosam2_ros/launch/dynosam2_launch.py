
from dynosam_ros_common.dynosam_node import DynosamNode
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch_ros.actions import Node


from dynosam_ros_common.launch_utils import get_default_dynosam_params_path

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("params_path", default_value=get_default_dynosam_params_path("dynosam2")),
        DeclareLaunchArgument("v", default_value="30"),
        DeclareLaunchArgument("output_path", default_value="/root/results/DynoSAM/"),
        DeclareLaunchArgument("dataset_path", default_value="/root/data/VDO/kitti/0004"),
        DynosamNode(
            package="dynosam2_ros",
            executable="dynosam2_node",
            output="screen",
            parameters=[
                {"dataset_path": LaunchConfiguration("dataset_path")},
                {"params_path": LaunchConfiguration("params_path")},
                {"online": False},
            ],
        ),
    ])
