from dynosam_ros.DynosamNode import DynoSAMNode
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
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/color/camera_info"),
        DeclareLaunchArgument("rgb_cam_topic", default_value="/camera/color/image_rect_color"),
        DeclareLaunchArgument("depth_cam_topic", default_value="/camera/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("rescale_width", default_value="640", description="Image width to rescale to"),
        DeclareLaunchArgument("rescale_height", default_value="480", description="Image height to rescale to"),

        DynoSAMNode(
                package="dynosam_ros",
                executable="dynosam_node",
                output="screen",
                parameters=[
                    {"params_folder_path": LaunchConfiguration("params_path")},
                    {"rescale_width": LaunchConfiguration("rescale_width")},
                    {"rescale_height": LaunchConfiguration("rescale_height")},
                    {"online": True},
                    {"input_image_mode": 1} # Corresponds with InputImageMode::RGBD}
                ],
                remappings=[
                    ("dataprovider/camera/camera_info",  LaunchConfiguration("camera_info_topic")),
                    ("image/rgb",  LaunchConfiguration("rgb_cam_topic")),
                    ("image/depth",  LaunchConfiguration("depth_cam_topic")),
                ]
            ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments=["0.0", "0.0", "0.0", "1.57", "-1.57", "0.0", "world", "robot"]
            )
        ])
