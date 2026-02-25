import launch.logging

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import OpaqueFunction, RegisterEventHandler, Shutdown, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit

def _get_utils_logger():
    return launch.logging.get_logger('dynosam_launch_utils.user')

def validate_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "path does not exist at {}".format(path)
        )
    return path

#NOTE: very helpful answers on ROS2 launch files: https://answers.ros.org/question/382000/ros2-makes-launch-files-crazy-too-soon-to-be-migrating/#:~:text=#!/usr/bin/env,condition=UnlessCondition(use_gui)%0A%20%20%20%20)%0A%0A%20%20])
def get_default_dynosam_params_path():
    #shoudl really be dynosam/<dataset>/example.flags like Kimera but we leave verbose for now
    dynosam_share_dir = get_package_share_directory('dynosam')

    logger = _get_utils_logger()
    logger.info("Loading default dynosam params folder path which is expected to be in the share directory of the dynosam package")

    if not os.path.exists(dynosam_share_dir):
        raise FileNotFoundError(
            "dynosam package share directory does not exist at path {}".format(dynosam_share_dir)
        )

    #check params folder has been correctly exported
    share_folders = os.listdir(dynosam_share_dir)
    if "params" not in share_folders:
        raise FileNotFoundError(
            "dynosam package share directory exists but 'params' folder does not exist at {}. "
            "Has the params folder been exported in the dynosam CMakeLists.txt like:\n"
            "\'install(DIRECTORY\n"
            "\tparams\n"
            "\tDESTINATION share/${PROJECT_NAME}\n"
            ")\'?".format(dynosam_share_dir)
        )

    return os.path.join(
        dynosam_share_dir,
        "params"
        ) + "/"


def load_dynosam_node(context, *args, **kwargs):
    """
    Constucts the set of launch actions for running a dynosam node - this includes
    a dynosam-style node and all the parameters that would be required to run the system, including ROS
    and GLOG params.

    This function is intended to be used within an OpaqueFunction in the form
    '
        LaunchDescription([OpaqueFunction(function=load_dynosam_node)])
    '
    where by default args=[] and kwargs={} of the OpaqueFunction, which are then parsed as the *args **kwargs
    of this fucntion. Args is unused. Kwargs are used to specify
        - executable: Executable file name (as in the node executable program to run). Default is dynosam_node.
        - should_output: if the program node (running the executable) should output to screen. Default is True.
        - world_to_robot_tf: If a static transform should be setup between the world and robot. Default is True.

    Context is parsed down from the LaunchService and is used to access the cmd line arguments from the user. These arguments
    should be the additional (non-ros) arguments that will be appended to the program Node arguments - this allows GLOG arguments
    to be parsed directly from the cmdline and into google::ParseCommandLineFlags(&non_ros_argc, &non_ros_argv_c, true)

    The function operates with the following logic
        - Load dynosam parameter path from the launch argument 'params_path'
        - Append flagfiles (files that end with .flags) found on the params_path to the arguments list.
            This allows any flag file stored on the params path to be loaded via GLOG
        - Append addional cmd line arguments from the context to the node arguments.
            This may include overwritten glog arguments which are manually specified
        - Get the executable name from kwargs and construct a launch_ros.actions.Node
            with package 'dynosam_ros', the parameters from dataset_path and params_path,
            as well as the additional arguments as above
        - Setup a tf transform between the world and robot (world to camera) if requested in the kwargs


    Args:
        context (_type_): _description_

    Returns:
        _type_: _description_
    """
    logger = _get_utils_logger()

    dynosam_dataset_path_config = LaunchConfiguration("dataset_path")
    dynosam_params_folder_config = LaunchConfiguration("params_path")
    glog_verbose_flag_config = LaunchConfiguration("v")
    online_config = LaunchConfiguration("online")
    wait_for_camera_params_config = LaunchConfiguration("wait_for_camera_params")
    camera_params_timeout_config = LaunchConfiguration("camera_params_timeout")
    output_path_config = LaunchConfiguration("output_path")

    # remap topics
    camera_info_config = LaunchConfiguration("camera_info")
    rgb_cam_topic_config = LaunchConfiguration("rgb_cam_topic")
    depth_cam_topic_config = LaunchConfiguration("depth_cam_topic")
    motion_mask_cam_topic_config = LaunchConfiguration("motion_mask_cam_topic")
    optical_flow_cam_topic_config = LaunchConfiguration("optical_flow_cam_topic")

    input_image_mode_config = LaunchConfiguration("input_image_mode")

    # additional cmdline arguments
    # args_config = LaunchConfiguration("argv")


    # Construct the flagfile arguments given the params folder
    # where we expect to find some files with the suffix .flags (for gflags)
    # these will be added to the arguments list as --flagfile="/path/to/flags/file.flags"
    def construct_flagfile_arguments(params_folder):
        arguments = []
        from pathlib import Path
        for file in os.listdir(params_folder):
            if Path(file).suffix == ".flags":
                arg = "--flagfile={}".format(os.path.join(params_folder, file))
                arguments.append(arg)
        return arguments

    def contruct_glags_verbose_argument(loaded_glog_verbose_flag):
        return "--v={}".format(loaded_glog_verbose_flag)

    # load programatically needed arguments
    loaded_params_folder = validate_path(dynosam_params_folder_config.perform(context))
    loaded_glog_verbose_flag = glog_verbose_flag_config.perform(context)
    loaded_output_path = output_path_config.perform(context)

    # print(args_config.perform(context))

    # arguments to the dynosam Node, these are passed as non-ros args.
    # these will (mostly) be used as GFLAG arguments
    arguments = construct_flagfile_arguments(loaded_params_folder)
    arguments.append(contruct_glags_verbose_argument(loaded_glog_verbose_flag))

    # context.argv is parsed in by the LaunchContext and will be the sys.argv[1:] (so not including the system name.)
    # we can use this to append to the actaul arguments we want to pass to the node. This is useful for
    # all the non-ros (ie GFLAGS/GLOG libs) we use  and allow us to modify this from otuside this launch file
    import rclpy
    import copy
    all_argv = copy.deepcopy(context.argv)
    non_ros_argv = rclpy.utilities.remove_ros_args(all_argv)

    # print(all_argv)
    # print(non_ros_argv)

    def list_difference(a, b):
        return [x for x in a if x not in b]
    # only_ros_argv = list_difference(all_argv, non_ros_argv)
    # argv = context.argv
    def construct_additional_arguments(argv):
        argv = list(argv)
        # if empty (the system name is already removed) then no args to add
        if len(argv) == 0:
            return None

        logger.info("Adding additional sys.argv arguements: {}".format(argv))
        return argv


    # add additional arguments - this will most often be any gflags passed along the command line
    additional_arguments = construct_additional_arguments(non_ros_argv)
    if additional_arguments:
        arguments.extend(additional_arguments)
    # put output path at end of additional arguments
    arguments.append("--output_path={}".format(loaded_output_path))
    # now add the ros args back to the end of the list
    # arguments.extend(only_ros_argv)

    logger.info("Adding arguments {}".format(arguments))

     # get details for the node to launch from the kwargs
    executable = kwargs.get("executable", "dynosam_node")
    logger.info("Setting executable to: {}".format(executable))

    node_kwargs = {}
    #specifies if the node should set output='screen' as part of the node params
    if kwargs.get("should_output", True):
        node_kwargs['output'] = 'screen'
        logger.info("Outputting to screen")

    nodes = []

    parameters=[
        {"params_folder_path": dynosam_params_folder_config},
        {"dataset_path": dynosam_dataset_path_config},
        {"online": online_config},
        {"wait_for_camera_params": wait_for_camera_params_config},
        {"camera_params_timeout": camera_params_timeout_config},
        {"input_image_mode": input_image_mode_config}
    ]

    is_online = bool(online_config.perform(context))
    # if we're offline we MUST set use_sim_time so that we use the /clock time
    # this is published in the ROS pipeline (also when is_online is false)
    if not is_online:
        parameters.append({"use_sim_time": True})

    program_node = Node(
        package='dynosam_ros',
        executable=executable,
        parameters=parameters,
        remappings=[
            ("dataprovider/camera/camera_info", camera_info_config),
            # ("dataprovider/image/rgb", rgb_cam_topic_config),
            # ("dataprovider/image/depth", depth_cam_topic_config),
            # ("dataprovider/image/mask", motion_mask_cam_topic_config),
            # ("dataprovider/image/flow", optical_flow_cam_topic_config)
            # for now the dataprovider sub-node is not being extended properly
            # but only when using the MultiSync class?
            ("image/rgb", rgb_cam_topic_config),
            ("image/depth", depth_cam_topic_config),
            ("image/mask", motion_mask_cam_topic_config),
            ("image/flow", optical_flow_cam_topic_config)
        ],
        # prefix='valgrind --tool=massif --massif-out-file=/tmp/massif.out',
        arguments=arguments,
        **node_kwargs
    )
    nodes.append(program_node)

    def on_node_completion_lambda(*args):
        reason_string = "Program node {} has exited".format(program_node.name)
        logger.info(reason_string + " - shutting down!")
        #NOTE: the reason text does not seem to propogate to the logger. I guess this is a bug with ROS2?
        action = Shutdown()
        action.visit(context)

    # set up handler to shutdown this context when the program node exists
    # this prevents any additional nodes hanging after the program finishes running
    nodes.append(RegisterEventHandler(OnProcessExit(
            target_action = program_node,
            on_exit = on_node_completion_lambda)))

    # if a static transform should be setup between the world and robot
    if kwargs.get("world_to_robot_tf", True):
        logger.info("Setting up world to robot transform")

        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=["0.0", "0.0", "0.0", "1.57", "-1.57", "0.0", "world", "robot"]
        ))

    return nodes

def declare_launch_arguments(default_dict, **kwargs):
    launch_args = []
    for key, internal_default in default_dict.items():
        default = kwargs.get(key, internal_default)
        launch_args.append(DeclareLaunchArgument(key, default_value=str(default)))

    # remove the above values from the kwargs so that they are not parsed to the load_dynosam_node function
    clean_kwargs = kwargs

    for provided_key in default_dict.keys():
        if provided_key in clean_kwargs:
            del clean_kwargs[provided_key]

    return launch_args, clean_kwargs

def inside_ros_launch():
    import sys
    argv = sys.argv
    exec = os.path.basename(argv[0])

    if exec == "ros2":
        command = argv[1]
        return command == "launch"
    return False



def generate_dynosam_launch_description(**kwargs):
    # these values can be overwritten by providing launch arguments of the same name, these
    # just provide the default if no launch arguments are given
    # defaults that can be changed via ROS args
    launch_args, clean_kwargs = declare_launch_arguments(
        {"dataset_path": "/root/data/VDO/kitti/kitti/0004",
         "v": 20,
         "params_path": get_default_dynosam_params_path(),
         "output_path": "/root/results/DynoSAM/",
         "online": False,
         "wait_for_camera_params": True,
         "camera_params_timeout": -1,
         "camera_info": "/dyno/camera/camera_info",
         "rgb_cam_topic": "/dyno/camera/rgb",
         "depth_cam_topic": "/dyno/camera/depth",
         "motion_mask_cam_topic": "/dyno/camera/motion_mask",
         "optical_flow_cam_topic": "/dyno/camera/optical_flow",
         "input_image_mode": 0},
         **kwargs)

    return LaunchDescription([
        # Must be inside launch description to be registered
        *launch_args,
        OpaqueFunction(function=load_dynosam_node, kwargs=clean_kwargs)
    ])
