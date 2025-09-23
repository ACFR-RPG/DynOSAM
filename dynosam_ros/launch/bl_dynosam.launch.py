#!/usr/bin/env python3
import os

from better_launch import BetterLaunch, launch_this

from ament_index_python.packages import get_package_share_directory
import sys

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

    if not os.path.exists(dynosam_share_dir):
        raise FileNotFoundError(
            "dynosam package share directory does not exist at path {}".format(dynosam_share_dir)
        )

    #check params folder has been correctly exported
    share_folders = os.listdir(dynosam_share_dir)
    if "params" not in share_folders:
        raise FileNotFoundError(
            "dynosam package share directory exists but \'params'\ folder does not exist at {}. "
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

def _append_flag_files(params_folder):
    arguments = []
    from pathlib import Path
    for file in os.listdir(params_folder):
        if Path(file).suffix == ".flags":
            arg = "--flagfile={}".format(os.path.join(params_folder, file))
            arguments.append(arg)
    return arguments

@launch_this
def first_steps(
    dataset_path,
    params_folder_path=None,
    output_path="/root/results/DynoSAM",
    online=False,
    wait_for_camera_params=True,
    camera_params_timeout=-1):

    bl = BetterLaunch()


    if params_folder_path is None:
        bl.logger.info("Loading default dynosam params folder path which is expected to be in the share directory of the dynosam package")
        params_folder_path = get_default_dynosam_params_path()

    params_folder_path = validate_path(params_folder_path)
    cmd_args = _append_flag_files(params_folder_path)

    params = {
            "dataset_path": dataset_path,
            "params_folder_path": params_folder_path,
            "output_path": output_path,
            "online": online,
            "wait_for_camera_params": wait_for_camera_params,
            "camera_params_timeout": camera_params_timeout
        }

    if online:
        params.update({"use_sim_time": True})

    # print(params)
    with bl.group("dynosam"):
        bl.node(
            "dynosam_ros",
            "dynosam_node",
            "dynosam",
            params=params,
            cmd_args=cmd_args,
            log_level = None,

            # remaps=remappings
        )

    # if params_folder_path:
    #     dynosam_node.set_live_params({"params_folder_path": params_folder_path})
