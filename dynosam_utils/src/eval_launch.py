from launch import LaunchDescription  # noqa: E402
from launch import LaunchIntrospector  # noqa: E402
from launch import LaunchService  # noqa: E402
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory

import evaluation.evaluation_lib as eval

from typing import Optional, List
import logging as log
import errno
from shutil import rmtree, move
import os
import sys

def create_full_path_if_not_exists(file_path):
    if not os.path.exists(file_path):
        try:
            log.info('Creating non-existent path: %s' % file_path)
            os.makedirs(file_path)
            return True
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                log.fatal("Could not create inexistent filename: " + file_path)
            return False
    else:
        return True

def move_output_from_to(from_dir, to_dir):
    try:
        if (os.path.exists(to_dir)):
            rmtree(to_dir)
    except:
        log.info("Directory:" + to_dir + " does not exist, we can safely move output.")
    try:
        if (os.path.isdir(from_dir)):
            move(from_dir, to_dir)
        else:
            log.info("There is no output directory...")
    except:
        print("Could not move output from: " + from_dir + " to: " + to_dir)
        raise
    try:
        os.makedirs(from_dir)
    except:
        log.fatal("Could not mkdir: " + from_dir)

def ensure_dir(dir_path):
    """ Check if the path directory exists: if it does, returns true,
    if not creates the directory dir_path and returns if it was successful"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return True

#TODO: refactor to experimental yaml file like Kimera?
def parser():
    import argparse
    basic_desc = "TODO:"
    shared_parser = argparse.ArgumentParser(add_help=True, description="{}".format(basic_desc))

    # input to the dynosam launch file and additional arguements (if any)
    # these will match the LaunchConfiguration (ie. ROS args) arguments in the dynosam launch file
    input_opts = shared_parser.add_argument_group("input options")
    evaluation_opts = shared_parser.add_argument_group("algorithm options")

    #TODO: this needs to match a number of variables in the Yaml files... streamline
    input_opts.add_argument("-d", "--dataset_path", type=str,
                                help="Absolute path to the dataset to run.",
                                default="/root/data/virtual_kitti")

    # By default not required as the dynosam_launch.py file will look for default params folder
    # in the share/ folder of the installed package
    input_opts.add_argument("-p", "--params_path", type=str,
                                help="Absolute path to the params to dun Dynosam with.",
                                required=False)

    input_opts.add_argument("-l", "--launch_file", type=str,
                                help="Which dynosam launch file to run!.",
                                default="dyno_sam_launch.py")


    evaluation_opts.add_argument("-r", "--run_pipeline", action="store_true",
                                 help="Run dyno?")

    evaluation_opts.add_argument("-o", "--output_path",
                                 help="Output folder path to store the logs in.",
                                 default="/root/results/DynoSAM/")

    evaluation_opts.add_argument("-n", "--name",
                                 help="Name of the experiment to run. This will be appended to the output_path file"
                                 " such that the output file path will be output_path/name",
                                 default="test")

    evaluation_opts.add_argument("-a", "--run_analysis",
                                 help="Runs analysis on the output files, expected to be found in the output_path/name folder ",
                                 action="store_true")

    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    sub_parsers = main_parser.add_subparsers(dest="subcommand")
    sub_parsers.required = True
    return shared_parser


def run(parsed_args, unknown_args):
    """
    Runs the dynosam launch file with given input arguments.
    The input arguments are twofold:
    parsed_args is a dictionary of "cmd line" arguments that generally come from the argument parser.
    Parsed args must contain keys for
        - dataset_path: str
        - output_path: str
        - name: str
        - run_pipeline: bool
        - run_analysis: bool
        - launch_file: str
    and optionally
        - params_path: str
    This should be a dictioanry of values rather than an argument parser instance.

    unknown args are a list of additional arguments that will be parsed to the dynosam launch file
    and appended to the **argv array in the actual executable. These should include arguments we want to actually
    process in the program and should include additional gflags that we want to modify directly (e.g. in the form --flag=value)

    Args:
        parsed_args (_type_): _description_
        unknown_args (_type_): _description_

    Returns:
        _type_: _description_
    """
    if "dataset_path" not in parsed_args:
        log.fatal("dataset_path key is missing from parsed args!")

    if "output_path" not in parsed_args:
        log.fatal("output_path key is missing from parsed args!")

    if "name" not in parsed_args:
        log.fatal("name key is missing from parsed args!")

    if "run_pipeline" not in parsed_args:
        log.fatal("run_pipeline key is missing from parsed args!")

    if "run_analysis" not in parsed_args:
        log.fatal("run_analysis key is missing from parsed args!")

    if "launch_file" not in parsed_args:
        log.fatal("launch_file key is missing from parsed args!")

    datset_path = parsed_args["dataset_path"]
    output_path = parsed_args["output_path"]
    name = parsed_args["name"]
    run_pipeline = parsed_args["run_pipeline"]
    run_analysis = parsed_args["run_analysis"]
    launch_file = parsed_args["launch_file"]

    launch_arguments={'dataset_path': datset_path}

    # if params path is not provided, the dyno_sam_launch will search for this
    # on the default path anyway (as specified in the launch file)
    if "params_path" in parsed_args and parsed_args["params_path"]:
        launch_arguments.update({"params_path": parsed_args["params_path"]})


    log.info(f"Running lauch using dynosam_ros launch file {launch_file}")

    dynosam_launch_file = os.path.join(
            get_package_share_directory('dynosam_ros'), 'launch', launch_file)

    ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([dynosam_launch_file]),
        launch_arguments=launch_arguments.items(),
    )

    output_folder_path = os.path.join(output_path, name)
    create_full_path_if_not_exists(output_folder_path)

    unknown_args.append("--output_path={}".format(output_folder_path))

    running_success = True
    if run_pipeline:
        # parse arguments down to the actual launch file
        # these will be appended to the Node(arguments=...) parameter and can be used
        # to directly change things like FLAG_ options etc...
        ls = LaunchService(argv=unknown_args)
        ls.include_launch_description(ld)
        running_success = ls.run()


    if run_analysis:
        evaluator =  eval.DatasetEvaluator(output_folder_path, parsed_args)
        evaluator.run_analysis()

    return running_success

#python3 eval_launch.py --dataset_path=/root/data/vdo_slam/kitti/kitti/0004 --name logging

import argcomplete


if __name__ == '__main__':
    ###
    parser = parser()
    argcomplete.autocomplete(parser)
    # args will be the arguments known to the parser and should define options for how to run the
    # pipeline. unknown are cmdline args not registered to the parser and will be additional arguments that
    # the user wants to parse to the LaunchService as real args to be used as GFLAGS.
    args, unknown = parser.parse_known_args()
    args_dictionary = vars(args)

    sys.exit(run(args_dictionary, unknown))
