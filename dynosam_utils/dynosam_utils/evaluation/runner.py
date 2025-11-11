from launch import LaunchService  # noqa: E402
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory

import logging as log
import errno
from shutil import rmtree, move
import os

try:
    import better_launch
    print("Found better_launch package")
except ImportError:
    better_launch = None


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


    log.info(f"Running launch using dynosam_ros launch file {launch_file}")

    dynosam_launch_file = os.path.join(
            get_package_share_directory('dynosam_ros'), 'launch', launch_file)

    output_folder_path = os.path.join(output_path, name)
    create_full_path_if_not_exists(output_folder_path)

    # update launch arguments with fully constructed output path
    launch_arguments.update({"output_path": output_folder_path})

    log.info(f"Setting output folder path: {output_folder_path}")

    ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([dynosam_launch_file]),
        launch_arguments=launch_arguments.items(),
    )

    running_success = True
    if run_pipeline:
        log.info("Running pipeline...")
        # parse arguments down to the actual launch file
        # these will be appended to the Node(arguments=...) parameter and can be used
        # to directly change things like FLAG_ options etc...
        ls = LaunchService(argv=unknown_args)
        ls.include_launch_description(ld)
        running_success = ls.run()


    if run_analysis:
        log.info("Running analysis...")
        import dynosam_utils.evaluation.evaluation_lib as eval
        evaluator =  eval.DatasetEvaluator(output_folder_path, parsed_args)
        evaluator.run_analysis()

    if running_success:
        return 0
    else:
        return 1


if better_launch is not None:
    from better_launch import BetterLaunch


    def bl_run(
            name,
            output_path,
            dataset_path,
            run_pipeline=True,
            run_anaysis=True,
            **kwargs):
        bl = BetterLaunch()

        print(kwargs)

        output_folder_path = os.path.join(output_path, name)
        create_full_path_if_not_exists(output_folder_path)

        if run_pipeline:
            # NOTE: because bl_run is not marked as a lunch file
            # pass_launch_func_args does not work!
            bl.logger.info("Running dynosam pipeline")
            bl.include(
                "dynosam_ros",
                "bl_dynosam.launch.py",
                dataset_path=dataset_path,
                output_path=output_folder_path,
                **kwargs)

        if run_anaysis:
            bl.logger.info("Running dynosam analysis")
