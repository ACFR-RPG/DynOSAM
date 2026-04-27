from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

import os, copy, rclpy
import launch.logging


class DynosamNode(Node):
    """Custom Node that auto-builds gflags + dynamic args."""

    DEFAULT_ROS_PACKGE = "dynosam_ros"
    DEFAULT_EXECUTABLE_NAME = "dynosam_node"

    def __init__(self, **kwargs):

        if "package" not in kwargs:
            kwargs.update("package", DynosamNode.DEFAULT_ROS_PACKGE)

        if "executable" not in kwargs:
            kwargs.update("executable", DynosamNode.DEFAULT_EXECUTABLE_NAME)

        self._dyno_kwargs = kwargs
        super().__init__(**kwargs)
        self._logger = launch.logging.get_logger("dynosam_launch.DynoSAMNode")

    def _get_dynamic_gflags(self, context):
        """Builds the GFlag argument list at runtime."""

        def try_params_from_input_kwargs(key):
            from launch.substitutions.substitution_failure import SubstitutionFailure
            try:
                return LaunchConfiguration(key).perform(context)
            except SubstitutionFailure as e:
                # try from direct arguments in case the user has directlly specified the value
                # as part of the input arguments
                user_parameters = self._dyno_kwargs["parameters"]
                # user parameters is a list of key->value mappings
                # search through to find the key
                def find_value(data, key, default=None):
                    return next((d[key] for d in data if key in d), default)
                value_or_none = find_value(user_parameters, key)

                if value_or_none is None:
                    raise Exception(f"Could not get param '{key}' as was not specified as a LaunchConfiguration or in the input paramters")

                return value_or_none

        params_path = try_params_from_input_kwargs("params_path")
        verbose = try_params_from_input_kwargs("v")
        output_path = try_params_from_input_kwargs("output_path")

        flagfiles = [
            f"--flagfile={os.path.join(params_path, f)}"
            for f in os.listdir(params_path)
            if f.endswith(".flags")
        ]

        args = flagfiles + [f"--v={verbose}", f"--output_path={output_path}"]

        # add non-ROS args from CLI
        # should come from the LaunchContext
        all_argv = copy.deepcopy(context.argv)
        self._logger.info(f"All argv {all_argv}")
        non_ros_argv = rclpy.utilities.remove_ros_args(all_argv)
        if non_ros_argv:
            self._logger.info(f"Appending extra non-ROS argv: {non_ros_argv}")
            args.extend(non_ros_argv)
        return args

    def execute(self, context):
        actions = super().execute(context)
        self._logger.info("IN context")
        """Called by the LaunchService when this node is executed."""
        # Compute dynamic arguments
        gflags_args = self._get_dynamic_gflags(context)

        self._logger.info(f"Resolved DynoSAM gflags: {gflags_args}")

        # Merge any existing static arguments
        existing_args = self.cmd
        self._logger.info(f"Existing args: {existing_args}")

        # insert gflag commends at the start (immediately after the executable) so
        # any additional flags (ie. provided by arguments) may be overwritten
        self.cmd[1:1] = gflags_args

        # If needed, modify parameters dynamically
        # (you can even call LaunchConfiguration.perform(context) here)
        # For example:
        # params_path = LaunchConfiguration("params_path").perform(context)
        # self.parameters.append({"params_path": params_path})

        return actions
