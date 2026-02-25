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

        super().__init__(**kwargs)
        self._logger = launch.logging.get_logger("dynosam_launch.DynoSAMNode")

    def _get_dynamic_gflags(self, context):
        """Builds the GFlag argument list at runtime."""
        params_path = LaunchConfiguration("params_path").perform(context)
        verbose = LaunchConfiguration("v").perform(context)
        output_path = LaunchConfiguration("output_path").perform(context)

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
        # self.parameters.append({"params_folder_path": params_path})

        return actions
