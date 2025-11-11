#!/usr/bin/python3


import os
import argparse
import subprocess

from ament_index_python.packages import get_package_prefix

def find_ros2_package_path(package_name):
    """
    Find the installation path of a ROS 2 package.

    Args:
        package_name (str): Name of the ROS 2 package.

    Returns:
        str: The full path to the package's install directory.

    Raises:
        ValueError: If the package is not found.
    """
    try:
        # Get the package install directory
        package_path = get_package_prefix(package_name)
        return package_path
    except ValueError as e:
        raise ValueError(f"Package '{package_name}' not found.") from e

def run_executable(executable_path, args):
    """
    Run an executable with given command-line arguments.

    Args:
        executable_path (str): Path to the executable file.
        args (list): List of command-line arguments to pass.

    Returns:
        int: The return code of the executable.
    """
    try:
        print(f"Running executable {executable_path}")
        # Combine the executable and arguments into a single command
        command = [executable_path] + args

        # Run the command and wait for it to finish
        result = subprocess.run(command, check=True, text=True)

        # Print standard output and error
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)

        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{e.cmd}' failed with return code {e.returncode}")
        print("Standard Output:\n", e.stdout)
        print("Standard Error:\n", e.stderr)
        return e.returncode
    except FileNotFoundError:
        print(f"Error: Executable '{executable_path}' not found.")
        return -1


def run_tests_for_package(package_name, unknown_args):
    package_path = find_ros2_package_path(package_name)

    print(f"Found package path for package {package_path}")
    tests_path = package_path + "/test/"


    # test that path exists
    if not os.path.exists(tests_path) or not os.path.isdir(tests_path):
        raise ValueError(f"Test folder is not valid or cannot be found at path {tests_path}")

    # find possible test exeutables
    executables = []
    for file_name in os.listdir(tests_path):
        full_path = os.path.join(tests_path, file_name)
        # Check if it's executable
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            executables.append(full_path)

    if len(executables) == 0:
        print(f"No executables found on test path {tests_path}. Skipping running tests...")
        return -1


    # # run tests
    for exec in executables:
        run_executable(exec, unknown_args)

possible_packages = ["dynosam", "dynosam_ros", "dynosam_cv"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Utility to run gtests for dynosam C++ packages.
        Additional args (anything besides --package, ie any gflags) are passed to gtest.\n
        Useful flags are --gtest_filter=POSTIVE_PATTERNS[-NEGATIVE_PATTERNS]"""
    )
    parser.add_argument(
        "--package",
        "-p",
        nargs="?",
        type=str,
        default="dynosam",
        help=f"Which package tests to run. Options are any of {possible_packages} or all",
    )

    args, unknown_args = parser.parse_known_args()

    if args.package == "all":
        pass
    else:
        # check valid package
        if args.package not in possible_packages:
            raise Exception(f"{args.package} is not a valud --package argument. Options are {possible_packages}")

        run_tests_for_package(args.package, unknown_args)
