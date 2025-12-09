# DynoSAM Installation

## Prerequisites
- [ROS2](https://docs.ros.org/en/kilted/Installation.html)
    > NOTE: this links to the kilted distribution which used during development. We have also tested on Jazzy for the NVIDIA ORIN
- [GTSAM](https://github.com/borglab/gtsam) >= 4.1
- [OpenCV](https://github.com/opencv/opencv) >= 3.4
- [OpenGV](https://github.com/MIT-SPARK/opengv)
- [Glog](http://rpg.ifi.uzh.ch/docs/glog.html), [Gflags](https://gflags.github.io/gflags/)
- [Gtest](https://github.com/google/googletest/blob/master/googletest/docs/primer.md) (installed automagically)
- [config_utilities](https://github.com/MIT-SPARK/config_utilities)
- [dynamic_slam_interfaces](https://github.com/ACFR-RPG/dynamic_slam_interfaces) (Required by default. Can be optionally made not a requirement. See Insallation instructions below)

External dependancies (for visualization) not required for compilation.
- [rviz_dynamic_slam_plugins](https://github.com/ACFR-RPG/rviz_dynamic_slam_plugins) (Plugin to display custom `dynamic_slam_interfaces` messages which are advertised by default.)


GPU acceleration:
- [TensorRT](https://github.com/NVIDIA/TensorRT)
- [CUDA](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

are now supported with the new Docker image.
This provides support for TensorRT accellerated object-instance segmentation (which is now part of the DynoSAM pipeline) and CUDA acceleration in the front-end.
Backwards compatability (i.e no CUDA support) is not currently a priority.

We provide detailed install instructions when using the Docker.

To install natively, install the dependancies as required by docker and build as a ROS2 package.

## Docker Install instructions

DynoSAM has been tested on x86_64 and aarm64 (with a NVIDIA ORIN) devices using the [two docker files](../../docker/) provided. See the [REAMDE.md](../../docker/README.md) for more detail on hardware used etc.

We provide scripts to build and create docker containers to run and develop DynoSAM which is intendend to be mounted within the created container.

### Folder Structure
To DynoSAM code to be changed within the docker environment, we mount the local version of the code within the container. To ensure this happens smoothly please download the DynoSAM code in the following structure

```
dynosam_pkg/
    DynoSAM/
    extras/
        rviz_dynamic_slam_plugins
        dynamic_slam_interfaces
    results/
```
where DynoSAM is this repository and anything in `extras` are other ROS packages that wish to be built alongside DynoSAM.

> NOTE: `dynosam_pkg' may be any folder, its just a place to put all the DynoSAM related code.

> NOTE: the reason for this structure is historical and due to the way the _create_container_ scripts are written; you don't need to do this but I provide complete instructures for simplicity.


### Docker Build
Build the relevant docker file:
```
cd dynosam_pkg/DynoSAM/docker &&
./build_docker_amd64.sh //or build_docker_l4t.sh
```
### Create Container
Once built, you should have a docker image called `acfr_rpg/dyno_sam_cuda` or `acfr_rpg/dynosam_cuda_l4t`

In the associated _create_container_ scripts modify the local variables to match the folder paths on the local machine. These folders will be mounted as a volume within the container
```
LOCAL_DATA_FOLDER=/path/to/some/datasets/
LOCAL_RESULTS_FOLDER=/path/to/dynosam_pkg/results/
LOCAL_DYNO_SAM_FOLDER=/path/to/dynosam_pkg/DynoSAM/
LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER=/path/to/dynosam_pkg/extras
```
Run the creation script. The created container will be called `dyno_sam`
```
./create_container_amd64.sh //or ./create_container_l4t_jetpack6.sh
```

Finally enter into the container and build DynoSAM
```
cd /home/user/dev_w &&
export MAKEFLAGS="-j10" && clear && colcon build
```

## CUDA dependancies
As of September 2025 we are adding cuda dependancies for certain OpenCV operations (and evenautually for running inference on images).

> NOTE: I have a very old GPU (RTX 2080 which is only running CUDA 12.2 with Driver Version: 535.183.01), get CUDA support however you are able!

The docker file has been updated and based off the [docker-ros-ml-images](https://github.com/ika-rwth-aachen/docker-ros-ml-images?tab=readme-ov-file#rwthikaros2-torch-ros-2-nvidia-cuda-nvidia-tensorrt-pytorch) to include
- CUDA
- TensorRT
- Pytorch
built ontop of ROS2.

The CUDA dependancies are mostly for OpenCV modules but also eventually for adding DNN inference into DynoSAM itself so processing of images
can be handled internally (ie YOLO).

- **Important:** Check [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) to set `CUDA_ARCH_BIN` flag
- NVIDIA GeForce RTX 2080 is 7.5

## Additional Installation Notes

`dynamic_slam_interfaces` is a require dependacy by default. This package is used to include custom messages that represet the state of each dynamic object per frame and is used by the ROS publishers.

To disable this dependancy compile the code as
```
colcon build --cmake-args -DENABLE_DYNAMIC_SLAM_INTERFACES=OFF
```
By default `ENABLE_DYNAMIC_SLAM_INTERFACES=ON` in the [CMakeLists.txt](./dynosam_ros/CMakeLists.txt). This CMake option will additionally change the _visualisation_ (and the output topics) used by DynoSAM. See the [ROS Visualisation](#ros-visualisation) section below.


Due to DynoSAM being build within ROS:
- Need to build GTSAM with `-DGTSAM_USE_SYSTEM_EIGEN=ON` to avoid issues with ROS and OpenCV-Eigen compatability. Confirmed from https://discourse.ros.org/t/announcing-gtsam-as-a-ros-1-2-package/32739 which explains that the GTSAM ROS package is build with this flag set to ON (and describes it as "problematic"). We still want to build GTSAM from source so we can control the version and other compiler flags.
Kimera-VIO's install instructions indicate that OpenGV must use the same version of Eigen as GTSAM, which can be set using compiler flags. Since we are using the ROS install Eigen, I have removed these flags and hope that the package manager with CMake can find the right (and only) version. This has not proved problematic... yet...

## Possible Compilation Issues
### Missing MPI Header Error
When first compiling DynoSAM, this error may appear as MPI relies on a non-existent directory. 
This issue _should_ be fixed a patch in the `dynosam_common` CMakeLists.txt which directly updates the `MPI_INCLUDE_PATH`.

However, if the issue persists, the following is a known fix.
1. Unset HPC-X variables
   ```bash
   unset OPENMPI_VERSION
   unset OMPI_MCA_coll_hcoll_enable 
   unset OPAL_PREFIX 
   export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/opt/hpcx' | grep -v '/usr/local/mpi/bin' | paste -sd:)
    ```
   Now verify `env | grep -i mpi echo $PATH` that the `OPAL_PREFIX` and HPC-X paths no longer appear
2. Install OpenMPI
   ```bash
   sudo apt update
   sudo apt install libopenmpi-dev openmpi-bin
   ```
   and verify that the following points to `/usr/bin/mpicc`
   ```bash
   which mpicc
   mpicc --version   
   ```
3. Clear the ROS2 workspace
   ```bash
   rm -rf build/ install/ log/
   ```
4. Build with the system MPI
   ```bash
   export MPI_C_COMPILER=/usr/bin/mpicc 
   export MPI_CXX_COMPILER=/usr/bin/mpicxx 
   colcon build
   ```
### Missing Dependencies 
- `nlohmannjson` may not be installed, so install with `sudo apt install nlohmann-json3-dev`
- GTSAM may not be installed, so install with `pip install gtsam`
   
