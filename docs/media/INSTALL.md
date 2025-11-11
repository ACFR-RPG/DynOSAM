# DynoSAM Installation

Tested on Ubuntu 20.04

> NOTE: this instructions are taken essentially verbatum from the included [Dockerfile](./../../docker/Dockerfile)

## CUDA dependancies
As of September 2025 we are adding cuda dependancies for certain OpenCV operations (and evenautually for running inference on images).

> NOTE: I have a very old GPU (RTX 2080 which is only running CIDA 12.2 with Driver Version: 535.183.01), get CUDA support however you are able!

The docker file has been updated and based off the [docker-ros-ml-images](https://github.com/ika-rwth-aachen/docker-ros-ml-images?tab=readme-ov-file#rwthikaros2-torch-ros-2-nvidia-cuda-nvidia-tensorrt-pytorch) to include
- CUDA
- TensorRT
- Pytorch
built ontop of ROS2.

The CUDA dependancies are mostly for OpenCV modules but also eventually for adding DNN inference into DynoSAM itself so processing of images
can be handled internally (ie YOLO).

- **Important:** Check [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) to set `CUDA_ARCH_BIN` flag
- NVIDIA GeForce RTX 2080 is 7.5

## Dependancies

```bash
sudo apt-get install -y --no-install-recommends apt-utils

sudo apt-get install -y cmake

sudo apt install unzip libjpeg-dev libpng-dev libpng++-dev libtiff-dev libgtk2.0-dev libatlas-base-dev gfortran libgflags2.2 libgflags-dev libgoogle-glog0v5 libgoogle-glog-dev  python3-dev python3-setuptools clang-format python3-pip nlohmann-json3-dev libsuitesparse-dev
```

GTSAM's Optional dependencies (highly recommended for speed) also include
```bash
sudo apt install libboost-all-dev libtbb-dev
```

For evaluation we also need various formatting tools
```bash
sudo apt-get install texlive-pictures texlive-science texlive-latex-extra latexmk
```

Python (pip) dependancies:
```bash
python3 -m pip install pylatex evo setuptools pre-commit scipy matplotlib argcomplete black pre-commit
```

NOTE: There seems to be a conflict between the version of matplotlib and numpy. In this case following worked for me:
```
sudo apt remove python3-matplotlib
python3 -m pip install numpy==1.26.3
```

## Install OpenCV

Note that you can use `apt-get install libopencv-dev libopencv-contrib-dev` on 20.04 instead of building from source.

> Also note, I have not actually tried this!

To build from source

```bash
git git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib && \
    git checkout tags/4.8.0

cd ../ &&
git clone git clone https://github.com/opencv/opencv.git

cd opencv && \
git checkout tags/4.8.0 && mkdir build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_opencv_python=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules .. # Use -DWITH_TBB=On if you have TBB

sudo make -j$(nproc) install
```

## Install GTSAM

Tested with release 4.2

```bash

git clone https://github.com/borglab/gtsam.git
cd gtsam && \
    git fetch && \
    git checkout tags/4.2.0 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DCMAKE_BUILD_TYPE=Release -DGTSAM_BUILD_UNSTABLE=ON -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON -DGTSAM_TANGENT_PREINTEGRATION=OFF .. && \

sudo make -j$(nproc) install
```

## Install OpenGV

```bash
git clone https://github.com/MIT-SPARK/opengv
cd opengv && mkdir build && cd build &&
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local ..

sudo make -j$(nproc) install

```
Note: This links to a forked version of OpenGV which contains an updated fix the CMakeLists.txt due to issues with -march=native being set. This links to original version of [OpenGV](https://github.com/laurentkneip/opengv), which contains the problematic flag.

## Install Config Utilities

```bash
git clone git@github.com:MIT-SPARK/config_utilities.git

cd config_utilities/config_utilities
mkdir build
cd build
cmake ..
make -j

sudo make install
```

## Additional Installation Notes


Due to DynoSAM being build within ROS:
- Need to build GTSAM with `-DGTSAM_USE_SYSTEM_EIGEN=ON` to avoid issues with ROS and OpenCV-Eigen compatability. Confirmed from https://discourse.ros.org/t/announcing-gtsam-as-a-ros-1-2-package/32739 which explains that the GTSAM ROS package is build with this flag set to ON (and describes it as "problematic"). We still want to build GTSAM from source so we can control the version and other compiler flags.
Kimera-VIO's install instructions indicate that OpenGV must use the same version of Eigen as GTSAM, which can be set using compiler flags. Since we are using the ROS install Eigen, I have removed these flags and hope that the package manager with CMake can find the right (and only) version. This has not proved problematic... yet...
