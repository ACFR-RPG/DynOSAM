# syntax=docker/dockerfile:latest
# FROM osrf/ros:humble-desktop-full
# torch contains ROS 2, NVIDIA CUDA, NVIDIA TensorRT, PyTorch
FROM rwthika/ros2-torch:kilted-desktop-full-torch2.5.0


MAINTAINER Jesse Morris "jesse.morris@sydney.edu.au"


# To avoid tzdata asking for geographic location...
ENV DEBIAN_FRONTEND=noninteractive

#Install build dependencies
RUN apt-get update && apt-get upgrade -y --no-install-recommends apt-utils
RUN apt-get update && apt-get install -y git cmake build-essential pkg-config
# Install xvfb to provide a display to container for GUI realted testing.
RUN apt-get update && apt-get install -y xvfb



RUN apt-get update \
    && apt-get install -y \
    python3-pip \
    openssh-client \
    software-properties-common \
    nano \
    vim \
    clang-format \
    && pip3 install black pre-commit \
    && rm -rf /var/lib/apt/lists/*

# ROS installs
RUN apt-get install -y \
      ros-kilted-ros2cli-common-extensions \
      ros-kilted-vision-opencv

# other deps
RUN apt-get install libpng++-dev

RUN python3 -m pip install pylatex evo setuptools pre-commit scipy argcomplete black pre-commit

# Install glog, gflags
RUN apt-get update && apt-get install -y libgflags2.2 libgflags-dev libgoogle-glog-dev

# Install xvfb to provide a display to container for GUI realted testing.
RUN apt-get update && apt-get install -y xvfb python3-dev python3-setuptools

RUN pip3 install setuptools pre-commit scipy matplotlib argcomplete

# install CSparse
RUN DEBIAN_FRONTEND=noninteractive apt install -y libsuitesparse-dev

# Parallelism C++ for CPU
RUN apt-get update && apt-get install -y libboost-all-dev libtbb-dev

WORKDIR /root/



# Install OpenCV for Ubuntu
RUN apt-get update && apt-get install -y \
      unzip \
      libjpeg-dev libpng-dev libpng++-dev libtiff-dev libgtk2.0-dev \
      libatlas-base-dev gfortran

RUN git clone https://github.com/opencv/opencv.git
RUN cd opencv && \
      git checkout tags/4.10.0 && \
      mkdir build

RUN git clone https://github.com/opencv/opencv_contrib.git
RUN cd opencv_contrib && \
      git checkout tags/4.10.0

RUN cd opencv/build && \
      cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_opencv_python=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D INSTALL_CUDA_LIBS=OFF \
      -D CMAKE_CUDA_ARCHITECTURES="75" \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUFFT=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_TBB=ON \
      -DOPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib/modules .. && \
      make -j$(nproc) install


RUN git clone https://github.com/MIT-SPARK/config_utilities.git
RUN cd config_utilities/config_utilities && mkdir build && \
      cd build && \
      cmake .. && make -j$(nproc) install



# Install GTSAM
RUN git clone https://github.com/borglab/gtsam.git
RUN cd gtsam && \
    git fetch && \
    git checkout tags/4.2.0 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DCMAKE_BUILD_TYPE=Release -DGTSAM_BUILD_UNSTABLE=ON -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON -DGTSAM_TANGENT_PREINTEGRATION=OFF .. && \
    make -j$(nproc) install


# Install Open_GV
RUN git clone https://github.com/MIT-SPARK/opengv
RUN cd opengv && \
      mkdir build
RUN cd opengv/build && \
      cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      .. && make -j$(nproc) install



RUN mkdir -p /home/user/dev_ws/src/core
RUN mkdir -p /home/user/dev_ws/src/third_parties
RUN mkdir -p /home/user/upstream_ws/src

SHELL ["/bin/bash", "-c"]

RUN source /opt/ros/kilted/setup.bash

WORKDIR /home/user/dev_ws
