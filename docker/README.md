# Docker Files for DynoSAM

Base images are pulled from [docker-ros-ml-images](https://github.com/ika-rwth-aachen/docker-ros-ml-images)

- Dockerfile.amd64 is a linux/amd64 image tested on x86_64 desktop
- Dockerfile.l4t_jetpack6 is build from linux/arm64 tested on an NVIDIA ORIN NX with Jetpack 6

## Jetson Settings
Architecture | aarch64
Ubuntu | 22.04.5 LTS (Jammy Jellyfish)
Jetson Linux | 36.4.7
Python | 3.10.12
ROS | jazzy
CMake | 3.22.1
CUDA | 12.6.77-1
cuDNN | 9.3.0.75-1
TensorRT | 10.7.0.23-1+cuda12.6
PyTorch | 2.8.0
GPUs | (Orin (nvgpu))
OpenCV | 4.10.0

> NOTE: The CUDA/Pytorch/TensorRT versions settings come with the base dockerfile but in practice we have been using CUDA 12.9. 

## Other versioning
matplotlib=3.6.3
numpy=1.26.4