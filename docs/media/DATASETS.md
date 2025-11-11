## Datasets

We provide a number of data providers which process datasets into the input format specified by DynoSAM which includes input images for the pipeline and ground truth data for evaluation.

All datasets (including pre-processed images) can be found at the [ACFR-RPG Datasets page](https://data.acfr.usyd.edu.au/rpg/).
The provided dataset loaders are written to parse the datasets as provided.

### i. KITTI Tracking Dataset
We use a modified version of the KITTI tracking dataset which includes ground truth motion data, as well dense optical-flow, depth and segmentation masks.

The required dataset loader can be specified by setting `--data_provider_type=0`

### ii. Oxford Multimotion Dataset (OMD)
Raw data can be downloaded from the [project page](https://robotic-esp.com/datasets/omd/).
For our 2024 T-RO paper we used a modified version of the dataset which can be downloaded from the above link.

The required dataset loader can be specified by setting `--data_provider_type=3`



### iii. Cluster Dataset
The [raw dataset](https://huangjh-pub.github.io/page/clusterslam-dataset/) can be downloaded for the the CARLA-* sequences, although we recommend using our provided data.

The required dataset loader can be specified by setting `--data_provider_type=2`

### iv. Virtual KITTI 2
Access [raw dataset](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) and extract in a folder. No pre-processing is needed on this dataset and the raw data can be parsed by DynoSAM directly.

The required dataset loader can be specified by setting `--data_provider_type=1`

### v. TartanAir Shibuya
Details coming soon...


### vi. VIODE
Details coming soon...

### Online Dataprovider
An online data-provider can be specified using the ROS arg `online:=True`.
This node subscribes to five topics:
 - `dataprovider/image/camera_info` (sensor_msgs.msg.CameraInfo)
 - `dataprovider/image/rgb` (sensor_msgs.msg.Image)
 - `dataprovider/image/depth` (sensor_msgs.msg.Image)
 - `dataprovider/image/mask` (sensor_msgs.msg.Image)
 - `dataprovider/image/flow` (sensor_msgs.msg.Image)

The __rgb__ image is expected to be a valid 8bit image (1, 3 and 4 channel images are accepted).
The __depth__ must be a _CV_64F_ image where the value of each pixel represents the _metric depth_.
The __mask__ must be a CV_32SC1 where the static background is of value 0 and all other objects are lablled with a tracking label $j$.
The __flow__ must be a CV_32FC2 representing a standard optical-flow image representation.


We also provide a launch file specified for online usage:
```
ros2 launch dynosam_ros dyno_sam_online_launch.py
```
> NOTE: see the launch file for example topic remapping



## Dataset specific comments

### VIODE
- Needs `max_background_depth` to be large (500-ish)



## General running comments
- The param `min_features_per_frame` should always be the same as `max_features_per_frame` otherwise we get strange behaviour
