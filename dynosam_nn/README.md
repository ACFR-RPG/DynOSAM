# Dynosam NN module

Implements TensorRT accelerated instance detection and tracking using YOLOv8.

`YoloV8ObjectDetector` is the main class but there are some others implemented, mostly in python.

Runs inference on an input image and returns a `ObjectDetectionResult` which contains _tracked_ instances of each object and the labelled object mask.
All this information is directly comptabale with the frontend.

To use instead of the pre-computed object masks, set the dynosam params set `prefer_provided_object_detection=false`.

> NOTE: If (for some reason) using with the offline dataset loaders, not that tracking happens internally and therefore the tracked object id may not align with the ground truth. Always use `prefer_provided_object_detection=false` or ensure that ground truth logging is off!


For object-level tracking we use a modified C++ implementation of ByteTracker

## Exporting Model and Weights
- DyoSAM looks for model weights in the `ros_ws/install.../dynosam_nn/weights` directory.
> NOTE: this is the _install_ directory (ie. in the docker container it will be `/home/user/dev_ws/install/dynosam_nn/share/dynosam_nn/weights/`)
- To export the model navigate to `dynosam_nn/export` and run
```
run python3 export_yolo_tensorrt.py
```
which should export a `.onnx` model to the weights directory.
- YoloV8ObjectDetector requires an exported `.onnx` file which will be converted to a `.engine` file when first loaded.

## Install
- python3 -m pip install "ultralytics==8.3.0" "numpy<2.0" "opencv-python<5.0"
- sudo apt install python3-pybind11

> NOTE: these dependancies should already be installed when using the Dockerfile


## Details
The OpenCV/Numpy converion code was taken verbatum from https://gitverse.ru/githubcopy/cvnp/content/master
Much of the TensorRT code and structure of the Model class was taken from https://github.com/MIT-SPARK/semantic_inference

## Embedded System Performance
We have tested our system on an NVIDIA ORIN NX.

As this platform supports multiple performance modes, it is important to select the highest mode to ensure best performance.

Change modes
```
sudo nvpmodel -m <POWER_MODEL>
```
The different power modes can be checked by vieweing
```
cat /etc/nvpmodel.conf
```
while the current settings can be viewed
```
sudo jetson_clocks --show
```
> NOTE: setting `POWER_MODEL=1` was the most performant for our hardware.

keep in mind, that the mode switching only changes the maximum ATTAINABLE clocks, the clocks are dynamically changed based on the load.
In order to set the clocks statically to their highest setting WITHIN the mode, execute:
```
sudo jetson_clocks
```
