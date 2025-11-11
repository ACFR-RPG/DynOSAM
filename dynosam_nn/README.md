# Dynosam NN module

Implements TensorRT accelerated instance detection and tracking using YOLOv8.

`YoloV8ObjectDetector` is the main class but there are some others implemented, mostly in python.

Runs inference on an input image and returns a `ObjectDetectionResult` which contains _tracked_ instances of each object and the labelled object mask.
All this information is directly comptabale with the frontend.

To use instead of the pre-computed object masks, set the dynosam params set `prefer_provided_object_detection=false`.

> NOTE: If (for some reason) using with the offline dataset loaders, not that tracking happens internally and therefore the tracked object id may not align with the ground truth. Always use `prefer_provided_object_detection=false` or ensure that ground truth logging is off!


For object-level tracking we use a modified C++ implementation of ByteTracker

# Model file
- YoloV8ObjectDetector requires an exported `.onnx` file which will be converted to a `.engine` file when first loaded. See [export_yolo_tensorrt.py](./export/export_yolo_tensorrt.py) for how to export this file. It should (and by default) be put in the _installed_ share directory of `dynsam_nn` under `weights`. By default the model config will look in this folder for all model weights.

# Install
- python3 -m pip install "ultralytics==8.3.0" "numpy<2.0" "opencv-python<5.0"
- sudo apt install python3-pybind11


## Details
The OpenCV/Numpy converion code was taken verbatum from https://gitverse.ru/githubcopy/cvnp/content/master
Much of the TensorRT code and structure of the Model class was taken from https://github.com/MIT-SPARK/semantic_inference
