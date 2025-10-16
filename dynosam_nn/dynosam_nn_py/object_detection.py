import numpy as np
from dynosam_nn_py._core import ObjectDetectionEngine, ObjectDetectionResult


class YOLODetectionEngine(ObjectDetectionEngine):

    def __init__(self):
        super().__init__()
        self._result: ObjectDetectionResult = ObjectDetectionResult()
        # from ultralytics import YOLO
        # import os
        # download_path = kwargs.get("download_dir", get_weights_folder())
        # model_path = os.path.join(download_path, model_name)
        # self._model = YOLO(model_path)

    def process(self, image: np.ndarray) -> ObjectDetectionResult:
        print(f"Processing image of shape {image.shape}")
        self._result = ObjectDetectionResult()
        self._result.success = False
        return self._result

    def result(self) -> ObjectDetectionResult:
        return self._result

    def load_model(self):
        print("Loading mode...")

    def on_destruction(self, ):
        print("Detructing YOLO")
