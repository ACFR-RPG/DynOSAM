import numpy as np
from dynosam_nn_py._core import (
    ObjectDetectionEngine,
    ObjectDetectionResult,
    SingleDetectionResult,
    get_nn_weights_path,
    mask_to_rgb
)
import cv2
import torch

from ultralytics.engine.model import Model as UltralyticsModel


class UltralyticsDetectionEngine(ObjectDetectionEngine):

    out_mask_dype = np.int32

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._result: ObjectDetectionResult = ObjectDetectionResult()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        self._model: UltralyticsModel = self.load_model(**kwargs)
        if self._model is None:
            raise Exception("Ultraltics model is none! Failed to load object detection model")
        self._model.to(self.device)

         # which class names (e.g. person, bicycle...) to include
        # all other clases will not be included in the mask output
        self._included_classes = kwargs.get("include_classes", [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "train",
            "truck"
        ])
        if "include_classes" in kwargs:
            del kwargs["include_classes"]
        # we will specify this argument to the model
        # the user should provide include_classes which is the classes by string
        # while YOLO takes their own class ids which we update once we know the mapping
        # of class names to YOLO id's
        if "classes" in kwargs:
            print(f"[WARNING] User defined argument 'classes' found in kwargs."
                    "Please specify 'include_classes' instead as we remap these to the models internal class labels!")
            del kwargs["classes"]

        # list of included classes via their ids not a string to identify them in YOLO
        # set on the first run
        self._included_classes_ids = []

        self._used_object_ids = {} # mapping of YOLO tracks to our global track
        self._global_object_id = 1
        self._kwargs = kwargs

    def load_model(self, **kwargs) -> UltralyticsModel:
        raise NotImplementedError

    def process(self, image: np.ndarray) -> ObjectDetectionResult:
        # do we need to put everything back on the CPU?
        with torch.no_grad():
            model_result =  self._model.track(
                image,
                persist=True,
                classes=self._included_classes_ids,
                retina_masks=True,
                imgsz=640,
                **self._kwargs)[0].cpu()

            names = model_result.names # dicionary of classifciation id to label (e.g 0: person). This is the entire dictionary of possible classes
            self._set_included_classes_ids(names)
            # reset result
            # reset info before checks so we can return safetly from this function
            self._result.labelled_mask = np.zeros(image.shape[:2], dtype=UltralyticsDetectionEngine.out_mask_dype)
            self._result.input_image = image
            self._result.detections.clear()

            if model_result.masks is None:
                return self._result


            # #assume image is in W, H, C
            class_ids = model_result.boxes.cls.cpu().numpy()    # cls, (N, 1), classifciation id
            probs = model_result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
            boxes = model_result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)


            # ignore if track was false?
            if not model_result.boxes.is_track:
                return self._result

            track_ids = model_result.boxes.id.int().cpu().tolist()
            detection_masks = model_result.masks.data.cpu().numpy()     # masks, (N, H, W)

            # masks should be resized into (N, H, W),
            # where N = number of masks
            # H,W are image height of the original image
            def process_masks(masks, image):
                from ultralytics.utils.ops import scale_image
                 # masks in order (N, H, W)
                masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
                masks = scale_image(masks, image.shape)
                masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)
                return masks

            detection_masks = process_masks(detection_masks, image)

            # converts box in x_min, y_min, x_max, y_max to x,y, width, height
            def bb_converter(box):
                x_min, y_min, x_max, y_max = box
                x = x_min
                y = y_min
                width  = x_max - x_min
                height = y_max - y_min
                return x,y,width, height

            for class_id, prob, box, track_id, detection_mask, in zip(class_ids, probs, boxes, track_ids, detection_masks):
                class_name = names[class_id]
                if class_name not in self._included_classes:
                    continue

                # track id's in YOLO start at 0. we want to index from 1
                # track_id += 1
                # assert track_id > 0
                object_id = self._asign_obj_label(track_id)

                detection_mask_img = np.where(detection_mask != 0, object_id, 0).astype(UltralyticsDetectionEngine.out_mask_dype)
                self._result.labelled_mask += detection_mask_img


                single_result = SingleDetectionResult()
                single_result.object_id = object_id
                single_result.bounding_box = bb_converter(box)
                single_result.confidence = prob
                single_result.class_name = class_name

                self._result.detections.append(single_result)

        return self._result

    def result(self) -> ObjectDetectionResult:
        return self._result


    def on_destruction(self):
        print("Detructing YOLO")

    def _asign_obj_label(self, track_id):
        if track_id in self._used_object_ids:
            return self._used_object_ids[track_id]

        object_id = self._global_object_id
        self._global_object_id += 1

        self._used_object_ids[track_id] = object_id
        print(f"New object track {object_id} (remapped from yolo track {track_id})")
        return object_id

    def _set_included_classes_ids(self, names):
        # ensure we only set it once!
        if len(self._included_classes_ids) == 0 and len(self._included_classes) > 0:
            for class_id, class_name in names.items():
                if class_name in self._included_classes:
                    self._included_classes_ids.append(class_id)
            print(f"Setting included classes ids {self._included_classes_ids}")


class YOLODetectionEngine(UltralyticsDetectionEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, **kwargs) -> UltralyticsModel:
        download_path = get_nn_weights_path()
        model_name: str = "yolov8n-seg.pt"

        import os
        model_path = os.path.join(download_path, model_name)

        from ultralytics import YOLO
        return YOLO(model_path)


class RTDETRDetectionEngine(UltralyticsDetectionEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self, **kwargs) -> UltralyticsModel:
        # download_path = get_nn_weights_path()
        # model_name: str = "rtdetr-l.pt"

        # import os
        # model_path = os.path.join(download_path, model_name)

        # from ultralytics import RTDETR
        # return RTDETR(model_path)
        raise Exception("RTDETRDetectionEngine not implemented!")
