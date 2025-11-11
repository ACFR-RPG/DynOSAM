from dynosam_nn_py import YOLODetectionEngine
import numpy as np

if __name__ == "__main__":
    detection = YOLODetectionEngine()

    img = np.zeros((640, 480, 3))
    detection.process(img)
