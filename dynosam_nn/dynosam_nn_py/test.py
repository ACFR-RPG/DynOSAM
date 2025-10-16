from dynosam_nn_py import YOLODetectionEngine

if __name__ == "__main__":
    detection = YOLODetectionEngine()
    detection.process(detection.mask())
