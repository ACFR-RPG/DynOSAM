from dynosam_nn_py import YOLODetectionEngine


if __name__ == "__main__":
    detection = YOLODetectionEngine()
    model = detection._model
    model.info()
    exported_path = model.export(format="onnx")
    print(f"Exported model to {exported_path}")
