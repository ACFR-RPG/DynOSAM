from dynosam_nn_py._core import (
    get_nn_weights_path
)

if __name__ == "__main__":
    download_path = get_nn_weights_path()
    model_name: str = "yolo11s.pt"

    import os
    model_path = os.path.join(download_path, model_name)

    from ultralytics import YOLO
    # Load the YOLO model
    model = YOLO(model_path)
    #Export the model to ONNX format
    export_path = model.export(format="onnx")
    print(f"Exporting YOLOv11 to {export_path}")
