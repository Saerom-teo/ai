from ultralytics import YOLO
import os


def export_onnx(base_path):
    model_list = [model for model in os.listdir(base_path) if model.endswith('.pt')]

    for model in model_list:
        model_path = os.path.join(base_path, model)
        print(model_path)

        # Load the YOLOv8 model
        model = YOLO(model_path)

        # Export the model to ONNX format
        model.export(format="onnx")


if __name__ == "__main__":
    base_path = "resources/models"
    export_onnx(base_path)

# python -m docs.export_onnx