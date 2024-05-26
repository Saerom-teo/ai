from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(data="recyclables.yaml", epochs=1, imgsz=640)