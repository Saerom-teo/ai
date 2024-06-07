from ultralytics import YOLO
from datetime import datetime
import os

def train_model():
    now = datetime.now()
    date_str = now.strftime("%m%d")

    base_model = "yolov8n.pt"
    model = YOLO(base_model)

    epochs = 100

    # 모델 훈련
    results = model.train(
        data="./datasets/recyclables_2/recyclables_2.yaml",
        epochs=epochs,
        imgsz=640,
        save_period=5,
        device=1,
        # project="./path/to/save",
        name=f"{os.path.splitext(base_model)[0]}_{date_str}_e{epochs}"      # 저장 폴더 이름
    )
    return results

if __name__ == '__main__':
    train_model()
