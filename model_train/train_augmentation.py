from ultralytics import YOLO

from ultralytics import YOLO
from datetime import datetime
import os

augmentation = {
    "hsv_h": 0.015,  # 색조 변경 범위
    "hsv_s": 0.7,    # 채도 변경 범위
    "hsv_v": 0.4,    # 밝기 변경 범위
    "degrees": 10.0,  # 회전 각도 범위
    "translate": 0.1, # 평행 이동 범위
    "scale": 0.5,    # 크기 조정 범위
    "shear": 2.0,    # 왜곡 각도 범위
    "perspective": 0.001, # 원근 변환 범위
    "flipud": 0.5,   # 위아래 뒤집기 확률
    "fliplr": 0.5,   # 좌우 뒤집기 확률
    "mosaic": 1.0,   # 모자이크 증강 활성화
    "mixup": 0.5,    # MixUp 증강 활성화
    "copy_paste": 0.5 # Copy-Paste 증강 활성화
}

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
        augmentations=augmentation,
        name=f"{os.path.splitext(base_model)[0]}_{date_str}_e{epochs}"      # 저장 폴더 이름
    )
    return results

if __name__ == '__main__':
    train_model()