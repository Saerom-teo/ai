from ultralytics import YOLO

def train_model():

    # 모델 불러오기
    model = YOLO("yolov8n.pt")

    # 모델 훈련
    results = model.train(data="./recyclables.yaml", epochs=30, imgsz=640)
    return results

if __name__ == '__main__':
    train_model()
