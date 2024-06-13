from ultralytics import YOLOv10

# model = YOLOv10('yolov10n.pt')
# model = YOLOv10.from_pretrained('jameslahm/yolov10n')
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

# model.train(data='coco.yaml', epochs=2, batch=16, imgsz=640)

# https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt


from datetime import datetime
import os

def train_model():
    now = datetime.now()
    date_str = now.strftime("%m%d")

    # base_model = "yolov8s.pt"
    base_model = 'yolov10n.pt'
    model = YOLOv10(base_model)

    epochs = 100
    batch=64

    # 모델 훈련
    results = model.train(
        data="./datasets/recyclables_2/recyclables_2.yaml",
        epochs=epochs,
        imgsz=640,
        save_period=5,
        device=1,
        batch=batch,
        # project="./path/to/save",
        name=f"{os.path.splitext(base_model)[0]}_{date_str}_e{epochs}_b{batch}"      # 저장 폴더 이름
    )
    return results

if __name__ == '__main__':
    train_model()
