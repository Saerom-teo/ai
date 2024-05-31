from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8n model
model = YOLO('./runs/detect/train2/weights/best.pt')

# metrics = model.val()
# print(metrics.box.map)  # map50-95
# print(metrics.box.map50)  # map50
# print(metrics.box.map75)  # map75
# print(metrics.box.maps)  # a list contains map50-95 of each category

# Define path to the image file
# source = "datasets/recyclables/images/test/000014.jpg"
# source = "datas/test/0036773.jpg"
source = "KakaoTalk_20240531_142144938_01.jpg"
# source = "datas/KakaoTalk_20240531_141407341.jpg"

# Run inference on the source
results = model.predict(source, conf=0.25)  # list of Results objects

for result in results:
    boxes = result.boxes
    print("====== boxes ======")
    print(boxes)
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
