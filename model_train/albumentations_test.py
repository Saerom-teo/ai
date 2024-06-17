import cv2
import numpy as np
import albumentations as A
import os

# 증강 설정
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0, shift_limit=0, p=1),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo', min_area=0.0, min_visibility=0.0))

def read_image(image_path):
    return cv2.imread(image_path)

def read_bboxes(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
        bboxes = []
        for line in lines:
            parts = line.strip().split()
            label = int(parts[0])
            bbox = list(map(float, parts[1:]))
            bboxes.append([label] + bbox)
    return bboxes

def save_image(image, save_path):
    cv2.imwrite(save_path, image)

def save_bboxes(bboxes, save_path):
    with open(save_path, 'w') as file:
        for bbox in bboxes:
            line = ' '.join(map(str, bbox))
            file.write(f"{line}\n")

def augment_data(image_path, label_path, save_image_path, save_label_path):
    image = read_image(image_path)
    bboxes = read_bboxes(label_path)

    # YOLO 형식으로 변환
    bboxes = [[bbox[1], bbox[2], bbox[3], bbox[4], bbox[0]] for bbox in bboxes]

    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    print(transformed_bboxes)
    # 다시 YOLO 형식으로 변환
    transformed_bboxes = [[bbox[0]] + bbox[1:] for bbox in transformed_bboxes]

    save_image(transformed_image, save_image_path)
    save_bboxes(transformed_bboxes, save_label_path)

# 예시 파일 경로
image_path = '0000003.jpg'
label_path = '0000003.txt'
save_image_path = 'augmented_example.jpg'
save_label_path = 'augmented_example.txt'

augment_data(image_path, label_path, save_image_path, save_label_path)
