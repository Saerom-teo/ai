import os
import shutil
import random

def split_dataset(images_folder, labels_folder, output_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # 이미지 파일 목록 가져오기
    images = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    
    # 이미지 파일 목록 섞기
    random.shuffle(images)
    
    # 분할 기준 설정
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count  # 나머지는 test에 할당

    # 파일을 분할할 폴더 경로 설정
    train_image_folder = os.path.join(output_folder, 'images/train')
    val_image_folder = os.path.join(output_folder, 'images/val')
    test_image_folder = os.path.join(output_folder, 'images/test')

    train_label_folder = os.path.join(output_folder, 'labels/train')
    val_label_folder = os.path.join(output_folder, 'labels/val')
    test_label_folder = os.path.join(output_folder, 'labels/test')

    # 폴더 생성
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(val_image_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)

    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(val_label_folder, exist_ok=True)
    os.makedirs(test_label_folder, exist_ok=True)

    # 이미지 파일을 각 폴더로 복사
    for i, image in enumerate(images):
        src_image_path = os.path.join(images_folder, image)
        src_label_path = os.path.join(labels_folder, os.path.splitext(image)[0] + '.txt')  # 라벨 파일 경로
        
        if i < train_count:
            dst_image_path = os.path.join(train_image_folder, image)
            dst_label_path = os.path.join(train_label_folder, os.path.splitext(image)[0] + '.txt')
        elif i < train_count + val_count:
            dst_image_path = os.path.join(val_image_folder, image)
            dst_label_path = os.path.join(val_label_folder, os.path.splitext(image)[0] + '.txt')
        else:
            dst_image_path = os.path.join(test_image_folder, image)
            dst_label_path = os.path.join(test_label_folder, os.path.splitext(image)[0] + '.txt')
        
        # 이미지 파일 복사
        shutil.copy(src_image_path, dst_image_path)
        
        # 라벨 파일 복사
        shutil.copy(src_label_path, dst_label_path)
    
    print(f"========= Result =========")
    print(f"Total images: {total_images}")
    print(f"Train images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Test images: {test_count}")


images_folder = 'test/images'
labels_folder = 'test/labels'
output_folder = 'datasets/recyclables'

split_dataset(images_folder, labels_folder, output_folder)
