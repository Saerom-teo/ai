import os
import shutil
import random

def split_dataset(data_folder, output_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    images_folder = os.path.join(data_folder, 'images')
    labels_folder = os.path.join(data_folder, 'labels')

    images = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    
    random.shuffle(images)
    
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count

    train_image_folder = os.path.join(output_folder, 'images/train')
    val_image_folder = os.path.join(output_folder, 'images/val')
    test_image_folder = os.path.join(output_folder, 'images/test')

    train_label_folder = os.path.join(output_folder, 'labels/train')
    val_label_folder = os.path.join(output_folder, 'labels/val')
    test_label_folder = os.path.join(output_folder, 'labels/test')

    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(val_image_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)

    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(val_label_folder, exist_ok=True)
    os.makedirs(test_label_folder, exist_ok=True)

    for i, image in enumerate(images):
        if i%500 == 0:
            print(f"file_num: {i}")
        src_image_path = os.path.join(images_folder, image)
        src_label_path = os.path.join(labels_folder, os.path.splitext(image)[0] + '.txt')
        
        if i < train_count:
            dst_image_path = os.path.join(train_image_folder, image)
            dst_label_path = os.path.join(train_label_folder, os.path.splitext(image)[0] + '.txt')
        elif i < train_count + val_count:
            dst_image_path = os.path.join(val_image_folder, image)
            dst_label_path = os.path.join(val_label_folder, os.path.splitext(image)[0] + '.txt')
        else:
            dst_image_path = os.path.join(test_image_folder, image)
            dst_label_path = os.path.join(test_label_folder, os.path.splitext(image)[0] + '.txt')
        
        shutil.copy(src_image_path, dst_image_path)
        
        shutil.copy(src_label_path, dst_label_path)
    
    print(f"========= Result =========")
    print(f"Total images: {total_images}")
    print(f"Train images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Test images: {test_count}")


if __name__ == "__main__":
    working_directory = os.getcwd()

    data_folder = os.path.join(working_directory, "datas", "regulation_datas")
    output_folder = os.path.join(working_directory, "datasets", "recyclables")

    print(data_folder)
    print(output_folder)

    split_dataset(data_folder, output_folder)