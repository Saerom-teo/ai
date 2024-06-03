import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import json


def create_dataset_structure(data_dir, dataset_dir):
    # Create dataset structure
    os.makedirs(os.path.join(dataset_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'test'), exist_ok=True)

def split_data_and_copy(src_dir, dst_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of the ratios must be 1"

    for label in os.listdir(src_dir):
        label_dir = os.path.join(src_dir, label)
        if not os.path.isdir(label_dir):
            continue
        
        # List all files for this label
        files = os.listdir(label_dir)
        
        # Split the files
        train_files, temp_files = train_test_split(files, test_size=1-train_ratio, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)

        # Copy the files to the corresponding directories with progress bar
        print(f'Copying files for label: {label}')
        for file in tqdm(train_files, desc="Training files"):
            new_filename = f"{label}_{file}"
            shutil.copy(os.path.join(label_dir, file), os.path.join(dst_dir, 'train', new_filename))
        
        for file in tqdm(val_files, desc="Validation files"):
            new_filename = f"{label}_{file}"
            shutil.copy(os.path.join(label_dir, file), os.path.join(dst_dir, 'val', new_filename))
        
        for file in tqdm(test_files, desc="Test files"):
            new_filename = f"{label}_{file}"
            shutil.copy(os.path.join(label_dir, file), os.path.join(dst_dir, 'test', new_filename))

def organize_dataset(data_dir, dataset_dir):
    create_dataset_structure(data_dir, dataset_dir)
    
    # Process images
    image_src_dir = os.path.join(data_dir, 'images')
    image_dst_dir = os.path.join(dataset_dir, 'images')
    split_data_and_copy(image_src_dir, image_dst_dir)
    
    # Process labels
    label_src_dir = os.path.join(data_dir, 'labels')
    label_dst_dir = os.path.join(dataset_dir, 'labels')
    split_data_and_copy(label_src_dir, label_dst_dir)

def count_labels(label_dir):
    label_counts = {'train': {}, 'val': {}, 'test': {}}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(label_dir, split)
        for label_file in os.listdir(split_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(split_dir, label_file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        label = line.split()[0]  # 라벨은 첫 번째 값
                        if label not in label_counts[split]:
                            label_counts[split][label] = 0
                        label_counts[split][label] += 1
                        
    # 디버깅 출력을 추가합니다.
    for split, counts in label_counts.items():
        print(f"Label counts for {split}: {counts}")
        
    return label_counts

def plot_label_distribution(label_counts, label_map, output_file):
    # 라벨 번호를 한국어 클래스 이름으로 변환
    label_counts_ko = {split: {label_map[int(k)]: v for k, v in counts.items()} for split, counts in label_counts.items()}
    
    df = pd.DataFrame(label_counts_ko).fillna(0)
    df = df.astype(int)
    
    # 디버깅 출력을 추가합니다.
    print("DataFrame for plotting:")
    print(df)
    
    df.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Label Distribution in Train, Validation, and Test Sets')
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.xticks(rotation=45)
    plt.legend(title='Dataset Split')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# JSON 파일에서 라벨 이름을 읽어오는 함수
def load_label_names(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    return {item['no']: item['en_cls'] for item in label_data['labels']}


# Example usage:
data_dir = 'datas/regulation_datas'
dataset_dir = 'datasets/recyclables'
json_file = 'resources/labels_data_cls.json'
# organize_dataset(data_dir, dataset_dir)

# Count labels in each dataset split
label_counts = count_labels(os.path.join(dataset_dir, 'labels'))

# Load label names from JSON
label_map = load_label_names(json_file)

# Plot and save the label distribution
plot_label_distribution(label_counts, label_map, 'label_distribution.png')