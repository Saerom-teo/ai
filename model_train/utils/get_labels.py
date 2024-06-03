import os
import json
from collections import defaultdict
from tqdm import tqdm
import file_utils

def get_labels(path):
    label_path = f"{path}/labels"
    label_dir = os.listdir(label_path)
    
    label_dic = defaultdict(int)
    for label_folder in label_dir:
        labels = os.listdir(os.path.join(label_path, label_folder))
        for label in tqdm(labels[:], desc=label_folder):
            label_file = os.path.join(label_path, label_folder, label)
            data_list = file_utils.extract_points_from_file(label_file)
            for data in data_list:
                # tag = f"{data['cls']}_{data['detail']}"
                tag = data['cls']
                label_dic[tag] += 1
    
    # Transform results to the desired JSON structure
    # result = {
    #     "labels": [
    #         {
    #             "no": idx,
    #             "ko_cls": tag.split('_')[0],
    #             "ko_detail": tag.split('_')[1],
    #             "count": count
    #         }
    #         for idx, (tag, count) in enumerate(label_dic.items())
    #     ]
    # }
    result = {
        "labels": [
            {
                "no": idx,
                "ko_cls": tag,
                "count": count
            }
            for idx, (tag, count) in enumerate(label_dic.items())
        ]
    }
    
    return result

def save_to_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    path = 'E:/temp/recyclables_origin'
    labels_data = get_labels(path)
    json_filepath = 'labels_data.json'
    save_to_json(labels_data, json_filepath)
    print(f"JSON file saved to {json_filepath}")