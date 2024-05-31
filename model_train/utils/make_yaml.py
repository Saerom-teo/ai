import yaml, os, json


def extract_points_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Label file '{file_path}' does not exist.")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    label_info = data.get("labels", [])

    data_dic = {}
    for label in label_info:
        no = label.get("no", "")
        en_cls = label.get("en_cls", "")
        en_detail = label.get("en_detail", "")
        data_dic[int(no)] = f"{en_cls}_{en_detail}"
    
    if not data_dic:
        print(f"Error: No points found in the label file '{file_path}'")
        return None
    
    return data_dic

def make_yaml(data_name, name):
    data_dic = extract_points_from_file(data_name)
    if not data_dic:
        print("No data found. YAML file will not be created.")
        return

    data = {
        "train" : f'./{name}/images/train',
        "val" : f'./{name}/images/val',
        "test" : f'./{name}/images/test',
        "names" : data_dic
    }

    with open(f'./datasets/{name}/{name}.yaml', 'w') as f:
        yaml.dump(data, f)

    print("YAML file 'recyclables.yaml' has been created successfully.")



if __name__ == "__main__":
    working_directory = os.getcwd()
    label_file = os.path.join(working_directory, "label_name.json")
    
    make_yaml(label_file, "recyclables")