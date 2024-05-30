import os
from tqdm import tqdm

import file_utils


def get_file_idx(output_path):
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    file_num = len(os.listdir(os.path.join(output_path, "images")))
    return  file_num-1 if file_num != 0 else 0

def append_to_file(filename, text):
    with open(filename, 'a') as file:
        file.write('\n' + text)

def preprocessing(path, label_info):
    image_folder = f"{path}/images"
    label_folder = f"{path}/labels"

    folder_list = sorted(os.listdir(image_folder))

    pbar = tqdm(folder_list, total=len(folder_list), desc='image_folders', ncols=90, position=0)

    for folder in pbar:
        images_path = os.path.join(image_folder, folder)

        pbar.set_postfix({'folder_name': folder})

        file_list = os.listdir(images_path)
        file_num = get_file_idx(output_path)
        file_index = 0
        for file in tqdm(file_list, total=len(file_list), desc='images'.ljust(13), ncols=90, position=1, leave=False):
            result_name = str(file_num + file_index).zfill(7)
            file_index+=1

            name = os.path.splitext(file)[0]

            image_file = os.path.join(images_path, file)
            label_file = os.path.join(label_folder, file_utils.search_label(label_folder, name + ".json"))

            datas = file_utils.extract_points_from_file(label_file)

            check = False
            for data in datas:
                if not file_utils.check_data(data["cls"], data["detail"], file_path=label_info):
                    append_to_file("labeel_not_include.txt", data["cls"] + "_" + data["detail"])
                    check = True
                    print(f'label is not exist. class: {data["cls"]}, detail: {data["detail"]}')
                    break
            if check:
                break

            resize_name, scale = file_utils.resize_image(image_file, os.path.join(output_path, "images"), result_name)
            if resize_name==None and scale == None:
                print(f"file error / name: {image_file}")
                break
            label_num_list, label_name_list, center_points_list = file_utils.get_center_points(datas, scale, resize_name, output_path, label_info)

            relative_coordinates_list = file_utils.get_relative_coordinates(resize_name, center_points_list)
            
            file_utils.list_to_txt(label_num_list, relative_coordinates_list, os.path.join(output_path, "labels"), result_name + ".txt")
            file_utils.draw_rectangle_from_ratios(resize_name, relative_coordinates_list, os.path.join(output_path, "annotation", label_name_list[0], "center"), result_name)
    
    pbar.close()

if __name__ == "__main__":
    path = f"datas/recyclables_origin"
    output_path = "datas/regulation_datas"

    working_directory = os.getcwd()
    path = os.path.join(working_directory, path)
    output_path = os.path.join(working_directory, output_path)
    label_info = os.path.join(working_directory, "label_name.json")

    preprocessing(path, label_info)