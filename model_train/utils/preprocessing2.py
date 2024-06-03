import os
from tqdm import tqdm
import file_utils
import shutil


def get_file_idx(output_path):
    """
    Create 'images' directory in output path if not exists.
    Return the current file index based on the number of files in 'images' directory.
    """
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    file_num = len(os.listdir(os.path.join(output_path, "images")))
    return file_num - 1 if file_num != 0 else 0

def append_to_file(filename, text):
    """
    Append the given text to the specified file.
    """
    with open(filename, 'a', encoding='utf-8') as file:
        file.write('\n' + text)

def process_image_folder(folder, image_folder, label_folder, output_path, label_info):
    """
    Process a single image folder: resize images, extract labels, and create annotations.
    """
    images_path = os.path.join(image_folder, folder)
    file_list = os.listdir(images_path)[:]
    file_num = get_file_idx(output_path)
    file_index = 0
    
    print(f'preprocessing... folder: {folder}, len: {len(file_list)}')
    for file in tqdm(file_list, total=len(file_list), desc='images'.ljust(13), leave=True):
        result_name = str(file_num + file_index).zfill(7)
        file_index += 1

        name = os.path.splitext(file)[0]

        image_file = os.path.join(images_path, file)
        label_file = os.path.join(label_folder, file_utils.search_label(label_folder, name + ".json"))

        datas = file_utils.extract_points_from_file(label_file)

        # Check if label exists in label_info
        if any(not file_utils.check_data2(data["cls"], file_path=label_info) for data in datas):
            for data in datas:
                if not file_utils.check_data2(data["cls"], file_path=label_info):
                    # append_to_file("label_not_include.txt", data["cls"])
                    # print(f'label is not exist. class: {data["cls"]}')
                    pass
            continue  # Skip to next file if any label is not found
        
        data_dict = file_utils.check_data2(datas[0]["cls"], file_path=label_info)
        resize_name, scale = file_utils.resize_image(image_file, os.path.join(output_path, "images", data_dict['en_cls']), result_name)
        if resize_name is None and scale is None:
            # print(f"file error / name: {image_file}")
            continue

        label_num_list, label_name_list, center_points_list = file_utils.get_center_points2(datas, scale, resize_name, output_path, label_info)
        # if len(label_name_list) > 1:
        #     append_to_file("image_in_over_2_label_type.txt", resize_name)
        
        relative_coordinates_list = file_utils.get_relative_coordinates(resize_name, center_points_list)
        
        file_utils.list_to_txt(label_num_list, relative_coordinates_list, os.path.join(output_path, "labels", data_dict['en_cls']), result_name + ".txt")
        file_utils.draw_rectangle_from_ratios(resize_name, relative_coordinates_list, os.path.join(output_path, "annotation", label_name_list[0], "center"), result_name)

def preprocessing(path, label_info):
    """
    Main preprocessing function to process all image folders and their images.
    """
    # try:
    #     shutil.rmtree(path)
    #     print(f"Successfully deleted the folder: {path}")
    # except Exception as e:
    #     print(f"Error deleting folder {path}: {e}")

    image_folder = f"{path}/images"
    label_folder = f"{path}/labels"
    folder_list = sorted(os.listdir(image_folder), reverse=True)[:]

    # pbar = tqdm(folder_list, total=len(folder_list), desc='image_folders', ncols=90, position=0)

    # for folder in pbar:
    for folder in folder_list:
        # pbar.set_postfix({'folder_name': folder})
        process_image_folder(folder, image_folder, label_folder, output_path, label_info)
    
    # pbar.close()

if __name__ == "__main__":
    path = 'E:/temp/recyclables_origin'
    output_path = "datas/regulation_datas_2"

    working_directory = os.getcwd()
    path = os.path.join(working_directory, path)
    output_path = os.path.join(working_directory, output_path)
    label_info = os.path.join(working_directory, "resources", "labels_data_cls.json")

    preprocessing(path, label_info)
