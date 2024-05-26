import os
import json

from lib import file_utils



def get_file_idx(output_path):
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    file_num = len(os.listdir(os.path.join(output_path, "images")))
    return  file_num-1 if file_num != 0 else 0


path = f"recyclables_origin/images/images02"
path2 = f"recyclables_origin/labels"

output_path = "test"

file_list = os.listdir(path)
file_num = get_file_idx(output_path)
for idx, file in enumerate(file_list[:]):
    result_name = str(file_num + idx).zfill(6)
    print("result_name: " + result_name)
    name = os.path.splitext(file)[0]

    image_file = os.path.join(path, file)
    label_file = os.path.join(path2, file_utils.search_label(name + ".json"))

    datas = file_utils.extract_points_from_file(label_file)

    for data in datas:
        if not file_utils.check_data(data["cls"], data["detail"]):
            raise Exception(f'label is not exist. class: {data["cls"]}, detail: {data["detail"]}')

    resize_name, scale = file_utils.resize_image(image_file, os.path.join(output_path, "images"), result_name)

    label_num_list, label_name_list, center_points_list = file_utils.get_center_points(datas, scale, resize_name, output_path)

    file_utils.list_to_txt(label_num_list, center_points_list, os.path.join(output_path, "labels"), result_name + ".txt")
    file_utils.draw_rectangle_to_center(resize_name, center_points_list, os.path.join(output_path, "annotation", label_name_list[0], "center"), result_name)