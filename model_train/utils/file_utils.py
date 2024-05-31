import os
import shutil
import json
import cv2
import numpy as np

def search_label(labels_path, name):
    
    # labels_path = "datas/recyclables_origin/labels/"
    folder_list = os.listdir(labels_path)

    for folder in folder_list:
        folder_list = os.listdir(os.path.join(labels_path, folder))
        if name in folder_list:
            return folder + "/" + name
        
def copy_to_parent_directory(file_path, target_path):
    os.makedirs(target_path, exist_ok=True)
    abs_file_path = os.path.abspath(file_path)
    file_name = os.path.basename(abs_file_path)
    destination_path = os.path.join(target_path, file_name)
    
    shutil.copy2(abs_file_path, destination_path)
    
    print(f"{destination_path}이 {target_path} 로 복사되었습니다.")    

def extract_points_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Label file '{file_path}' does not exist.")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    data_list = []
    annotation_info = data.get("ANNOTATION_INFO", [])
    # print("annotation_info: " + str(len(annotation_info)))
    for annotation in annotation_info:
        cls = annotation.get("CLASS", "")
        detail = annotation.get("DETAILS", "")
        shape_type = annotation.get("SHAPE_TYPE", "")
        points = annotation.get("POINTS", [])

        data_dic = {"cls": cls, "detail": detail, "shape_type": shape_type, "points": points}
        data_list.append(data_dic)
    if not data_list:
        print(f"Error: No points found in the label file '{file_path}'")
        return None
    
    return data_list

def to_center_coordinates(box):
    """
    입력된 [x, y, w, h] 배열을 [cx, cy, w, h] 배열로 변환합니다.
    
    :param box: 리스트 [x, y, w, h], 
                x와 y는 왼쪽 위 모서리 좌표, 
                w는 너비, 
                h는 높이.
    :return: 리스트 [cx, cy, w, h], 
             cx와 cy는 중심 좌표, 
             w는 너비, 
             h는 높이.
    """
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2
    return [cx, cy, w, h]

def to_center_coordinates_2d(boxes):
    x, y, w, h = boxes
    return [x + w / 2, y + h / 2, w, h]

def draw_rectangle(image_path, points, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not open or find the image '{image_path}'")
        return
    
    x, y, w, h = points
    
    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
    
    output_file_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_file_path, image)
    print(f"Image saved with rectangle at '{output_file_path}'")

def draw_rectangle_to_center(image_path, points_list, output_dir, result_name):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not open or find the image '{image_path}'")
        return
    
    for points in points_list:
        cx, cy, w, h = points
        
        x = cx - w / 2
        y = cy - h / 2
        
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
    
    output_file_path = os.path.join(output_dir, result_name + os.path.splitext(image_path)[1])
    cv2.imwrite(output_file_path, image)
    # print(f"Image saved with rectangles at '{output_file_path}'")

def draw_rectangle_from_ratios(image_path, ratios_list, output_dir, result_name):
    """
    이미지 경로와 비율로 주어진 [cx, cy, w, h] 값의 리스트를 받아서, 이미지에 사각형을 그린 후 저장합니다.
    
    Parameters:
        image_path (str): 이미지 파일의 경로.
        ratios_list (list of lists): 비율로 주어진 [cx, cy, w, h] 값의 리스트.
        output_dir (str): 출력 디렉토리 경로.
        result_name (str): 결과 이미지 파일 이름.
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image '{image_path}'")
        return
    
    # 이미지의 가로 및 세로 길이 얻기
    height, width, _ = image.shape
    
    # 각 비율값을 실제 좌표로 변환하여 사각형 그리기
    for ratios in ratios_list:
        cx_ratio, cy_ratio, w_ratio, h_ratio = ratios
        
        # 비율값을 실제 좌표값으로 변환
        cx = cx_ratio * width
        cy = cy_ratio * height
        w = w_ratio * width
        h = h_ratio * height
        
        # 사각형의 좌상단 좌표 계산
        x = cx - w / 2
        y = cy - h / 2
        
        # 사각형 그리기
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
    
    # 결과 이미지 저장
    output_file_path = os.path.join(output_dir, result_name + os.path.splitext(image_path)[1])
    cv2.imwrite(output_file_path, image)
    # print(f"Image saved with rectangles at '{output_file_path}'")

def resize_image(image_path, output_dir, result_name):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image '{image_path}'")
        return None, None
    
    height, width = image.shape[:2]

    if width < height:
        scale = 640 / width
    else:
        scale = 640 / height

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    output_file_path = os.path.join(output_dir, result_name + os.path.splitext(image_path)[1])

    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(output_file_path, resized_image)
    # print(f"Resized image saved at '{output_file_path}'")

    return output_file_path, scale

def list_to_txt(label_no_list, two_d_list, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        for num, row in zip(label_no_list, two_d_list):
            line = str(num) + ' ' + ' '.join(map(str, row))
            file.write(line + '\n')

def draw_polygon(image_path, points_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not open or find the image '{image_path}'")
        return
    
    points_array = np.array(points_list, np.int32)

    cv2.polylines(image, [points_array], isClosed=True, color=(0, 255, 0), thickness=2)
    
    output_file_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_file_path, image)
    # print(f"Image saved with polygon at '{output_file_path}'")

def polygon_to_bounding_box(points_list):
    min_x = min(points_list, key=lambda x: x[0])[0]
    min_y = min(points_list, key=lambda x: x[1])[1]
    max_x = max(points_list, key=lambda x: x[0])[0]
    max_y = max(points_list, key=lambda x: x[1])[1]
    
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    w = max_x - min_x
    h = max_y - min_y
    
    return [cx, cy, w, h]

def check_data(ko_cls, ko_detail, file_path="lable_name.json"):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for label in data.get("labels", []):
        if label.get("ko_cls") == ko_cls and label.get("ko_detail") == ko_detail:
            return {"no": label.get("no"), "en_cls": label.get("en_cls"), "en_detail": label.get("en_detail")}
    
    return False

def get_center_points(datas, scale, resize_name, output_path, file_info):
    label_num_list = []
    label_name_list = []
    center_points_list = []
    for data in datas:
        en_data = check_data(data["cls"], data["detail"], file_path=file_info)
        label_no = en_data["no"]
        label_name = en_data["en_cls"] + "_" + en_data["en_detail"]
        point_array = [[point*scale for point in points] for points in data["points"]]

        if data["shape_type"] == "BOX":
            center_points = to_center_coordinates_2d(point_array[0])
        elif data["shape_type"] == "POLYGON":
            draw_polygon(resize_name, point_array, os.path.join(output_path, "annotation", label_name, "poly_to_box"))
            center_points = polygon_to_bounding_box(point_array)
        else:
            raise Exception(f'shape_type : {data["shape_type"]}')

        label_num_list.append(label_no)
        label_name_list.append(label_name)
        center_points_list.append(center_points)
    return label_num_list, label_name_list, center_points_list

def get_relative_coordinates(image_path, coordinates_list):
    """
    이미지 경로와 [cx, cy, w, h] 값의 리스트를 받아서, 이미지의 크기에 대한 비율로 변환된 값을 반환합니다.
    
    Parameters:
        image_path (str): 이미지 파일의 경로.
        coordinates_list (list of lists): [cx, cy, w, h] 값의 리스트.
        
    Returns:
        list of lists: 이미지 크기에 대한 비율로 변환된 [cx, cy, w, h] 값의 리스트.
    """
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")
    
    # 이미지의 가로 및 세로 길이 얻기
    height, width, _ = image.shape
    
    # 비율 변환된 값을 저장할 리스트
    relative_coordinates_list = []
    
    for coord in coordinates_list:
        cx, cy, w, h = coord
        relative_cx = cx / width
        relative_cy = cy / height
        relative_w = w / width
        relative_h = h / height
        relative_coordinates_list.append([relative_cx, relative_cy, relative_w, relative_h])
    
    return relative_coordinates_list