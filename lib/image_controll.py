from PIL import Image, ImageDraw, ImageFont
from ultralytics.engine.results import Results
from typing import List
from collections import defaultdict
import math

def draw_boxes(image, results: List[Results]) -> Image:
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except OSError:
        font_path = "/workspace/resources/font/Arial.ttf"
        font = ImageFont.truetype(font_path, size=20)

    for result in results:
        names = result.names
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            tag = names.get(cls, "Unknown")

            draw.rectangle([(xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])], outline="red", width=3)

            text = f"Tag: {tag}, Conf: {conf:.2f}"
            draw.text((xyxy[0], xyxy[1] - 20), text, fill="red", font=font)
    
    return image


def predict_summary(results: List[Results], model_name: str):
    result_messages = []
    for index, result in enumerate(results):
        names = result.names
        boxes = result.boxes
        total_time_ms = sum(result.speed.values())
        total_time_str = f"{total_time_ms:.2f} ms"
        
        result_dic = defaultdict(int)
        for box in boxes:
            cls = box.cls[0].item()
            tag = names.get(cls, "Unknown")
            result_dic[tag] += 1

        image_shape_str = "x".join(map(str, result.orig_shape))
        total_type_num_str = ", ".join([f"{key} {value}" for key, value in dict(result_dic).items()])
        result_messages.append(f'{index}: {image_shape_str} {total_type_num_str}, {total_time_str}')
    return model_name + ' '+ ' '.join(result_messages)

def combine_images_grid(image_paths, output_path='combined_image.jpg'):
    # 이미지 리스트 불러오기
    images = [Image.open(path) for path in image_paths]

    # 이미지 개수
    num_images = len(images)

    # 격자 크기 계산
    grid_rows = math.floor(math.sqrt(num_images))
    grid_cols = math.ceil(num_images / grid_rows)
    
    # 각각의 이미지 크기 불러오기
    widths, heights = zip(*(i.size for i in images))

    # 각 행과 열에서 최대 너비와 높이 계산
    max_width = max(widths)
    max_height = max(heights)
    
    # 전체 이미지의 너비와 높이 계산
    total_width = grid_cols * max_width
    total_height = grid_rows * max_height

    # 새로운 빈 이미지 생성
    combined_image = Image.new('RGB', (total_width, total_height))

    # 이미지 격자에 붙이기
    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        x_offset = col * max_width
        y_offset = row * max_height
        combined_image.paste(img, (x_offset, y_offset))

    # 합쳐진 이미지 저장
    combined_image.save(output_path)
    print(f'Combined image saved to {output_path}')

# 사용 예제
# image_paths = [
#     'static/results/yolov8n_0604_e50_KakaoTalk_20240531_142144938_04.jpg',
#     'static/results/yolov8n_0604_e50_KakaoTalk_20240611_133910242_01.jpg',
#     'static/results/yolov8n_0604_e50_KakaoTalk_20240611_133910242_02.jpg',
#     'static/results/yolov8n_0604_e50_KakaoTalk_20240611_133910242_03.jpg',
#     'static/results/yolov8n_0604_e50_KakaoTalk_20240611_133910242.jpg'
# ]
# image_paths = [
#     'static/results/yolov8n_0607_e50_b128_KakaoTalk_20240531_142144938_04.jpg',
#     'static/results/yolov8n_0607_e50_b128_KakaoTalk_20240611_133910242_01.jpg',
#     'static/results/yolov8n_0607_e50_b128_KakaoTalk_20240611_133910242_02.jpg',
#     'static/results/yolov8n_0607_e50_b128_KakaoTalk_20240611_133910242_03.jpg',
#     'static/results/yolov8n_0607_e50_b128_KakaoTalk_20240611_133910242.jpg'
# ]
# image_paths = [

# ]
# combine_images_grid(image_paths, output_path='combined_image_grid.jpg')