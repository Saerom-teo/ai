from PIL import Image, ImageDraw, ImageFont
from ultralytics.engine.results import Results
from typing import List
from collections import defaultdict

def draw_boxes(image, results: List[Results]) -> Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size=20)
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


def predict_summary(results: List[Results]):
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
    return ' '.join(result_messages)