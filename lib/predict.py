from ultralytics import YOLO
from typing import Dict, Tuple, List
import os
import requests
import numpy as np
import cv2
from ultralytics.engine.results import Results

from lib.const import DEFAULT_MODEL, RESULT_SAVE_DIR, UPLOAD_DIR
from lib.logger_config import setup_logger
from lib.image_controll import predict_summary
from lib.upload_image import upload_image, upload_to_s3
from domain.predict_schema import PredictionRequest


logger = setup_logger()

async def predict(models: Dict[str, YOLO], request: PredictionRequest) -> List[Results]:
    model_name = request.modelName if request.modelName else DEFAULT_MODEL
    model = models.get(model_name)

    images = download_images(request.images)

    if not model:
        raise ValueError(f"Model {model_name} not found in models dictionary")
    
    results: List[Results] = []
    for image in images:
        result = model.predict(image, conf=0.25, verbose=False)
        results.append(result[0])
    
    logger.info(f"📌 Prediction results - {predict_summary(results, model_name)}")
    return results

        # url = upload_to_s3(image_path)

        # result_images.append(url)

    # result_images = upload_image(RESULT_SAVE_DIR, results)


    # return {"results": results, "result_images": result_images, "predict_summary": predict_summary(results, model_name)}


def download_images(urls: List[str]):
    save_dir = UPLOAD_DIR
    os.makedirs(save_dir, exist_ok=True)

    images = []
    for url in urls:
        name = url.split("/")[-1]
        save_path = os.path.join(save_dir, name)
        try:
            response = requests.get(url)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                f.write(response.content)

            images.append(save_path)

        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            images.append(url)
    
    return images


def result_analyze(results: List[Results]):
    result_images = []
    filtered_cls_list = []
    for result in results:
        filtered_cls, filtered_xywh = filter_classes(result.boxes.cls.cpu().numpy(), result.boxes.xywh.cpu().numpy(), [3, 4])

        image_path = result.path
        result_image_path = draw_circle_on_image(image_path, filtered_xywh)

        url = upload_to_s3(result_image_path)
        result_images.append(url)
        filtered_cls_list.extend(filtered_cls)

    return filtered_cls_list, result_images


def filter_classes(cls: np.ndarray, xywh: np.ndarray, exclude_classes: list) -> Tuple[List[int], np.ndarray]:
    mask = np.isin(cls, exclude_classes, invert=True)
    filtered_cls = cls[mask].astype(int).tolist()
    filtered_xywh = xywh[mask]
    return filtered_cls, filtered_xywh

def draw_circle_on_image(image_path: str, xywh: np.ndarray) -> str:
    os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    for box in xywh:
        x, y, w, h = box
        center_x, center_y = int(x), int(y)
        radius = int(min(w, h) / 2)
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 255), 3)

    result_image_path = os.path.join(RESULT_SAVE_DIR, f"result_{os.path.basename(image_path)}")
    cv2.imwrite(result_image_path, image)
    return result_image_path