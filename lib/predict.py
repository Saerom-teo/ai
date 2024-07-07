from ultralytics import YOLO
from typing import Dict, List
import os
import requests
from ultralytics.engine.results import Results

from lib.const import DEFAULT_MODEL, RESULT_SAVE_DIR, UPLOAD_DIR
from lib.logger_config import setup_logger
from lib.image_controll import predict_summary
from lib.upload_image import upload_image
from domain.predict_schema import PredictionRequest


logger = setup_logger()

async def predict(models: Dict[str, YOLO], request: PredictionRequest):
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

    # Save the results
    result_images = upload_image(RESULT_SAVE_DIR, results)

    return {"results": results, "result_images": result_images, "predict_summary": predict_summary(results, model_name)}


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

        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            images.append(url)
    
    return images