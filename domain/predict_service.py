from ultralytics import YOLO
from fastapi import UploadFile
from typing import Dict, List
import os
import requests
from ultralytics.engine.results import Results

from lib.const import DEFAULT_MODEL, RESULT_SAVE_DIR, UPLOAD_DIR
from lib.logger_config import setup_logger
from lib.image_controll import predict_summary
from domain.predict_schema import PredictionRequest


logger = setup_logger()

async def predict(models: Dict[str, YOLO], request: PredictionRequest, save=False):
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
    result_images = []
    if save:
        save_dir = RESULT_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)
        for result in results:
            image_path = os.path.join(save_dir, f"{os.path.basename(result.path)}.jpg")
            result.save(filename=image_path)
            result_images.append(image_path)

    return {"results": results, "result_images": result_images, "predict_summary": predict_summary(results, model_name)}

async def save_uploaded_file(uploaded_file: UploadFile) -> str:
    upload_dir = UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await uploaded_file.read())
    return file_path

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
    
    return images