from ultralytics import YOLO
from fastapi import UploadFile
from typing import Dict
import os

from lib.const import DEFAULT_MODEL, RESULT_SAVE_DIR, UPLOAD_DIR
from lib.logger_config import setup_logger
from lib.image_controll import predict_summary
from domain.predict_schema import PredictionRequest

logger = setup_logger()

async def predict(models: Dict[str, YOLO], request: PredictionRequest, save=False):
    model_name = request.modelName if request.modelName else DEFAULT_MODEL
    model = models.get(model_name)

    if not model:
        raise ValueError(f"Model {model_name} not found in models dictionary")

    results = model.predict(request.images, conf=0.25, verbose=False)
    logger.info(f"📌 Prediction results - {predict_summary(results, model_name)}")

    # Save the results
    result_images = []
    if save:
        save_dir = RESULT_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)
        for result in results:
            image_path = os.path.join(save_dir, f"{os.path.splitext(model_name)[0]}_{os.path.basename(result.path)}")
            result.save(filename=image_path)
            result_images.append(image_path)

    return {"result_images": result_images}

async def save_uploaded_file(uploaded_file: UploadFile) -> str:
    upload_dir = UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await uploaded_file.read())
    return file_path