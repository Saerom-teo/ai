from fastapi import UploadFile
from ultralytics import YOLO
from typing import Dict
import os

from lib.predict import predict
from .predict_schema import PredictionRequest, PredictionResponse
from lib.const import UPLOAD_DIR

async def predict_json(models: Dict[str, YOLO], request: PredictionRequest):    
    result = await predict(models, request)

    response_data  = PredictionResponse(
        result='clear',
        images=result['result_images']
    )
    return response_data

async def predict_image(models, file):
    file_path = await save_uploaded_file(file)

    data_dict = {"images": [file_path]}
    request = PredictionRequest(**data_dict)
    
    result = await predict(models, request)
    os.remove(file_path)

    response_data  = PredictionResponse(
        result='clear',
        images=result['result_images']
    )
    return response_data


async def save_uploaded_file(uploaded_file: UploadFile) -> str:
    upload_dir = UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await uploaded_file.read())
    return file_path