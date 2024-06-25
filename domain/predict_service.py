from ultralytics import YOLO
from typing import Dict
import os

from lib.predict import predict, save_uploaded_file
from .predict_schema import PredictionRequest, PredictionResponse

async def predict_json(models: Dict[str, YOLO], request: PredictionRequest):    
    result = await predict(models, request, save=True)

    response_data  = PredictionResponse(
        result='clear',
        images=result['result_images']
    )
    return response_data

async def predict_image(models, file):
    file_path = await save_uploaded_file(file)

    data_dict = {"images": [file_path]}
    request = PredictionRequest(**data_dict)
    
    result = await predict(models, request, save=True)
    os.remove(file_path)

    response_data  = PredictionResponse(
        result='clear',
        images=result['result_images']
    )
    return response_data