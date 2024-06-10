from fastapi import APIRouter, UploadFile, File, Depends
from ultralytics import YOLO
import os
from typing import Dict

from lib.model_manager import get_models
from lib.logger_config import setup_logger
from domain.predict_service import predict, save_uploaded_file
from domain.predict_schema import PredictionRequest

logger = setup_logger()
router = APIRouter()

@router.post("/json")
async def upload_json(request: PredictionRequest, models: Dict[str, YOLO] = Depends(get_models)):
    result = await predict(models, request, save=True)
    return result

@router.post("/image")
async def upload_image(file: UploadFile = File(...), models: Dict[str, YOLO] = Depends(get_models)):
    file_path = await save_uploaded_file(file)

    data_dict = {"images": [file_path]}
    request = PredictionRequest(**data_dict)
    
    result = await predict(models, request, save=True)
    os.remove(file_path)
    return result