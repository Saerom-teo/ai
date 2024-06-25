from fastapi import APIRouter, UploadFile, File, Depends
from ultralytics import YOLO
from typing import Dict

from lib.model_manager import get_models
from lib.logger_config import setup_logger

from .predict_service import predict_json, predict_image
from .predict_schema import PredictionRequest, PredictionResponse

logger = setup_logger()
router = APIRouter()

@router.post("/json", response_model=PredictionResponse)
async def upload_json(request: PredictionRequest, models: Dict[str, YOLO] = Depends(get_models)):
    response_data = await predict_json(models, request)
    return response_data

@router.post("/image", response_model=PredictionResponse)
async def upload_image(file: UploadFile = File(...), models: Dict[str, YOLO] = Depends(get_models)):
    response_data = await predict_image(models, file)
    return response_data

@router.post("/test", response_model=PredictionResponse)
async def upload_json(request: PredictionRequest, models: Dict[str, YOLO] = Depends(get_models)):
    logger.info("📍 Handling test endpoint")
    response = {
        "result": "clear",
        "images": [
            "static/results/1718609562963.jpg.jpg",
            "static/results/1718609563649.jpg.jpg",
            "static/results/1718609563762.jpg.jpg",
            "static/results/1718609563959.jpg.jpg",
            "static/results/1718609564154.jpg"
        ]
    }

    return response