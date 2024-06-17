from typing import Dict
from ultralytics import YOLO

from lib.model_load import load_all_models
from lib.logger_config import setup_logger

logger = setup_logger()
models: Dict[str, YOLO] = {}

def load_models(ext='onnx'):
    global models
    models = load_all_models(ext)

def get_models() -> Dict[str, YOLO]:
    return models

def clear_all_models():
    models.clear()
    logger.info("Cleared all models.")
