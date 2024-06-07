from fastapi import APIRouter

from lib.model_manager import get_models
from lib.logger_config import setup_logger


logger = setup_logger()
router = APIRouter()

@router.get("/model-list")
async def model_list():
    models = get_models()

    response = {
        "model_list": list(models.keys())
    }
    
    return response