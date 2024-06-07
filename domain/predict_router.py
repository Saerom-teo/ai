from fastapi import APIRouter, UploadFile, Response, File, Depends
from ultralytics import YOLO
from PIL import Image
import io, os
from typing import Dict

from lib.model_manager import get_models
from lib.logger_config import setup_logger
from lib.image_controll import draw_boxes, predict_summary

from schema.predict_schema import PredictionRequest


logger = setup_logger()
router = APIRouter()


@router.post("/predict-json")
async def upload_image(request: PredictionRequest, 
                       models: Dict[str, YOLO] = Depends(get_models)):
    save_dir = 'static/results/'
    os.makedirs(save_dir, exist_ok=True)

    model = models.get('yolov8n_0531_e30.pt')
    results = model.predict(request.images, conf=0.25, verbose=False)
    
    for result in results:
        image_name = os.path.basename(result.path)
        result.save(filename=os.path.join(save_dir, image_name))
    
    logger.info(f"📌 Prediction results - {predict_summary(results)}")

    return {"predictions": predict_summary(results), "save_dir": save_dir}


@router.post("/predict-image")
async def upload_image(file: UploadFile = File(...), models: Dict[str, YOLO] = Depends(get_models)):
    model = models.get('yolov8n_0531_e30.pt')
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model.predict(image, conf=0.25, verbose=False)
    logger.info(f"📌 Prediction results - {predict_summary(results)}")

    # Draw boxes on the image
    image_with_boxes = draw_boxes(image, results)

    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image_with_boxes.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")