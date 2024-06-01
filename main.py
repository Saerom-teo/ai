from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
from ultralytics import YOLO
import io

from lib.logger_config import setup_logger
from lib.model_load import model_load
from lib.image_controll import draw_boxes, predict_summary


app = FastAPI()
logger = setup_logger()
model = YOLO(model_load())


@app.post("/predict")
async def upload_image(file: UploadFile = File(...)):
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