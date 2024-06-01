from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO
import io

from lib.logger_config import setup_logger
from lib.model_load import model_load


app = FastAPI()
logger = setup_logger()
model = YOLO(model_load())


@app.post("/predict")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model.predict(image, conf=0.25)
    print(results)


    return {"info": f"received image with shape "}