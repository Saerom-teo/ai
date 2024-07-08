from fastapi import UploadFile
from ultralytics import YOLO
from typing import Dict
import os

from lib.predict import predict, result_analyze
from .predict_schema import PredictionRequest, PredictionResponse
from lib.const import UPLOAD_DIR, RESULT_SAVE_DIR


async def predict_json(models: Dict[str, YOLO], request: PredictionRequest):
    results = await predict(models, request)

    filtered_cls, result_images = result_analyze(results)

    response_data  = PredictionResponse(
        result='clear' if len(filtered_cls)==0 else 'deny',
        images=result_images
    )
    return response_data

async def predict_image(models, file):
    file_path = await save_uploaded_file(file)
    file_path = file_path.replace("\\", "/")
    data_dict = {"images": [file_path]}
    request = PredictionRequest(**data_dict)
    
    results = await predict(models, request)

    filtered_cls, result_images = result_analyze(results)

    response_data  = PredictionResponse(
        result='clear' if len(filtered_cls)==0 else 'deny',
        images=result_images
    )
    return response_data

async def predict_all(models: Dict[str, YOLO], file):
    file_path = await save_uploaded_file(file)
    file_path = file_path.replace("\\", "/")

    for model_name in models.keys():
        print(model_name)
        data_dict = {"modelName": model_name,"images": [file_path]}
        request = PredictionRequest(**data_dict)
    
        results = await predict(models, request)

        # Save the results
        os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
        result_images = []
        for result in results:
            image_name = f"{os.path.splitext(os.path.basename(result.path))[0]}_{model_name.split('.')[0]}"
            print(image_name)
            image_path = os.path.join(RESULT_SAVE_DIR, f"{image_name}.jpg")
            result.save(filename=image_path)

    # response_data  = PredictionResponse(
    #     result='clear',
    #     images=result['result_images']
    # )
    # return response_data



async def save_uploaded_file(uploaded_file: UploadFile) -> str:
    upload_dir = UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await uploaded_file.read())
        return file_path