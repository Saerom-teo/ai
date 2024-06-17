from typing import Dict
from ultralytics import YOLO
import os
import asyncio

from lib.model_manager import load_models, get_models
from domain.predict_service import predict
from domain.predict_schema import PredictionRequest
from lib.image_controll import combine_images_grid

load_models()
models: Dict[str, YOLO] = get_models()

async def test(model, result_file_path, base_path):
    file_path = os.path.join("resources", "test_images")
    files = os.listdir(file_path)

    results = []
    for file in files:
        file_name = os.path.join(file_path, file)
        data_dict = {"modelName": model, "images": [file_name]}
        request = PredictionRequest(**data_dict)
        result = await predict(models, request, save=True)
        results.append(result['result_images'][0])

    combine_images_grid(results, f'{base_path}/{os.path.basename(model)}.jpg')

    # exec_times = []
    # for file in files*6:
    #     file_name = os.path.join(file_path, file)
    #     data_dict = {"modelName": model, "images": [file_name]}
    #     request = PredictionRequest(**data_dict)
    #     result = await predict(models, request, save=True)
    #     speed = result['results'][0].speed
    #     exec_time = round(speed['preprocess'] + speed['inference'] + speed['postprocess'], 2)
    #     exec_times.append(exec_time)

    # avg_exec_time = round(sum(exec_times) / len(exec_times), 2)
    
    # with open(result_file_path, "a") as result_file:
    #     result_file.write("================================================================\n")
    #     result_file.write(f"{model}, avg_exec_time: {avg_exec_time}\n")
    #     result_file.write("================================================================\n")

if __name__ == "__main__":
    base_path = "static/test"
    result_file_name = "execution_times.txt"
    result_file_path = os.path.join(base_path, result_file_name)

    os.makedirs(base_path, exist_ok=True)
    if os.path.exists(result_file_path):
        os.remove(result_file_path)

    for model_name in models.keys():
        asyncio.run(test(model_name, result_file_path, base_path))

# python -m docs.model_test