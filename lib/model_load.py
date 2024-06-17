import gdown
import os, json
from ultralytics import YOLO
from typing import List, Dict, Any

from lib.logger_config import setup_logger


logger = setup_logger()


def model_download(name: str, id: str, resource_dir: str) -> str:
    logger.info("Starting model download...")
    
    file_name = None
    try:
        file_name = gdown.download(id=id, output=os.path.join(resource_dir, name), quiet=True)
        logger.info(f"Downloaded {file_name}")
    except gdown.exceptions.FileURLRetrievalError as e:
        logger.error(f"Failed to download the model - {name}")
    
    return file_name

def make_model_dict(info: Dict[str, Any], resource_dir: str, ext: str) -> Dict[str, YOLO]:
    files = [file for file in os.listdir(resource_dir) if file.endswith(f'.{ext}')]

    model_dict = {}
    for model in info['model_list']:
        if model['use']:
            name = f"{model['name']}.{ext}"
            file_name = os.path.join(resource_dir, name) if name in files else model_download(name, model[ext], resource_dir)
            if file_name:
                task = model.get('task', 'detect')
                model_dict[name] = YOLO(file_name, task=task)
    
    return model_dict


def load_all_models(ext: str) -> Dict[str, YOLO]:
    logger.info("Starting model load process...")
    resource_dir = os.path.join('resources', 'models')
    os.makedirs(resource_dir, exist_ok=True)

    with open('resources/model_info.json') as f:
        info = json.load(f)

    if ext == 'pt':
        model_dict = make_model_dict(info, resource_dir, ext='pt')
    else:
        model_dict = make_model_dict(info, resource_dir, ext='onnx')

    logger.info(f"💡 Completed loading all models: {list(model_dict.keys())}")
    return model_dict


if __name__ == "__main__":
    file_names = load_all_models()
    print(file_names)