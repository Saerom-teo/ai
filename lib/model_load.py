import gdown
import os, json
from ultralytics import YOLO

from lib.logger_config import setup_logger


logger = setup_logger()


def model_download(model, resource_dir):
    logger.info("Starting model download...")
    
    file_name = gdown.download(id=model['id'], output=os.path.join(resource_dir, model['name']), quiet=True)
    logger.info(f"Downloaded {file_name}")
    
    return file_name


def load_all_models():
    logger.info("Starting model load process...")
    resource_dir = os.path.join('resources', 'models')
    os.makedirs(resource_dir, exist_ok=True)

    with open('resources/model_info.json') as f:
        info = json.load(f)

    files = os.listdir(resource_dir)
    pt_files = [file for file in files if file.endswith('.pt')]

    model_dict = {}
    for model in info['model_list']:
        file_name = os.path.join(resource_dir, model['name']) if model['name'] in pt_files else model_download(model, resource_dir)
        model_dict[model['name']] = YOLO(file_name)

    logger.info(f"💡 Completed loading all models: {list(model_dict.keys())}")
    return model_dict


if __name__ == "__main__":
    file_names = load_all_models()
    print(file_names)