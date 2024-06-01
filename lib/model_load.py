from dotenv import load_dotenv
import gdown
import shutil
import os

from lib.logger_config import setup_logger

load_dotenv()
MODEL_ID = os.getenv('MODEL_ID')
logger = setup_logger()


def model_download(resource_dir):
    logger.info("Starting model download...")
    file_name = gdown.download(id=MODEL_ID, quiet=True)
    shutil.move(file_name, resource_dir)
    return file_name


def model_load():
    logger.info("Starting model load process...")
    resource_dir = os.path.join('resources', 'models')
    os.makedirs(resource_dir, exist_ok=True)

    files = os.listdir(resource_dir)
    pt_files = [file for file in files if file.endswith('.pt')]

    file_name = model_download(resource_dir) if not pt_files else pt_files[-1]

    logger.info(f"Using model file: {file_name}")
    return os.path.join(resource_dir, file_name)