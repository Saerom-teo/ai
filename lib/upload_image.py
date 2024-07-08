from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError
import os
from ultralytics.engine.results import Results
from typing import List

from lib.logger_config import log_warning, log_error, log_info


load_dotenv()

ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
SECRET_KEY = os.getenv('S3_SECRET_KEY')
REGION_NAME = os.getenv('S3_REGION_NAME')
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

S3_ACCESS = True
if ACCESS_KEY is None or SECRET_KEY is None or REGION_NAME is None or BUCKET_NAME is None:
    log_warning("S3 credentials or configuration are missing. Cannot upload file.")
    S3_ACCESS = False
else:
    log_info("🔗 S3 credentials and configuration loaded successfully.")


def upload_image(save_dir: str, results: List[Results]):
    os.makedirs(save_dir, exist_ok=True)

    result_images = []

    for result in results:
        image_path = os.path.join(save_dir, f"{os.path.basename(result.path)}")
        result.save(filename=image_path)

        url = upload_to_s3(image_path)

        result_images.append(url)

    return result_images


def upload_to_s3(file_name, object_name=None):
    if not S3_ACCESS:
        return file_name
    log_info("⚔️ 왜 업로드가 안될까")
    if object_name is None:
        object_name = os.path.basename(file_name)

    try:
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

        object_name = f'AI/{object_name}'
        s3.upload_file(
            file_name, 
            BUCKET_NAME, 
            object_name, 
        )

        file_url = f"https://{BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/{object_name}"
        log_info(f"⚔️ 왜 업로드가 안될까2222 - {file_url}")

        # os.remove(file_name)
        return file_url

    except FileNotFoundError:
        log_error(f"The file '{file_name}' was not found")
        return file_name
    except NoCredentialsError:
        log_error("Credentials not available. Please check your AWS credentials.")
        return file_name