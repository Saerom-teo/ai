from dotenv import load_dotenv
import boto3
import os
from botocore.exceptions import NoCredentialsError

load_dotenv()

ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
SECRET_KEY = os.getenv('S3_SECRET_KEY')
REGION_NAME = os.getenv('S3_REGION_NAME')
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')


def upload_to_s3(file_name, object_name=None):
    if object_name is None:
        object_name = file_name

    try:
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

        object_name = f'AI/{object_name}'
        response = s3.upload_file(
            file_name, 
            BUCKET_NAME, 
            object_name, 
            # ExtraArgs={'ACL':'public-read'}
        )

        file_url = f"https://{BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/{object_name}"

        print(response)
        print(file_url)

    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

    return True

if __name__ == "__main__":
    uploaded = upload_to_s3('resources/test_images/test01.jpg')
    if uploaded:
        print("File was uploaded successfully")
    else:
        print("File upload failed")
