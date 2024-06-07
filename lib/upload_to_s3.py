from dotenv import load_dotenv
import boto3
import os
from botocore.exceptions import NoCredentialsError

load_dotenv()

ACCESS_KEY = os.getenv('MODEL_ID')
SECRET_KEY = os.getenv('MODEL_ID')
BUCKET_NAME = 'arzip-bucket'
REGION_NAME = 'ap-northeast-2'

def upload_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name

    try:
        # s3_client = boto3.client('s3')
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

        # response = s3_client.upload_file(file_name, bucket, object_name)
        object_name = f'AI/{object_name}'
        response = s3.upload_file(
            file_name, 
            bucket, 
            object_name, 
            ExtraArgs={'ACL':'public-read'}
        )

        print(response)

    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

    return True

def save_file_in_S3(fbx_file_path):
    try:
        with open(fbx_file_path, "rb") as fbx:
            fbx_data = fbx.read()

        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

        file_name = fbx_file_path.split('/')[-1]
        object_name = f'AI/{file_name}'

        try:
            s3.put_object(Bucket=BUCKET_NAME, Key=object_name, Body=fbx_data)
            file_url = f"https://{BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/{object_name}"
            print('Save fbx file in AWS S3 done')
        except NoCredentialsError:
            raise HTTPException(status_code=401, detail="Credential problem")
        except Exception as e:
            raise HTTPException(status_code=500, detail="Something went wrong!!")

        return file_url
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

# 예제 사용
uploaded = upload_to_s3('local_file.txt', 'your-bucket-name', 's3_file.txt')
if uploaded:
    print("File was uploaded successfully")
else:
    print("File upload failed")
