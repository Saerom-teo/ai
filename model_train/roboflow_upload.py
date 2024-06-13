import roboflow

YOUR_API_KEY_HERE = 'k3UFrgbNGcEqJ7ZNBGb9'
rf = roboflow.Roboflow(api_key=YOUR_API_KEY_HERE)

# get a project
project = rf.workspace().project("saerom")

# Upload image to dataset
project.upload_dataset(
    dataset_path="./dataset/",
    num_workers=10,
    dataset_format="yolov8",
    project_license="MIT",
    project_type="object-detection"
)

