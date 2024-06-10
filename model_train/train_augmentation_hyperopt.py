import albumentations as A
import cv2
import os
import glob

def augment_images(input_dir, output_dir, augmentations, num_augmentations=1):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))  # Assuming images are in .jpg format

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_name = os.path.basename(image_path).split('.')[0]

        for i in range(num_augmentations):
            augmented = augmentations(image=image)
            augmented_image = augmented['image']
            output_path = os.path.join(output_dir, f"{image_name}_aug_{i}.jpg")
            cv2.imwrite(output_path, augmented_image)
            
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from ultralytics import YOLO
from datetime import datetime
import os

# Define the search space for hyperparameters
search_space = {
    'hsv_h': hp.uniform('hsv_h', 0.0, 0.1),
    'hsv_s': hp.uniform('hsv_s', 0.0, 0.7),
    'hsv_v': hp.uniform('hsv_v', 0.0, 0.4),
    'degrees': hp.uniform('degrees', 0.0, 10.0),
    'translate': hp.uniform('translate', 0.0, 0.1),
    'scale': hp.uniform('scale', 0.5, 1.5),
    'shear': hp.uniform('shear', 0.0, 2.0),
    'perspective': hp.uniform('perspective', 0.0, 0.001),
    'flipud': hp.uniform('flipud', 0.0, 1.0),
    'fliplr': hp.uniform('fliplr', 0.0, 1.0)
}

# Define the objective function for hyperparameter optimization
def objective(params):
    augmentations = A.Compose([
        A.HueSaturationValue(hue_shift_limit=params['hsv_h'], sat_shift_limit=params['hsv_s'], val_shift_limit=params['hsv_v']),
        A.ShiftScaleRotate(shift_limit=params['translate'], scale_limit=params['scale']-1, rotate_limit=params['degrees'], shear_limit=params['shear']),
        A.Perspective(scale=params['perspective']),
        A.VerticalFlip(p=params['flipud']),
        A.HorizontalFlip(p=params['fliplr']),
    ])

    input_dir = "./datasets/recyclables_2/images"
    output_dir = "./datasets/recyclables_2/augmented_images"
    augment_images(input_dir, output_dir, augmentations)

    now = datetime.now()
    date_str = now.strftime("%m%d")

    base_model = "yolov8n.pt"
    model = YOLO(base_model)
    epochs = 10  # Number of epochs for each trial

    # Model training
    results = model.train(
        data="./datasets/recyclables_2/recyclables_2.yaml",
        epochs=epochs,
        imgsz=640,
        save_period=1,
        device=1,
        name="hyperopt_trial"
    )

    # Assume that the metric to minimize is 'val_loss'
    val_loss = results[-1]['metrics']['val_loss']  # Adjust based on actual result structure

    return {'loss': val_loss, 'status': STATUS_OK}

# Run hyperparameter optimization
trials = Trials()
best_params = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=10,  # Number of trials
    trials=trials
)

print("Best parameters found: ", best_params)

def train_model(best_params):
    augmentations = A.Compose([
        A.HueSaturationValue(hue_shift_limit=best_params['hsv_h'], sat_shift_limit=best_params['hsv_s'], val_shift_limit=best_params['hsv_v']),
        A.ShiftScaleRotate(shift_limit=best_params['translate'], scale_limit=best_params['scale']-1, rotate_limit=best_params['degrees'], shear_limit=best_params['shear']),
        A.Perspective(scale=best_params['perspective']),
        A.VerticalFlip(p=best_params['flipud']),
        A.HorizontalFlip(p=best_params['fliplr']),
    ])

    input_dir = "./datasets/recyclables_2/images"
    output_dir = "./datasets/recyclables_2/augmented_images"
    augment_images(input_dir, output_dir, augmentations, num_augmentations=5)
    
    now = datetime.now()
    date_str = now.strftime("%m%d")

    base_model = "yolov8n.pt"
    model = YOLO(base_model)

    epochs = 100

    # Model training with the best parameters
    results = model.train(
        data="./datasets/recyclables_2/recyclables_2.yaml",
        epochs=epochs,
        imgsz=640,
        save_period=5,
        device=1,
        name=f"{os.path.splitext(base_model)[0]}_{date_str}_e{epochs}"  # Save folder name
    )
    return results

if __name__ == '__main__':
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials
    )
    print("Best parameters found: ", best_params)
    train_model(best_params)
