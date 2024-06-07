from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    modelName: str = Field(..., example="yolov8n_0531_e30.pt")
    images: List[str] = Field(..., example=[
        "image1.jpg",
        "image2.jpg"
    ])