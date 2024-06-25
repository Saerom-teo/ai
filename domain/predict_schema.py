from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class PredictionRequest(BaseModel):
    modelName: Optional[str] = Field(None, example="yolov8n_0531_e30_b16.onnx")
    images: List[str] = Field(..., example=[
        "image1.jpg",
        "image2.jpg"
    ])

class PredictionResponse(BaseModel):
    result: Literal['clear', 'deny'] = Field(example="clear")
    images: List[str]