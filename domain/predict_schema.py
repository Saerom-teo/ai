from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from lib.const import DEFAULT_MODEL

class PredictionRequest(BaseModel):
    modelName: Optional[str] = Field(None, example=DEFAULT_MODEL)
    images: List[str] = Field(..., example=[
        "image1.jpg",
        "image2.jpg"
    ])

class PredictionResponse(BaseModel):
    result: Literal['clear', 'deny'] = Field(example="clear")
    images: List[str]