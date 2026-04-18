from typing import Dict, List, Optional

from pydantic import BaseModel


class PredictResponse(BaseModel):
    model_name: str
    input_mode: str
    uploaded_input_url: str
    pre_image_url: str
    generated_post_url: str
    reference_post_url: Optional[str] = None
    metrics: dict
    description: Optional[Dict] = None
    landmarks: Optional[Dict] = None
    disclaimer: str
