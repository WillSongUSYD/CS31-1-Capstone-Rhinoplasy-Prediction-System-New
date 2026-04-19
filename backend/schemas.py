from typing import Dict, List, Optional

from pydantic import BaseModel


class PredictResponse(BaseModel):
    model_name: str
    input_mode: str
    # URL fields come from _artifact_url(), which returns Optional[str].
    # Today every call path puts artifacts under PREDICTIONS_DIR so the URLs
    # are always resolvable, but making these fields Optional is defense in
    # depth against a future regression that would otherwise surface as a
    # 500 at Pydantic-validation time instead of a graceful null in the
    # response body.
    uploaded_input_url: Optional[str] = None
    pre_image_url: Optional[str] = None
    generated_post_url: Optional[str] = None
    reference_post_url: Optional[str] = None
    metrics: dict
    description: Optional[Dict] = None
    landmarks: Optional[Dict] = None
    disclaimer: str
