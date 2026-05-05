"""Pydantic schemas for Sign Language API."""

import math

from pydantic import BaseModel, Field, field_validator


class PredictionResult(BaseModel):
    """Single prediction result."""
    letter: str
    probability: float


class ClassifyResponse(BaseModel):
    """Response for classification."""
    letter: str
    label: int
    confidence: float
    top_predictions: list[PredictionResult]


class ClassifyFromArrayRequest(BaseModel):
    """Request to classify from pixel array."""
    pixels: list[float] = Field(..., min_length=784, max_length=784)

    @field_validator('pixels')
    @classmethod
    def validate_pixels(cls, v):
        """Validate pixel array."""
        if not v:
            raise ValueError("Pixels array cannot be empty")

        if len(v) != 784:
            raise ValueError(f"Pixels array must have exactly 784 elements, got {len(v)}")

        # Check for NaN/Inf
        for i, val in enumerate(v):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Pixel at index {i} is NaN or Inf")

        return v


class InfoResponse(BaseModel):
    """API info response."""
    name: str
    version: str
    description: str
    num_classes: int
    letters: list[str]
    image_size: int
    model_loaded: bool
