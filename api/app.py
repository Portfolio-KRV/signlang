"""FastAPI application for Sign Language Recognition."""

import base64
import io
import logging

import numpy as np
from fastapi import File, HTTPException, Request, UploadFile
from PIL import Image

from api.schemas import (
    ClassifyFromArrayRequest,
    ClassifyResponse,
    InfoResponse,
    PredictionResult,
)
from src import __version__
from src.api_common import create_app, limiter, register_error_handlers
from src.config import API_PORT, IMAGE_SIZE, LETTERS, NUM_CLASSES
from src.exceptions import (
    ConfigurationError,
    DataError,
    DataValidationError,
    ModelError,
    ModelLoadError,
    ModelNotLoadedError,
)
from src.model import SignLanguageClassifier
from src.validators import validate_array_shape, validate_no_nan_inf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security: Maximum array size to prevent DoS
MAX_ARRAY_SIZE = IMAGE_SIZE * IMAGE_SIZE * 10  # Allow up to 10x the expected size

app = create_app(
    title="Sign Language Recognition API",
    description="Classify American Sign Language letters from images",
    version=__version__,
)
register_error_handlers(
    app,
    {
        ModelNotLoadedError: (503, "model_not_loaded"),
        ModelLoadError: (500, "model_load_error"),
        ModelError: (500, "model_error"),
        DataValidationError: (422, "validation_error"),
        DataError: (500, "data_error"),
        ConfigurationError: (500, "configuration_error"),
    },
    expose_message=(DataValidationError,),
)

# Global classifier instance
_classifier: SignLanguageClassifier | None = None


def get_classifier() -> SignLanguageClassifier:
    """Get or create classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = SignLanguageClassifier()
        if not _classifier.load():
            raise ModelNotLoadedError("Model not loaded.")
    return _classifier


def image_to_array(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array for model input."""
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.float32)


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    """Root endpoint."""
    return {"message": "Sign Language Recognition API", "status": "running"}


@app.get("/info", response_model=InfoResponse)
@limiter.limit("60/minute")
async def get_info(request: Request):
    """Get API information."""
    model_loaded = _classifier is not None and _classifier.is_loaded
    return InfoResponse(
        name="Sign Language Recognition",
        version=__version__,
        description="Classify American Sign Language letters (A-Y, excluding J and Z) using CNN",
        num_classes=NUM_CLASSES,
        letters=LETTERS,
        image_size=IMAGE_SIZE,
        model_loaded=model_loaded,
    )


@app.post("/classify", response_model=ClassifyResponse)
@limiter.limit("10/minute")
async def classify_image(request: Request, file: UploadFile = File(...)):
    """Classify sign language letter from uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    max_file_size = 10 * 1024 * 1024  # 10 MB
    contents = await file.read(max_file_size + 1)
    if len(contents) > max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_file_size // (1024 * 1024)}MB",
        )

    classifier = get_classifier()
    image = Image.open(io.BytesIO(contents))
    arr = image_to_array(image)
    result = classifier.predict(arr)

    return ClassifyResponse(
        letter=result["letter"],
        label=result["label"],
        confidence=result["confidence"],
        top_predictions=[
            PredictionResult(letter=p["letter"], probability=p["probability"])
            for p in result["top_predictions"]
        ],
    )


@app.post("/classify/base64", response_model=ClassifyResponse)
@limiter.limit("20/minute")
async def classify_base64(request: Request, image_data: dict):
    """Classify sign language letter from base64 encoded image."""
    classifier = get_classifier()
    base64_str = image_data.get("image", "")
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    try:
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
    except (ValueError, OSError) as e:
        raise DataValidationError(f"Invalid base64 image: {e}") from e

    arr = image_to_array(image)
    result = classifier.predict(arr)

    return ClassifyResponse(
        letter=result["letter"],
        label=result["label"],
        confidence=result["confidence"],
        top_predictions=[
            PredictionResult(letter=p["letter"], probability=p["probability"])
            for p in result["top_predictions"]
        ],
    )


@app.post("/classify/array", response_model=ClassifyResponse)
@limiter.limit("20/minute")
async def classify_array(request: Request, body: ClassifyFromArrayRequest):
    """Classify sign language letter from pixel array."""
    classifier = get_classifier()

    try:
        arr = validate_array_shape(body.pixels, max_length=MAX_ARRAY_SIZE, field_name="pixels")
        arr = validate_no_nan_inf(arr, "pixels")
    except ValueError as e:
        raise DataValidationError(str(e)) from e

    arr = arr.astype(np.float32)
    result = classifier.predict(arr)

    return ClassifyResponse(
        letter=result["letter"],
        label=result["label"],
        confidence=result["confidence"],
        top_predictions=[
            PredictionResult(letter=p["letter"], probability=p["probability"])
            for p in result["top_predictions"]
        ],
    )


@app.get("/sample/{letter}")
@limiter.limit("60/minute")
async def get_sample(request: Request, letter: str):
    """Get a sample image for a specific letter."""
    letter = letter.upper()
    if letter not in LETTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid letter. Must be one of: {', '.join(LETTERS)}",
        )

    import pandas as pd

    from src.config import LETTER_TO_CSV_LABEL, TEST_CSV

    df = pd.read_csv(TEST_CSV)
    csv_label = LETTER_TO_CSV_LABEL[letter]
    samples = df[df["label"] == csv_label]
    if len(samples) == 0:
        raise HTTPException(status_code=404, detail="No sample found")

    sample = samples.sample(1).values[0]
    pixels = sample[1:].tolist()

    arr = np.array(pixels, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)
    img = Image.fromarray(arr, mode="L").resize((112, 112), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"letter": letter, "label": csv_label, "pixels": pixels, "image": image_b64}


@app.on_event("startup")
async def preload_model():
    """Pre-load the classifier model on startup."""
    global _classifier
    logger.info("Loading sign language classifier...")
    _classifier = SignLanguageClassifier()
    if _classifier.load():
        logger.info("Sign language classifier loaded successfully")
    else:
        logger.warning("Model not found. Train it first using train.py")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
