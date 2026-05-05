"""CNN model for Sign Language Recognition."""

import os

import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model

from .config import IMAGE_SIZE, LABEL_TO_LETTER, MODELS_DIR, NUM_BLOCKS, NUM_CLASSES, NUM_FILTERS
from .exceptions import ModelLoadError, ModelNotLoadedError


def build_model(
    num_filters: int = NUM_FILTERS,
    num_blocks: int = NUM_BLOCKS,
    input_shape: tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 1)
) -> Sequential:
    """Build the CNN model for sign language classification.

    Architecture: 3 blocks of (Conv2D x Conv2D x MaxPool) + Dense layers

    Args:
        num_filters: Number of filters per convolutional layer
        num_blocks: Number of convolutional blocks
        input_shape: Input image shape

    Returns:
        Compiled Keras model
    """
    model = Sequential(name="signlang_cnn")

    for i in range(num_blocks):
        # First conv in block
        if i == 0:
            model.add(Conv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                input_shape=input_shape
            ))
        else:
            model.add(Conv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ))

        # Second conv in block
        model.add(Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ))

        # MaxPooling
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(units=110, activation="tanh"))
    model.add(Dense(units=NUM_CLASSES, activation="softmax"))

    return model


class SignLanguageClassifier:
    """Classifier for sign language letters."""

    def __init__(self):
        """Initialize classifier."""
        self.model: Sequential | None = None
        self.is_loaded = False

    def build(self) -> None:
        """Build the model."""
        self.model = build_model()
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def load(self, weights_path: str | None = None) -> None:
        """Load model weights.

        Args:
            weights_path: Path to weights file

        Raises:
            ModelLoadError: If model file not found or failed to load
        """
        if weights_path is None:
            weights_path = MODELS_DIR / "signlang_model.keras"

        if not os.path.exists(weights_path):
            raise ModelLoadError(f"Model file not found at {weights_path}")

        try:
            self.model = load_model(weights_path)
            self.is_loaded = True
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {weights_path}: {e}") from e

    def save(self, weights_path: str | None = None) -> None:
        """Save model weights."""
        if weights_path is None:
            weights_path = MODELS_DIR / "signlang_model.keras"

        if self.model is not None:
            self.model.save(weights_path)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for prediction.

        Args:
            image: Input image (28x28 or 784 flat)

        Returns:
            Preprocessed image ready for model
        """
        # Handle flat array
        if image.ndim == 1:
            image = image.reshape(IMAGE_SIZE, IMAGE_SIZE)

        # Ensure 2D
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

        # Normalize to [-1, 1]
        if image.max() > 1:
            image = image / 127.5 - 1

        return image.astype(np.float32)

    def predict(self, image: np.ndarray) -> dict:
        """Predict the sign language letter.

        Args:
            image: Input image (28x28 grayscale)

        Returns:
            Dictionary with prediction results

        Raises:
            ModelNotLoadedError: If model not loaded
        """
        if not self.is_loaded and self.model is None:
            raise ModelNotLoadedError("Model must be loaded before prediction. Call load() first.")

        # Preprocess
        processed = self.preprocess_image(image)

        # Predict
        probs = self.model.predict(processed, verbose=0)[0]
        predicted_label = int(np.argmax(probs))
        predicted_letter = LABEL_TO_LETTER[predicted_label]
        confidence = float(probs[predicted_label])

        # Get top 5 predictions
        top_indices = np.argsort(probs)[::-1][:5]
        top_predictions = [
            {
                "letter": LABEL_TO_LETTER[i],
                "probability": float(probs[i])
            }
            for i in top_indices
        ]

        return {
            "letter": predicted_letter,
            "label": predicted_label,
            "confidence": confidence,
            "top_predictions": top_predictions
        }

    def predict_batch(self, images: np.ndarray) -> list:
        """Predict multiple images.

        Args:
            images: Batch of images

        Returns:
            List of predictions

        Raises:
            ModelNotLoadedError: If model not loaded
        """
        if not self.is_loaded and self.model is None:
            raise ModelNotLoadedError("Model must be loaded before prediction. Call load() first.")

        # Ensure correct shape
        if images.ndim == 2:
            # Single image flattened
            images = images.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
        elif images.ndim == 3:
            # Multiple images or single 28x28
            if images.shape[0] == IMAGE_SIZE:
                images = images.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
            else:
                images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

        # Normalize
        if images.max() > 1:
            images = images / 127.5 - 1

        # Predict
        all_probs = self.model.predict(images, verbose=0)

        results = []
        for probs in all_probs:
            predicted_label = int(np.argmax(probs))
            results.append({
                "letter": LABEL_TO_LETTER[predicted_label],
                "label": predicted_label,
                "confidence": float(probs[predicted_label])
            })

        return results


def get_classifier() -> SignLanguageClassifier:
    """Get a loaded classifier instance.

    Returns:
        Loaded SignLanguageClassifier

    Raises:
        ModelLoadError: If model weights not found
    """
    classifier = SignLanguageClassifier()
    classifier.load()  # This will raise ModelLoadError if not found
    return classifier
