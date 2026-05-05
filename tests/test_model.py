"""Tests for model module."""

import numpy as np
import pytest

from src.config import IMAGE_SIZE, LETTERS, NUM_CLASSES
from src.exceptions import ModelNotLoadedError
from src.model import SignLanguageClassifier, build_model


class TestBuildModel:
    """Tests for build_model function."""

    def test_builds_model(self):
        """Test that model is built."""
        model = build_model()
        assert model is not None

    def test_correct_input_shape(self):
        """Test correct input shape."""
        model = build_model()
        assert model.input_shape == (None, IMAGE_SIZE, IMAGE_SIZE, 1)

    def test_correct_output_shape(self):
        """Test correct output shape."""
        model = build_model()
        assert model.output_shape == (None, NUM_CLASSES)

    def test_custom_filters(self):
        """Test model with custom filter count."""
        model = build_model(num_filters=64)
        assert model is not None

    def test_custom_blocks(self):
        """Test model with custom block count."""
        model = build_model(num_blocks=2)
        assert model is not None


class TestSignLanguageClassifier:
    """Tests for SignLanguageClassifier class."""

    def test_init(self):
        """Test initialization."""
        classifier = SignLanguageClassifier()
        assert classifier.model is None
        assert classifier.is_loaded is False

    def test_build(self):
        """Test build method."""
        classifier = SignLanguageClassifier()
        classifier.build()
        assert classifier.model is not None

    def test_preprocess_flat_array(self):
        """Test preprocessing flat array."""
        classifier = SignLanguageClassifier()
        arr = np.random.randint(0, 256, (784,), dtype=np.uint8)
        processed = classifier.preprocess_image(arr)
        assert processed.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 1)
        assert processed.min() >= -1
        assert processed.max() <= 1

    def test_preprocess_2d_array(self):
        """Test preprocessing 2D array."""
        classifier = SignLanguageClassifier()
        arr = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        processed = classifier.preprocess_image(arr)
        assert processed.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 1)

    def test_predict_without_load_raises(self):
        """Test that predict without load raises error."""
        classifier = SignLanguageClassifier()
        arr = np.random.rand(784)
        with pytest.raises(ModelNotLoadedError):
            classifier.predict(arr)


class TestConfig:
    """Tests for configuration."""

    def test_letters_count(self):
        """Test that we have 24 letters."""
        assert len(LETTERS) == NUM_CLASSES

    def test_letters_exclude_j_z(self):
        """Test that J and Z are excluded."""
        assert "J" not in LETTERS
        assert "Z" not in LETTERS
