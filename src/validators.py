"""Input validation utilities."""

import math

import numpy as np


def validate_numeric_range(
    value: float,
    min_value: float | None = None,
    max_value: float | None = None,
    field_name: str = "value"
) -> float:
    """Validate that a numeric value is within specified range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        field_name: Name of the field (for error messages)

    Returns:
        The validated value

    Raises:
        ValueError: If value is out of range, NaN, or Inf
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{field_name} must be a finite number")

    if min_value is not None and value < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")

    if max_value is not None and value > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}")

    return value


def validate_array_shape(
    arr: np.ndarray | list,
    expected_shape: tuple | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    field_name: str = "array"
) -> np.ndarray:
    """Validate array shape and size.

    Args:
        arr: Array to validate
        expected_shape: Expected shape tuple
        min_length: Minimum array length
        max_length: Maximum array length
        field_name: Name of the field (for error messages)

    Returns:
        Validated numpy array

    Raises:
        ValueError: If array shape or size is invalid
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if arr.size == 0:
        raise ValueError(f"{field_name} cannot be empty")

    if expected_shape and arr.shape != expected_shape:
        raise ValueError(
            f"{field_name} has shape {arr.shape}, expected {expected_shape}"
        )

    if min_length and arr.size < min_length:
        raise ValueError(f"{field_name} must have at least {min_length} elements")

    if max_length and arr.size > max_length:
        raise ValueError(f"{field_name} must have at most {max_length} elements")

    return arr


def validate_no_nan_inf(arr: np.ndarray | list, field_name: str = "array") -> np.ndarray:
    """Validate that array contains no NaN or Inf values.

    Args:
        arr: Array to validate
        field_name: Name of the field (for error messages)

    Returns:
        Validated numpy array

    Raises:
        ValueError: If array contains NaN or Inf
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if np.any(np.isnan(arr)):
        raise ValueError(f"{field_name} contains NaN values")

    if np.any(np.isinf(arr)):
        raise ValueError(f"{field_name} contains Inf values")

    return arr


def sanitize_text_input(text: str, max_length: int = 10000) -> str:
    """Sanitize text input.

    Args:
        text: Text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text

    Raises:
        ValueError: If text is invalid
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    if len(text) > max_length:
        raise ValueError(f"Text exceeds maximum length of {max_length} characters")

    # Strip potentially dangerous characters
    text = text.strip()

    return text
