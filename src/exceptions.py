"""Custom exceptions for the project."""


class ProjectError(Exception):
    """Base exception for all project errors."""
    pass


class ModelError(ProjectError):
    """Base exception for model-related errors."""
    pass


class ModelNotLoadedError(ModelError):
    """Model not loaded or failed to load."""
    pass


class ModelLoadError(ModelError):
    """Error loading model from disk."""
    pass


class PredictionError(ModelError):
    """Error during model prediction."""
    pass


class DataError(ProjectError):
    """Base exception for data-related errors."""
    pass


class DataLoadError(DataError):
    """Error loading data from disk or external source."""
    pass


class DataValidationError(DataError):
    """Invalid or malformed data."""
    pass


class ConfigurationError(ProjectError):
    """Configuration error (missing files, invalid settings, etc.)."""
    pass
