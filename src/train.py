"""Training script for Sign Language model."""

import logging

import pandas as pd
from keras.callbacks import EarlyStopping, History
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .logging_config import setup_logging

logger = logging.getLogger(__name__)

from .config import BATCH_SIZE, EPOCHS, IMAGE_SIZE, MODELS_DIR, PATIENCE, TEST_CSV, TRAIN_CSV

logger = logging.getLogger(__name__)

from .model import SignLanguageClassifier


def load_data():
    """Load and preprocess the sign language MNIST data."""
    # Load training data
    df_train = pd.read_csv(TRAIN_CSV)
    X_train = df_train.values[:, 1:].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y_train = df_train.values[:, 0]

    # Load test data
    df_test = pd.read_csv(TEST_CSV)
    X_test = df_test.values[:, 1:].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y_test = df_test.values[:, 0]

    # Split training into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    # Normalize to [-1, 1]
    X_train = X_train / 127.5 - 1
    X_val = X_val / 127.5 - 1
    X_test = X_test / 127.5 - 1

    # One-hot encode labels
    y_train_oh = pd.get_dummies(y_train).values
    y_val_oh = pd.get_dummies(y_val).values
    y_test_oh = pd.get_dummies(y_test).values

    return (X_train, y_train_oh), (X_val, y_val_oh), (X_test, y_test_oh)


def create_data_augmentation():
    """Create data augmentation generator."""
    return ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,  # Don't flip - would confuse letters
        vertical_flip=False,
    )


def train_model(use_augmentation: bool = True, epochs: int = EPOCHS):
    """Train the sign language classifier.

    Args:
        use_augmentation: Whether to use data augmentation
        epochs: Number of training epochs

    Returns:
        Trained classifier and history
    """
    logger.info("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()

    logger.info("Training samples: %d", len(X_train))
    logger.info("Validation samples: %d", len(X_val))
    logger.info("Test samples: %d", len(X_test))

    # Build model
    logger.info("Building model...")
    classifier = SignLanguageClassifier()
    classifier.build()
    classifier.model.summary()

    # Callbacks
    callbacks = [
        History(),
        EarlyStopping(
            patience=PATIENCE,
            monitor="val_loss",
            restore_best_weights=True
        )
    ]

    # Train
    logger.info("Training...")
    if use_augmentation:
        datagen = create_data_augmentation()
        datagen.fit(X_train)

        history = classifier.model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = classifier.model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_acc = classifier.model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test accuracy: %.4f", test_acc)
    logger.info("Test loss: %.4f", test_loss)

    # Save model
    logger.info("Saving model...")
    classifier.is_loaded = True
    classifier.save()
    logger.info("Model saved to %s", MODELS_DIR / 'signlang_model.keras')

    return classifier, history


if __name__ == "__main__":
    setup_logging()
    train_model(use_augmentation=True, epochs=50)
