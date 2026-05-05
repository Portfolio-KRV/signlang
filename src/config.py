"""Configuration for Sign Language Recognition."""

from pathlib import Path
from typing import Any

# Paths
PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Data files
TRAIN_CSV = DATA_DIR / "sign_mnist_train.csv"
TEST_CSV = DATA_DIR / "sign_mnist_test.csv"

# Model parameters
IMAGE_SIZE: int = 28
NUM_CLASSES = 24
NUM_FILTERS = 128
NUM_BLOCKS = 3

# Letters mapping (J and Z are missing - they require movement)
LETTERS: list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
           "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
           "V", "W", "X", "Y"]

# Two coexisting label spaces — keep them straight:
#
# 1. Model output indexes (0-23, compacted): the model has 24 outputs because
#    train.load_data() one-hot encodes with pd.get_dummies, which drops the
#    missing CSV value 9 and yields columns 0-23. Use LABEL_TO_LETTER /
#    LETTER_TO_LABEL on anything coming out of (or going into) model.predict.
LABEL_TO_LETTER: dict[int, str] = {i: letter for i, letter in enumerate(LETTERS)}
LETTER_TO_LABEL: dict[str, int] = {letter: i for i, letter in enumerate(LETTERS)}

# 2. Raw CSV labels (0-24, skipping 9 = J): the Sign Language MNIST CSV keeps
#    the original alphabet indices, so K=10, L=11, ..., Y=24. Use
#    CSV_LABEL_TO_LETTER / LETTER_TO_CSV_LABEL when reading the CSV directly.
CSV_LABEL_TO_LETTER: dict[int, str] = dict(
    zip(list(range(0, 9)) + list(range(10, 25)), LETTERS)
)
LETTER_TO_CSV_LABEL: dict[str, int] = {l: c for c, l in CSV_LABEL_TO_LETTER.items()}

# Training parameters
BATCH_SIZE: int = 32
EPOCHS: int = 50
PATIENCE = 10

# API
API_PORT = 8015
