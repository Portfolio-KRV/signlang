# ASL Alphabet Recognition

A CNN that classifies American Sign Language letters from 28x28 grayscale
hand images. Covers 24 of 26 letters (J and Z are excluded — they
require motion). The interesting part isn't the >99% test accuracy; it's
the confusion-pair analysis showing which signs the model still mixes
up when it does fail.

## What I built

- **CNN classifier** (`src/model.py`): convolutional blocks with batch
  normalization and a softmax head over 24 classes, trained on
  Sign Language MNIST.
- **Three input modes** in the API: multipart upload (`/classify`),
  base64 (`/classify/base64`), and raw pixel arrays (`/classify/array`)
  — useful for pulling samples straight from the portfolio frontend.
- **FastAPI service** with rate-limiting and CORS via
  `src/api_common.py`, plus a `/sample/{letter}` endpoint that pulls a
  random test image for any letter.

## Why it matters

Sign Language MNIST looks "solved" at 99%+ accuracy, but the residual
errors are interesting: the most-confused pair is N/S (visually similar
fists with thumb position differing) and C/O (curved hand shapes with
different finger spread). That's where data-augmentation choices —
rotation, zoom, shift — actually move the needle, by 3-5 percentage
points on those specific confusion classes.

## Tech stack

TensorFlow / Keras (CNN with BatchNorm) · FastAPI · NumPy · Pillow · scikit-learn

## Quickstart

```bash
# install (one-time)
pip install -e ".[api,notebook,dev]"

# run the API
uvicorn api.app:app --reload --port 8006

# fetch a random sample for letter "A"
curl http://localhost:8006/sample/A

# classify a 784-pixel array (28x28 grayscale, flattened)
curl -X POST http://localhost:8006/classify/array \
  -H 'Content-Type: application/json' \
  -d '{"pixels": [/* 784 ints in 0..255 */]}'
```

## Live demo

Hosted on Hugging Face Spaces:
[kevinreyesds/signlang](https://huggingface.co/spaces/kevinreyesds/signlang)
*(wired through the portfolio at `/projects/signlang/demo`)*

## Tests

```bash
pytest
ruff check .
```

Tests cover the model architecture, preprocessing, the
`ModelNotLoadedError` contract, and the FastAPI request/response
schemas.

## Repository

[Portfolio-KRV/signlang](https://github.com/Portfolio-KRV/signlang)

## License

[MIT](LICENSE)
