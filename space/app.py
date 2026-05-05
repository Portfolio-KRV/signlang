"""Gradio Space: Sign Language (ASL alphabet) recognition.

A 3-block CNN trained on the Sign Language MNIST dataset (28x28 grayscale)
classifies hand signs for letters A-Y (J and Z are excluded - they require
movement). Accepts webcam capture, uploaded images, or one of the 24
per-letter samples bundled below.

Source code:
    https://github.com/Portfolio-KRV/signlang
"""

from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from tensorflow import keras

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "signlang_model.keras"
SAMPLES_DIR = HERE / "sample_images"
CHART_PATH = HERE / "alphabet_chart.png"

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
           "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
           "V", "W", "X", "Y"]

_model = None


def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(str(MODEL_PATH))
    return _model


def preprocess(image: np.ndarray) -> np.ndarray:
    """Convert to 28x28 grayscale, normalize to [-1, 1]."""
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image.astype(np.float32) / 127.5 - 1.0
    return image.reshape(1, 28, 28, 1)


EMPTY_RESULT = (
    "Show your hand or pick a sample below — I'll predict instantly. "
    "_(J and Z aren't supported — they require movement.)_"
)


def predict(image):
    if image is None:
        return None, EMPTY_RESULT

    if isinstance(image, Image.Image):
        image = np.array(image)

    preview = cv2.resize(image, (168, 168))
    if preview.ndim == 3:
        preview_gray = cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY)
    else:
        preview_gray = preview

    model = get_model()
    x = preprocess(image)
    probs = model.predict(x, verbose=0)[0]

    top_idx = np.argsort(probs)[::-1][:5]
    best_letter = LETTERS[int(top_idx[0])]
    best_prob = float(probs[top_idx[0]])

    lines = [
        f"### Prediction: **{best_letter}**  ·  {best_prob*100:.1f}% confident",
        "",
        "**Top-5 candidates**",
    ]
    for rank, idx in enumerate(top_idx, start=1):
        letter = LETTERS[int(idx)]
        prob = float(probs[idx])
        bar = "▓" * int(prob * 30)
        lines.append(f"{rank}. **{letter}** — {prob*100:.1f}% `{bar}`")

    lines.append("")
    lines.append(
        "_Heads-up: the model was trained on Sign Language MNIST — uniform "
        "gray background, hand centered. Real-world photos with clutter or "
        "soft lighting will trip it up._"
    )
    return preview_gray, "\n".join(lines)


INTRO_MD = """
# Sign Language Recognition (ASL alphabet)

A small CNN that recognizes hand signs for 24 letters of the American Sign
Language alphabet (A-Y, excluding J and Z — those require motion). Runs in
~50 ms on CPU.

**How to try it**
1. Pick a sample below, upload a photo, or use your webcam.
2. The model predicts as soon as the image loads — no button to press.

→ [Source on GitHub](https://github.com/Portfolio-KRV/signlang)
"""

# Per-letter samples come from one row per letter in the test set.
sample_paths = sorted(SAMPLES_DIR.glob("*.png"), key=lambda p: p.stem)

with gr.Blocks(title="Sign Language Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO_MD)

    with gr.Accordion("ASL alphabet — reference (click to collapse)", open=True):
        gr.Image(
            value=str(CHART_PATH) if CHART_PATH.exists() else None,
            show_label=False,
            container=False,
            interactive=False,
        )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Your hand sign — webcam or upload",
                type="numpy",
                sources=["webcam", "upload"],
                height=320,
            )
            if sample_paths:
                gr.Examples(
                    examples=[[str(p)] for p in sample_paths],
                    inputs=input_image,
                    label="Or click a sample (one per letter, from the test set)",
                    examples_per_page=24,
                )
        with gr.Column(scale=1):
            preview_out = gr.Image(
                label="What the model sees (28x28 grayscale)",
                height=224,
            )
            results_md = gr.Markdown(EMPTY_RESULT)

    input_image.change(predict, inputs=input_image, outputs=[preview_out, results_md])


if __name__ == "__main__":
    demo.launch()
