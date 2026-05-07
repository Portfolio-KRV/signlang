"""Gradio Space: Sign Language (ASL alphabet) recognition with hand detection.

Pipeline:
    Image -> MediaPipe Hands (detect + crop) -> CNN (28x28 grayscale)

The CNN is the same 3-block model trained on Sign Language MNIST (which
expects uniform-background, centered-hand inputs). MediaPipe Hands runs
first to detect and crop the hand region from real-world photos, getting
the input close to the dataset domain. If MediaPipe can't find a hand
(e.g. on a pre-cropped dataset sample), the image goes straight to the
CNN.

Source code:
    https://github.com/Portfolio-KRV/signlang
"""

from pathlib import Path

import cv2
import gradio as gr
import mediapipe as mp
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
_hands = None


def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(str(MODEL_PATH))
    return _model


def get_hands_detector():
    """Lazily build the MediaPipe Hands detector (static-image mode)."""
    global _hands
    if _hands is None:
        _hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,
        )
    return _hands


def extract_hand_region(image: np.ndarray, padding: float = 0.2):
    """Detect a hand with MediaPipe and return a square crop around it.

    Returns:
        (cropped_rgb, detected_bool). If no hand is detected, returns
        (None, False) and the caller can fall back to the original image.
    """
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        rgb = image

    h, w = rgb.shape[:2]
    results = get_hands_detector().process(rgb)

    if not results.multi_hand_landmarks:
        return None, False

    landmarks = results.multi_hand_landmarks[0]
    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_w = max(1.0, x_max - x_min)
    box_h = max(1.0, y_max - y_min)
    pad_x = box_w * padding
    pad_y = box_h * padding

    # Square crop centered on the hand, so the CNN sees a 1:1 image like
    # its training set.
    side = max(box_w + 2 * pad_x, box_h + 2 * pad_y)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    sx_min = max(0, int(cx - side / 2))
    sy_min = max(0, int(cy - side / 2))
    sx_max = min(w, int(cx + side / 2))
    sy_max = min(h, int(cy + side / 2))

    cropped = rgb[sy_min:sy_max, sx_min:sx_max]
    if cropped.size == 0:
        return None, False
    return cropped, True


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
    "Show your hand or pick a sample below — I'll detect it and predict instantly. "
    "_(J and Z aren't supported — they require movement.)_"
)


def predict(image):
    if image is None:
        yield None, EMPTY_RESULT
        return

    yield None, "⏳ Detecting hand and classifying…"

    if isinstance(image, Image.Image):
        image = np.array(image)

    # 1. Try to find and crop the hand. On dataset samples (already pre-
    #    cropped to a tight 28x28 of just the hand), MediaPipe will fail
    #    to find a "hand" pose — we fall back to the original image.
    cropped, hand_detected = extract_hand_region(image)
    cnn_input = cropped if hand_detected else image

    detection_note = (
        "✓ Hand detected — cropped to the hand region before classifying."
        if hand_detected
        else "ℹ️ No hand detected — passed the image straight to the CNN "
             "(this is normal for pre-processed dataset samples)."
    )

    # 2. Build the preview the user sees of "what the model sees".
    preview = cv2.resize(cnn_input, (168, 168))
    if preview.ndim == 3:
        preview_gray = cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY)
    else:
        preview_gray = preview

    # 3. Classify.
    model = get_model()
    x = preprocess(cnn_input)
    probs = model.predict(x, verbose=0)[0]

    top_idx = np.argsort(probs)[::-1][:5]
    best_letter = LETTERS[int(top_idx[0])]
    best_prob = float(probs[top_idx[0]])

    lines = [
        f"### Prediction: **{best_letter}**  ·  {best_prob*100:.1f}% confident",
        "",
        f"_{detection_note}_",
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
        "_Heads-up: the CNN was trained on Sign Language MNIST (28x28 grayscale, "
        "uniform background, hand centered). MediaPipe Hands handles the framing "
        "for real-world photos, but unusual angles, partial occlusion, or "
        "low contrast can still trip it up._"
    )
    yield preview_gray, "\n".join(lines)


INTRO_MD = """
# Sign Language Recognition (ASL alphabet)

A small CNN that recognizes hand signs for 24 letters of the American Sign
Language alphabet (A-Y, excluding J and Z — those require motion).

**Pipeline:** the input image first goes through **MediaPipe Hands** to
detect and crop your hand, then the cropped region is passed to the CNN.
This keeps real-world photos close to the dataset domain the CNN was
trained on (Sign Language MNIST: uniform background, centered hand).

**How to try it**
1. Pick a sample below, upload a photo, or use your webcam.
2. The pipeline runs as soon as the image loads — no button to press.

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
                label="What the CNN sees (after hand crop, 28x28 grayscale)",
                height=224,
            )
            results_md = gr.Markdown(EMPTY_RESULT)

    input_image.change(predict, inputs=input_image, outputs=[preview_out, results_md])


if __name__ == "__main__":
    demo.launch()
