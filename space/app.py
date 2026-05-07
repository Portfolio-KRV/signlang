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
    """Lazily build the MediaPipe Hands detector (static-image mode).

    Low confidence threshold (0.1) — we'd rather have a few false
    detections we can sanity-check via the overlay than miss real hands.
    """
    global _hands
    if _hands is None:
        _hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.1,
        )
    return _hands


# MediaPipe Hands rinde mejor en resoluciones medias. Imágenes muy chicas
# pierden landmarks por upscaling interno; muy grandes se vuelven lentas.
MIN_LONG_EDGE = 480
MAX_LONG_EDGE = 1280
# Imágenes ≤ este lado son tratadas como samples del dataset (28×28
# pre-cropped). MediaPipe no encuentra "manos" en ellas y eso es esperado.
SAMPLE_HEURISTIC = 100


def normalize_for_detection(image: np.ndarray) -> np.ndarray:
    """Resize the image to a range MediaPipe handles well, keeping aspect."""
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge < MIN_LONG_EDGE:
        scale = MIN_LONG_EDGE / long_edge
    elif long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
    else:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_hand_region(image: np.ndarray, padding: float = 0.2):
    """Detect a hand with MediaPipe and return a square crop around it.

    Returns:
        (cropped_rgb, detected_bool, landmarks_or_none, bbox_or_none).
        If no hand is detected, returns (None, False, None, None) and the
        caller can fall back to the original image.
    """
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        rgb = image

    rgb = normalize_for_detection(rgb)
    h, w = rgb.shape[:2]
    results = get_hands_detector().process(rgb)

    if not results.multi_hand_landmarks:
        return None, False, None, None

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
        return None, False, None, None

    bbox = (sx_min, sy_min, sx_max, sy_max)
    return cropped, True, landmarks, bbox


def render_detection_overlay(image: np.ndarray, landmarks, bbox) -> np.ndarray:
    """Draw the MediaPipe landmark skeleton + crop box on the original image.

    Lets the user see what MediaPipe found before the CNN runs — useful
    for diagnosing why a real-world photo classifies wrong.
    """
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        overlay = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        overlay = image.copy()

    if landmarks is not None:
        mp.solutions.drawing_utils.draw_landmarks(
            overlay,
            landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 144, 255), 3)
    return overlay


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
        yield None, None, EMPTY_RESULT
        return

    yield None, None, "⏳ Detecting hand and classifying…"

    if isinstance(image, Image.Image):
        image = np.array(image)

    # 1. Try to find and crop the hand. Behavior diverges by image size:
    #    - Small images (≤100px) are dataset samples — MediaPipe usually
    #      can't find a "hand" pose in the 28×28 crops, but the CNN can
    #      classify them directly. Fall back to the original image.
    #    - Larger images are real photos — if MediaPipe fails, the CNN
    #      will produce noise. Tell the user to retake.
    h_in, w_in = image.shape[:2]
    is_sample = max(h_in, w_in) <= SAMPLE_HEURISTIC

    cropped, hand_detected, landmarks, bbox = extract_hand_region(image)

    if hand_detected:
        cnn_input = cropped
        detection_overlay = render_detection_overlay(image, landmarks, bbox)
        detection_note = (
            f"✓ Hand detected — cropped to a {bbox[2]-bbox[0]}×{bbox[3]-bbox[1]}px "
            "square around the hand before classifying."
        )
    elif is_sample:
        cnn_input = image
        detection_overlay = image
        detection_note = (
            "ℹ️ Dataset sample — passed straight to the CNN "
            "(MediaPipe doesn't try on pre-cropped 28×28 inputs)."
        )
    else:
        # Real-world photo where MediaPipe couldn't find a hand. Skipping
        # CNN inference and asking for a better shot is more honest than
        # serving a noisy prediction.
        yield image, None, (
            "### ⚠️ Couldn't locate a hand in your photo\n\n"
            "MediaPipe couldn't detect a hand in this image. Try:\n\n"
            "- Better lighting (avoid backlight, move closer to a lamp)\n"
            "- Center your hand in frame, palm facing the camera\n"
            "- Plain background helps the detector lock on\n"
            "- Or pick a sample below to see the CNN classify directly"
        )
        return

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
        "_Heads-up: the CNN was trained on Sign Language MNIST (28×28 grayscale, "
        "uniform background, hand centered). MediaPipe Hands handles the framing "
        "for real-world photos, but unusual angles, partial occlusion, or "
        "low contrast can still trip it up. Compare the two previews on the "
        "right to see what each stage of the pipeline produced._"
    )
    yield detection_overlay, preview_gray, "\n".join(lines)


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
            detection_out = gr.Image(
                label="Step 1 — MediaPipe Hands (landmarks + crop box)",
                height=240,
            )
            preview_out = gr.Image(
                label="Step 2 — What the CNN sees (28×28 grayscale)",
                height=180,
            )
            results_md = gr.Markdown(EMPTY_RESULT)

    input_image.change(
        predict,
        inputs=input_image,
        outputs=[detection_out, preview_out, results_md],
    )


if __name__ == "__main__":
    demo.launch()
