"""Generate animated preview of sign language recognition.

Uses the test set to show the actual model classifying real ASL letters,
with confidence bar.

Usage:
    python scripts/generate_preview.py [output.gif]
"""

import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CSV_LABEL_TO_LETTER, LABEL_TO_LETTER, IMAGE_SIZE, MODELS_DIR

FRAME_DURATION_MS = 1500
TEST_CSV = Path(__file__).parents[1] / "data" / "sign_mnist_test.csv"


def render_frame(image_28: np.ndarray, true_letter: str, probs: np.ndarray) -> Image.Image:
    top3 = np.argsort(probs)[::-1][:3]
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=100,
                                         gridspec_kw={"width_ratios": [1, 1.4]})
    ax_img.imshow(image_28, cmap="gray")
    ax_img.set_title(f"True: {true_letter}", fontsize=13, weight="bold")
    ax_img.axis("off")

    letters = [LABEL_TO_LETTER[int(i)] for i in top3]
    values = [probs[int(i)] * 100 for i in top3]
    colors = ["#10b981" if l == true_letter else "#94a3b8" for l in letters]
    ax_bar.barh(letters[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("confidence (%)")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_title(f"Top-3 predictions — predicted: {letters[0]}", fontsize=12)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main(output_path: Path) -> int:
    print("Loading model + test data...")
    model = keras.models.load_model(str(MODELS_DIR / "signlang_model.keras"))
    df = pd.read_csv(TEST_CSV)

    # CSV labels for A, F, N, T, V, Y (CSV uses raw alphabet indices, J=9 is absent)
    selected_csv_labels = [0, 5, 13, 19, 21, 24]
    rng = np.random.default_rng(42)

    frames = []
    for csv_label in selected_csv_labels:
        rows = df[df["label"] == csv_label]
        if len(rows) == 0:
            continue
        idx = int(rng.choice(len(rows)))
        row = rows.iloc[idx]
        img = row.values[1:].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
        img_in = (img / 127.5 - 1.0).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
        probs = model.predict(img_in, verbose=0)[0]
        letter = CSV_LABEL_TO_LETTER[csv_label]
        frames.append(render_frame(img.astype(np.uint8), letter, probs))
        print(f"  rendered {letter}")

    target = frames[0].size
    frames = [f.resize(target) for f in frames]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving GIF to {output_path}...")
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=FRAME_DURATION_MS, loop=0, optimize=True)
    print(f"Done. {len(frames)} frames, {output_path.stat().st_size / 1024:.0f} KB.")
    return 0


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parents[2] / "portfolio-website" / "public" / "previews" / "signlang.gif"
    )
    sys.exit(main(out))
