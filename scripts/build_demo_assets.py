"""Generate demo assets for the Gradio space.

Pulls one clean sample per letter (A-Y, excluding J and Z) from the test set
and saves:
  - space/sample_images/<letter>.png  (upscaled per-letter sample)
  - space/alphabet_chart.png          (6x4 reference grid with letter labels)

Usage:
    python scripts/build_demo_assets.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CSV_LABEL_TO_LETTER, IMAGE_SIZE, TEST_CSV

SPACE_DIR = Path(__file__).parents[1] / "space"
SAMPLES_DIR = SPACE_DIR / "sample_images"
CHART_PATH = SPACE_DIR / "alphabet_chart.png"

SAMPLE_UPSCALE = 6  # 28 -> 168 px


def select_one_per_letter(df: pd.DataFrame, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for label, letter in CSV_LABEL_TO_LETTER.items():
        rows = df[df["label"] == label]
        if len(rows) == 0:
            print(f"  ! no rows for label {label} ({letter})")
            continue
        idx = int(rng.choice(len(rows)))
        img = rows.iloc[idx].values[1:].astype(np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)
        out[letter] = img
    return out


def save_per_letter_samples(images: dict[str, np.ndarray]) -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in SAMPLES_DIR.glob("*.png"):
        old.unlink()
    for letter, img in images.items():
        pil = Image.fromarray(img, mode="L").resize(
            (IMAGE_SIZE * SAMPLE_UPSCALE, IMAGE_SIZE * SAMPLE_UPSCALE),
            resample=Image.NEAREST,
        )
        pil.save(SAMPLES_DIR / f"{letter}.png")
    print(f"Saved {len(images)} samples -> {SAMPLES_DIR}")


def render_alphabet_chart(images: dict[str, np.ndarray]) -> None:
    cols, rows = 6, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.85), dpi=140)
    fig.suptitle("ASL alphabet — imitate any of these poses", fontsize=14, weight="bold")

    letters = list(images.keys())
    for ax, letter in zip(axes.flat, letters):
        ax.imshow(images[letter], cmap="gray", interpolation="nearest")
        ax.set_title(letter, fontsize=14, weight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")

    for ax in list(axes.flat)[len(letters):]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(CHART_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved chart -> {CHART_PATH} ({CHART_PATH.stat().st_size / 1024:.0f} KB)")


def main() -> int:
    print(f"Reading {TEST_CSV.name}...")
    df = pd.read_csv(TEST_CSV)
    print(f"  {len(df)} rows, {df['label'].nunique()} unique labels")

    images = select_one_per_letter(df)
    if len(images) != len(CSV_LABEL_TO_LETTER):
        print(f"  ! got {len(images)} letters, expected {len(CSV_LABEL_TO_LETTER)}")

    save_per_letter_samples(images)
    render_alphabet_chart(images)
    return 0


if __name__ == "__main__":
    sys.exit(main())
