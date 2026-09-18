"""
Generate the two required proposal figures from the real, downloaded data:

  1. figures/sample_grid.png       - one representative image per class
  2. figures/class_distribution.png - bar chart of images per class

Run after scripts/inspect_data.py:
    python scripts/make_plots.py
"""
import os
import csv
import random
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "2750")
COUNTS_CSV = os.path.join(DATA_DIR, "class_counts.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

random.seed(42)


def load_counts():
    classes, counts = [], []
    with open(COUNTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes.append(row["class"])
            counts.append(int(row["num_images"]))
    return classes, counts


def sample_grid(classes):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
    for ax, cls in zip(axes.flat, classes):
        cls_dir = os.path.join(RAW_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")]
        chosen = random.choice(files)
        img = Image.open(os.path.join(cls_dir, chosen))
        ax.imshow(img)
        ax.set_title(cls, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        "EuroSAT representative samples (one per class, 64×64 RGB, Sentinel-2)",
        fontsize=13,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "sample_grid.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


def distribution_plot(classes, counts):
    order = sorted(range(len(classes)), key=lambda i: counts[i], reverse=True)
    classes_sorted = [classes[i] for i in order]
    counts_sorted = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(classes_sorted, counts_sorted, color="#4C72B0")
    ax.set_ylabel("Number of images")
    ax.set_title("EuroSAT class distribution (27,000 images, 10 classes)")
    ax.set_xticklabels(classes_sorted, rotation=40, ha="right")
    for b, c in zip(bars, counts_sorted):
        ax.text(b.get_x() + b.get_width() / 2, c + 20, str(c), ha="center", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "class_distribution.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    classes, counts = load_counts()
    sample_grid(classes)
    distribution_plot(classes, counts)


if __name__ == "__main__":
    main()
