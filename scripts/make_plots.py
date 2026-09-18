
import os
import csv
import random
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "EuroSAT_RGB")
COUNTS_CSV = os.path.join(DATA_DIR, "class_counts.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

random.seed(42)

# Theme palette (sampled from the project's brand: deep green-black bg,
# mint accent, lavender secondary accent)
BG = "#0d1f1c"
CARD = "#0f211e"
BORDER = "#1f4038"
MINT = "#8ef7a0"
LAVENDER = "#716be6"
WHITE = "#ffffff"
MUTED = "#a6b3af"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": WHITE,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "DejaVu Sans",
})


def load_counts():
    classes, counts = [], []
    with open(COUNTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes.append(row["class"])
            counts.append(int(row["num_images"]))
    return classes, counts


def sample_grid(classes):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.9), facecolor=BG)
    for ax, cls in zip(axes.flat, classes):
        cls_dir = os.path.join(RAW_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")]
        chosen = random.choice(files)
        img = Image.open(os.path.join(cls_dir, chosen))
        ax.imshow(img)
        ax.set_title(cls, fontsize=11.5, color=MINT, fontweight="bold", pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "EuroSAT representative samples  ·  one per class  ·  64×64 RGB, Sentinel-2",
        fontsize=13, color=WHITE, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "sample_grid.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out}")


def distribution_plot(classes, counts):
    order = sorted(range(len(classes)), key=lambda i: counts[i], reverse=True)
    classes_sorted = [classes[i] for i in order]
    counts_sorted = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
    bars = ax.bar(classes_sorted, counts_sorted, color=MINT, width=0.65, zorder=3)
    # Tint the smallest classes with the lavender accent to draw the eye to imbalance
    min_count = min(counts_sorted)
    for b, c in zip(bars, counts_sorted):
        if c == min_count:
            b.set_color(LAVENDER)

    ax.set_ylabel("Number of images", fontsize=10)
    ax.set_title("EuroSAT class distribution  ·  27,000 images, 10 classes",
                 fontsize=13, color=WHITE, fontweight="bold", pad=14)
    ax.set_xticks(range(len(classes_sorted)))
    ax.set_xticklabels(classes_sorted, rotation=40, ha="right", fontsize=9.3)
    ax.grid(axis="y", color=BORDER, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(length=0)

    for b, c in zip(bars, counts_sorted):
        ax.text(b.get_x() + b.get_width() / 2, c + 45, str(c), ha="center",
                 fontsize=9, color=WHITE, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "class_distribution.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    classes, counts = load_counts()
    sample_grid(classes)
    distribution_plot(classes, counts)


if __name__ == "__main__":
    main()
