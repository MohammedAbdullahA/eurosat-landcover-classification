"""
Generate the two figures for the Word-document proposal (light/academic
style, since they're embedded in a white-page Word doc rather than the
dark-themed web page):

  1. figures/docx_sample_grid.png       - one representative image per class
  2. figures/docx_counts_intensity.png  - image counts per class, paired
                                           with per-class mean RGB channel
                                           intensity

Run after scripts/sample_inspection.py (needs data/class_intensity_stats.csv)
and scripts/inspect_data.py (needs data/class_counts.csv):
    python scripts/make_docx_figures.py
"""
import os
import csv
import random
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "EuroSAT_RGB")
COUNTS_CSV = os.path.join(DATA_DIR, "class_counts.csv")
INTENSITY_CSV = os.path.join(DATA_DIR, "class_intensity_stats.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

random.seed(42)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
})


def load_csv_dict(path, key):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row[key]] = row
    return rows


def sample_grid(classes):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.6), facecolor="white")
    for ax, cls in zip(axes.flat, classes):
        cls_dir = os.path.join(RAW_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")]
        chosen = random.choice(files)
        img = Image.open(os.path.join(cls_dir, chosen))
        ax.imshow(img)
        ax.set_title(cls, fontsize=11.5, color="#1a1a1a", fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#c9c9c9")
            spine.set_linewidth(1)
    fig.suptitle(
        "EuroSAT representative samples — one per class, 64×64 RGB, Sentinel-2",
        fontsize=13, color="#1a1a1a", fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "docx_sample_grid.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def counts_intensity_plot(classes, counts, intensity):
    order = sorted(classes, key=lambda c: -int(counts[c]["num_images"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.4), facecolor="white")

    # Left: image counts per class
    vals = [int(counts[c]["num_images"]) for c in order]
    bars = ax1.bar(order, vals, color="#3d6ea5", width=0.65, zorder=3)
    ax1.set_title("Image count per class", fontsize=12, fontweight="bold", color="#1a1a1a")
    ax1.set_ylabel("Number of images")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(order, rotation=42, ha="right", fontsize=8.6)
    ax1.grid(axis="y", color="#e2e2e2", linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width()/2, v + 40, f"{v:,}", ha="center", fontsize=7.8)

    # Right: mean R/G/B channel intensity per class (same order)
    r_vals = [float(intensity[c]["mean_R"]) for c in order]
    g_vals = [float(intensity[c]["mean_G"]) for c in order]
    b_vals = [float(intensity[c]["mean_B"]) for c in order]
    x = range(len(order))
    w = 0.26
    ax2.bar([i - w for i in x], r_vals, width=w, color="#c0392b", label="Mean R")
    ax2.bar(x, g_vals, width=w, color="#27ae60", label="Mean G")
    ax2.bar([i + w for i in x], b_vals, width=w, color="#2e6e9e", label="Mean B")
    ax2.set_title("Mean pixel intensity per channel", fontsize=12, fontweight="bold", color="#1a1a1a")
    ax2.set_ylabel("Mean intensity (0–255)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(order, rotation=42, ha="right", fontsize=8.6)
    ax2.grid(axis="y", color="#e2e2e2", linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "docx_counts_intensity.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    counts = load_csv_dict(COUNTS_CSV, "class")
    intensity = load_csv_dict(INTENSITY_CSV, "class")
    classes = sorted(counts.keys())
    os.makedirs(FIG_DIR, exist_ok=True)
    sample_grid(classes)
    counts_intensity_plot(classes, counts, intensity)


if __name__ == "__main__":
    main()
