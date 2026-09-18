"""
Create a fixed, stratified train/validation/test split BEFORE any model
training, as required by the proposal, and produce a presentation-ready
summary table (console + themed PNG) verifying the stratification held.

EuroSAT ships as a single pool of 27,000 labeled images with no official
train/val/test split, so we define our own:

    70% train / 15% validation / 15% test, stratified by class,
    fixed random seed (42) for reproducibility.

Writes data/splits/{train,val,test}.csv, each with columns:
    filepath,label
and data/split_summary.csv with the per-class, per-split breakdown.

Run after scripts/download_data.py:
    python scripts/split_dataset.py
"""
import os
import csv
import random
from collections import defaultdict
from tabulate import tabulate
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "EuroSAT_RGB")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
SUMMARY_CSV = os.path.join(DATA_DIR, "split_summary.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42

# Theme (matches scripts/make_plots.py)
BG = "#0d1f1c"
MINT = "#8ef7a0"
MINT_DARK_TEXT = "#0a1a17"
WHITE = "#ffffff"
MUTED = "#a6b3af"
BORDER = "#24443c"


def save_table_image(rows, headers, out_path, title):
    n_rows = len(rows) + 1
    fig_h = 0.4 * n_rows + 0.6
    fig, ax = plt.subplots(figsize=(7.2, fig_h), facecolor=BG)
    ax.axis("off")
    ax.set_title(title, color=WHITE, fontsize=12, fontweight="bold", pad=14)

    col_widths = [0.24, 0.13, 0.11, 0.11, 0.12, 0.22]
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(MINT)
            cell.set_text_props(color=MINT_DARK_TEXT, fontweight="bold")
        else:
            cell.set_facecolor(BG if r % 2 else "#132a25")
            cell.set_text_props(color=WHITE if c == 0 else MUTED)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved table image -> {out_path}")


def main():
    random.seed(SEED)
    if not os.path.isdir(RAW_DIR):
        raise SystemExit(f"Could not find {RAW_DIR}. Run scripts/download_data.py first.")

    os.makedirs(SPLIT_DIR, exist_ok=True)
    classes = sorted(
        d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))
    )

    splits = defaultdict(list)
    per_class_split_counts = {cls: {"train": 0, "val": 0, "test": 0} for cls in classes}

    for cls in classes:
        cls_dir = os.path.join(RAW_DIR, cls)
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg"))
        random.shuffle(files)

        n = len(files)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for name, subset in [("train", train_files), ("val", val_files), ("test", test_files)]:
            per_class_split_counts[cls][name] = len(subset)
            for f in subset:
                splits[name].append((os.path.join("data", "raw", "EuroSAT_RGB", cls, f), cls))

    for name, rows in splits.items():
        random.shuffle(rows)
        out_path = os.path.join(SPLIT_DIR, f"{name}.csv")
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["filepath", "label"])
            writer.writerows(rows)

    # ---- Presentation-ready summary table (verifies stratification held) ----
    summary_rows = []
    totals = {"train": 0, "val": 0, "test": 0}
    for cls in sorted(classes):
        c = per_class_split_counts[cls]
        total = c["train"] + c["val"] + c["test"]
        summary_rows.append([
            cls, c["train"], c["val"], c["test"], total,
            f"{c['train']/total:.0%}/{c['val']/total:.0%}/{c['test']/total:.0%}",
        ])
        for k in totals:
            totals[k] += c[k]
    grand_total = sum(totals.values())
    summary_rows.append([
        "TOTAL", totals["train"], totals["val"], totals["test"], grand_total,
        f"{totals['train']/grand_total:.0%}/{totals['val']/grand_total:.0%}/{totals['test']/grand_total:.0%}",
    ])

    headers = ["Class", "Train", "Val", "Test", "Total", "Train/Val/Test %"]
    print(tabulate(summary_rows, headers=headers, tablefmt="fancy_grid"))

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(summary_rows)
    print(f"\nWrote split summary to {SUMMARY_CSV}")

    for name in ("train", "val", "test"):
        print(f"{name}: {len(splits[name])} images -> {os.path.join(SPLIT_DIR, name + '.csv')}")

    os.makedirs(FIG_DIR, exist_ok=True)
    save_table_image(
        [[r[0], f"{r[1]:,}", f"{r[2]:,}", f"{r[3]:,}", f"{r[4]:,}", r[5]] for r in summary_rows],
        headers,
        os.path.join(FIG_DIR, "split_summary_table.png"),
        "EuroSAT — stratified 70/15/15 train/val/test split (seed=42)",
    )


if __name__ == "__main__":
    main()
