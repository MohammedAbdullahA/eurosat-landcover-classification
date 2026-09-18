"""
Inspect the downloaded EuroSAT dataset: verify class folders, count real
images per class, check image dimensions, and write a summary CSV used
by scripts/make_plots.py and the proposal.

Run after scripts/download_data.py:
    python scripts/inspect_data.py
"""
import os
import csv
from collections import Counter
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "2750")
OUT_CSV = os.path.join(DATA_DIR, "class_counts.csv")


def main() -> None:
    if not os.path.isdir(RAW_DIR):
        raise SystemExit(
            f"Could not find {RAW_DIR}. Run scripts/download_data.py first."
        )

    classes = sorted(
        d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))
    )
    print(f"Found {len(classes)} class folders: {classes}\n")

    counts = Counter()
    dims = Counter()
    total = 0
    for cls in classes:
        cls_dir = os.path.join(RAW_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")]
        counts[cls] = len(files)
        total += len(files)
        # Sample a few images per class to confirm dimensions/mode
        for f in files[:5]:
            with Image.open(os.path.join(cls_dir, f)) as im:
                dims[(im.size, im.mode)] += 1

    print(f"{'Class':<22}{'#Images':>10}")
    print("-" * 32)
    for cls in classes:
        print(f"{cls:<22}{counts[cls]:>10}")
    print("-" * 32)
    print(f"{'TOTAL':<22}{total:>10}\n")

    print("Image size/mode found in sample scan:")
    for (size, mode), n in dims.items():
        print(f"  {size} {mode}: seen {n} times")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "num_images"])
        for cls in classes:
            writer.writerow([cls, counts[cls]])
    print(f"\nWrote per-class counts to {OUT_CSV}")


if __name__ == "__main__":
    main()
