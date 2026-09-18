"""
Full inspection pass over every EuroSAT image: verifies dimensions/mode,
checks for corrupted files, and computes per-class mean RGB channel
intensity (used for the proposal's "sample inspection" writeup and for
the counts+intensity figure).

Run after scripts/download_data.py:
    python scripts/sample_inspection.py
"""
import os
import csv
import numpy as np
from PIL import Image, UnidentifiedImageError

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "EuroSAT_RGB")
OUT_CSV = os.path.join(DATA_DIR, "class_intensity_stats.csv")


def main():
    classes = sorted(
        d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))
    )

    total = 0
    corrupted = 0
    non_standard_shape = 0
    rows = []

    for cls in classes:
        cls_dir = os.path.join(RAW_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")]
        means = []          # per-image (R,G,B) mean
        brightness = []     # per-image overall mean brightness (for outlier check)
        for f in files:
            path = os.path.join(cls_dir, f)
            total += 1
            try:
                with Image.open(path) as im:
                    if im.size != (64, 64) or im.mode != "RGB":
                        non_standard_shape += 1
                    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
            except (UnidentifiedImageError, OSError):
                corrupted += 1
                continue
            m = arr.reshape(-1, 3).mean(axis=0)
            means.append(m)
            brightness.append(arr.mean())

        means = np.array(means)
        brightness = np.array(brightness)
        # "possible cloud/haze" heuristic: unusually bright images (top 2%
        # of this class's own brightness distribution)
        thresh = np.percentile(brightness, 98)
        bright_outliers = int((brightness >= thresh).sum())

        r, g, b = means.mean(axis=0)
        rows.append({
            "class": cls,
            "n": len(files),
            "mean_R": round(float(r), 1),
            "mean_G": round(float(g), 1),
            "mean_B": round(float(b), 1),
            "mean_brightness": round(float(brightness.mean()), 1),
            "bright_outliers_98th_pct": bright_outliers,
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Inspected {total} images across {len(classes)} classes.")
    print(f"Corrupted / unreadable files: {corrupted}")
    print(f"Files not exactly 64x64 RGB: {non_standard_shape}")
    print(f"\nPer-class mean RGB intensity (0-255 scale):")
    for r in sorted(rows, key=lambda x: -x["mean_brightness"]):
        print(f"  {r['class']:<22} R={r['mean_R']:>6.1f}  G={r['mean_G']:>6.1f}  "
              f"B={r['mean_B']:>6.1f}  brightness={r['mean_brightness']:>6.1f}  "
              f"bright_outliers={r['bright_outliers_98th_pct']}")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
