# Land Use and Land Cover Classification from Satellite Imagery (EuroSAT)

Deep learning course project. This repository holds the code supporting the
project proposal: dataset acquisition, inspection, the required figures, and
the fixed train/validation/test split that model-training work will build on
for the rest of the semester.

## Problem

Automatically classifying the land use / land cover shown in a satellite
image patch, using only its pixels. Manual photo-interpretation of Sentinel-2
imagery does not scale to continent-wide monitoring; a trained classifier
can support applications such as tracking urban sprawl, monitoring
agricultural land, spotting deforestation, and rapid land-cover mapping
after natural disasters.

- **Input:** a 64x64 RGB satellite image patch (3-band optical, derived from
  Sentinel-2 imagery).
- **Output:** one of 10 land-use/land-cover class labels.
- **Task type:** supervised multi-class (single-label) image classification.

## Dataset: EuroSAT

- Helber, P., Bischke, B., Dengel, A., Borth, D. (2019). *EuroSAT: A Novel
  Dataset and Deep Learning Benchmark for Land Use and Land Cover
  Classification.* IEEE JSTARS. [arXiv:1709.00029](https://arxiv.org/abs/1709.00029)
- Official repo: https://github.com/phelber/eurosat
- **27,000** labeled, geo-referenced images across **10 classes**, with a
  moderate class imbalance (verified by our own inspection,
  `scripts/inspect_data.py` -> `data/class_counts.csv`):

  | Class | # Images |
  |---|---|
  | AnnualCrop | 3,000 |
  | Forest | 3,000 |
  | HerbaceousVegetation | 3,000 |
  | Residential | 3,000 |
  | SeaLake | 3,000 |
  | Highway | 2,500 |
  | Industrial | 2,500 |
  | PermanentCrop | 2,500 |
  | River | 2,500 |
  | Pasture | 2,000 |
  | **Total** | **27,000** |

  See `figures/class_distribution.png` and `figures/sample_grid.png`
  (one real image per class).
- Images are 64x64 px, RGB, JPEG (verified on every file we inspected).
- **License:** the EuroSAT labels/repo are released under the MIT license;
  the underlying Sentinel-2 imagery is free, full and open under the EU
  Copernicus data policy. Free to use for academic coursework.
- The dataset ships as a single unsplit pool — see "Split" below for how we
  divide it.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the dataset inspection and required figures

```bash
# 1. Download + extract the official archive (~90-95 MB) into data/raw/
python scripts/download_data.py
# If your network blocks the host, download it yourself first, e.g.:
#   curl -L -o data/EuroSAT.zip "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"

# 2. Inspect the real files: verify class folders, count images/class,
#    check image dimensions, write data/class_counts.csv
python scripts/inspect_data.py

# 3. Generate the two required proposal figures into figures/
python scripts/make_plots.py

# 4. Create the fixed, stratified 70/15/15 train/val/test split
#    (data/splits/{train,val,test}.csv) used for all model training
#    this semester
python scripts/split_dataset.py
```

## Train / validation / test split

EuroSAT does not ship an official split, so we define one **before any
model training**, stratified by class so all three sets keep the same
~2,000-3,000-per-class proportions as the full dataset:

| Split | Fraction | Purpose |
|---|---|---|
| Train | 70% | Fit model weights |
| Validation | 15% | Model selection, hyperparameter tuning, early stopping |
| Test | 15% | Final, untouched evaluation at the end of the semester |

Split is stratified per class with a fixed random seed (`42`) for
reproducibility (`scripts/split_dataset.py`).

## Validation metric

**Macro-averaged F1-score** on the validation set is the single metric used
to track and compare models throughout the semester. Because class sizes
range from ~2,000 to ~3,000 images (not perfectly balanced), macro-F1
weighs every class equally and will not hide poor performance on the
smaller classes (e.g. Pasture) the way plain accuracy could. Accuracy is
also reported for reference, but macro-F1 is the metric used for model
selection and early stopping.

## Planned progression over the semester

1. Baseline: a small CNN trained from scratch.
2. Deeper CNN / regularization / data augmentation (rotation, flips — valid
   for satellite imagery since there is no canonical "up").
3. Transfer learning: fine-tune an ImageNet-pretrained backbone (e.g.
   ResNet-18/50) — this is close to how the original EuroSAT paper reached
   98.57% accuracy.
4. Stretch goal: repeat the comparison using the 13-band multispectral
   version of EuroSAT (`EuroSATallBands.zip`) instead of RGB, to see whether
   the extra Sentinel-2 spectral bands (e.g. near-infrared) improve
   land-cover discrimination.

## Repository layout

```
.
├── data/                 # not committed (see .gitignore) — created locally by download_data.py
├── figures/              # sample_grid.png, class_distribution.png (generated)
├── proposal/             # proposal PDF + source
├── scripts/
│   ├── download_data.py
│   ├── inspect_data.py
│   ├── make_plots.py
│   └── split_dataset.py
├── requirements.txt
└── README.md
```

## Team

- Abdelrahman (arbody.arb@gmail.com)
- (add teammates here)
