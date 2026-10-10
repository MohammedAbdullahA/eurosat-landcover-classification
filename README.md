# Land Use and Land Cover Classification from Satellite Imagery (EuroSAT)

Repository: https://github.com/arbody/eurosat-landcover-classification

Deep learning course project — Milestone 1 (Project Proposal and Experimental
Plan). This repository holds the code supporting the proposal: dataset
acquisition, full-dataset inspection, the required figures, and the fixed
train/validation/test split that model-training work will build on for the
rest of the semester.

## Problem

Automatically classifying the land use / land cover shown in a satellite
image patch, using only its pixels. Manual photo-interpretation of Sentinel-2
imagery does not scale to continent-wide monitoring; a trained classifier
can support applications such as tracking urban sprawl, monitoring
agricultural land, spotting deforestation, and rapid land-cover mapping
after natural disasters.

- **Input:** a 64x64 RGB satellite image patch (3-band optical, derived from
  Sentinel-2 imagery), covering roughly 640m x 640m on the ground at 10m
  spatial resolution.
- **Output:** one of 10 land-use/land-cover class labels, plus the predicted
  probability across all ten.
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

- Images are 64x64 px, RGB, JPEG. We verified this file-by-file across all
  27,000 images (`scripts/sample_inspection.py`): zero corrupted files, zero
  images of a non-standard shape. We also computed per-class mean RGB
  intensity and flagged each class's brightest 2% of images as likely
  haze/cloud outliers rather than discarding them outright (see
  `data/class_intensity_stats.csv`).
- **License:** the EuroSAT labels/repo are released under the MIT license;
  the underlying Sentinel-2 imagery is free, full and open under the EU
  Copernicus data policy. Confirmed against both the GitHub repository and
  the Zenodo download page. Free to use for academic coursework.
- The dataset ships as a single unsplit pool — see "Train / validation /
  test split" below for how we divide it.

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

# 3. Full-dataset integrity + intensity pass: verify every one of the
#    27,000 images (shape, corruption), compute per-class mean RGB
#    intensity and brightness outliers -> data/class_intensity_stats.csv
python scripts/sample_inspection.py

# 4. Generate the required proposal figures into figures/
python scripts/make_plots.py

# 5. Create the fixed, stratified 70/15/15 train/val/test split
#    (data/splits/{train,val,test}.csv) used for all model training
#    this semester. Also prints and saves a per-class train/val/test
#    breakdown, verifying the stratification held exactly.
python scripts/split_dataset.py
```

Each script prints a clean, aligned console table (via `tabulate`) in
addition to writing its CSV output, so the pipeline's terminal output is
presentation-ready, not just raw prints. `figures/docx_sample_grid.png` and
`figures/docx_counts_intensity.png` (the two figures embedded in the
submitted proposal) are committed directly; other generated plots/tables
are build artifacts you can reproduce locally with the commands above and
are not tracked in git.

| Output | Produced by | Description |
|---|---|---|
| `figures/docx_sample_grid.png` | `make_plots.py` | One real image per class (committed) |
| `figures/docx_counts_intensity.png` | `make_plots.py` | Class counts + per-class mean RGB intensity (committed) |
| `data/class_counts.csv` | `inspect_data.py` | Raw per-class counts |
| `data/class_intensity_stats.csv` | `sample_inspection.py` | Per-class mean RGB intensity + brightness-outlier counts |
| `data/split_summary.csv` | `split_dataset.py` | Raw train/val/test breakdown per class |

## Train / validation / test split

EuroSAT does not ship an official split, so we define one **before any
model training**, stratified by class so all three sets keep the same
~2,000-3,000-per-class proportions as the full dataset:

| Split | Fraction | # Images | Purpose |
|---|---|---|---|
| Train | 70% | 18,900 | Fit model weights |
| Validation | 15% | 4,050 | Model selection, hyperparameter tuning, early stopping |
| Test | 15% | 4,050 | Final, untouched evaluation at the end of the semester |

The split is stratified per class with a fixed random seed (`42`) for
reproducibility (`scripts/split_dataset.py`), and the exact per-class
breakdown (confirming every class keeps the 70/15/15 proportion) is in
`data/split_summary.csv` and reproduced as Table 3 of the proposal.

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
├── data/
│   ├── raw/                      # not committed (see .gitignore) — created locally by download_data.py
│   ├── class_counts.csv          # committed — real per-class counts
│   ├── class_intensity_stats.csv # committed — per-class mean RGB intensity + outliers
│   ├── split_summary.csv         # committed — per-class train/val/test breakdown
│   └── splits/{train,val,test}.csv  # committed — the fixed stratified split
├── figures/
│   ├── docx_sample_grid.png        # committed — Figure 1 in the proposal
│   └── docx_counts_intensity.png   # committed — Figure 2 in the proposal
├── notebooks/
│   └── Milestone2_MLP_EuroSAT.ipynb  # Milestone 2: data pipeline + shallow/deep MLPs (executed)
├── proposal/
│   └── EuroSAT_Proposal.docx     # submitted Milestone 1 proposal (3 pages)
├── scripts/
│   ├── download_data.py
│   ├── inspect_data.py
│   ├── sample_inspection.py
│   ├── make_plots.py
│   └── split_dataset.py
├── requirements.txt
└── README.md
```

## Milestone 2: Data pipeline and MLP networks

`notebooks/Milestone2_MLP_EuroSAT.ipynb` is fully executed, so every output is visible without
re-running it. It contains:

- **A reusable PyTorch data pipeline** (`build_data_pipeline`). It loads the images and shows one
  per class, makes a stratified 80/20 train/test split, then a stratified 80/20 train/validation
  split (17,280 / 4,320 / 5,400 images). It flattens each image and standardises each channel
  using statistics from the training split only, encodes the labels, converts everything to
  tensors and builds the `DataLoader`s.
- **A flexible `MLP` class.** The shallow model has 2 hidden layers (512 → 256). The deep model
  has 5 (512 → 256 → 256 → 128 → 64). Every hidden layer uses ReLU, BatchNorm and Dropout 0.3,
  and the output is 10 logits with softmax.
- **A 3 × 3 comparison** of loss functions (cross-entropy, label smoothing, focal) against
  optimizers (SGD with momentum, RMSprop, Adam), followed by a 4-value learning-rate study and
  a shallow-vs-deep test comparison with learning curves.

| Result | Value |
|---|---|
| Best loss / optimizer / learning rate | Focal loss (γ = 2) / RMSprop / 1e-3 |
| Regularization strength | L2 weight decay = 0 (selected from {0, 1e-4, 1e-3}); dropout 0.3 |
| Shallow MLP — test accuracy / macro-F1 | 0.687 / 0.681 |
| Deep MLP — test accuracy / macro-F1 | 0.669 / 0.662 |

To re-run it: `pip install -r requirements.txt`, download the data (see above), then open the
notebook from the `notebooks/` folder. A full run takes about 25 minutes on a 2-core CPU.

## Team

- Mohammed Alroomi (202250760)
- Abdelrahman Osman (202253120)

## Proposal

`proposal/EuroSAT_Proposal.docx` is the submitted Milestone 1 proposal
(3 pages: title and team, problem definition, dataset description with
verified counts and sample inspection, both required figures, the
train/validation/test split with a per-class breakdown, and the validation
metric).

## Requirements checklist

| Requirement | Where it's satisfied |
|---|---|
| Define a real problem | Proposal §1, README "Problem" |
| Pick a real dataset and inspect it | Proposal §2; `scripts/inspect_data.py` + `scripts/sample_inspection.py` verify all 27,000 images |
| Define train/val/test splits before training | Proposal §3; `scripts/split_dataset.py`, stratified 70/15/15, seed 42, fixed before any model is trained |
| Pick a single validation metric | Proposal §4 — macro-averaged F1 |
| Sample visualization plot | Proposal Figure 1 (`figures/docx_sample_grid.png`) |
| Distribution plot | Proposal Figure 2 (`figures/docx_counts_intensity.png`) |
| 2-3 page proposal with title, team, and all requirements | `proposal/EuroSAT_Proposal.docx` — exactly 3 pages |
| GitHub repo with supporting code and README | This repository |
