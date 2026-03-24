# 3D Brain Tumor Segmentation
**Jadavpur University — Electronics & Tele-Communication Engineering**  
*Niladri Sekhar Mondal & Rivuprovo De | Supervised by Prof. Ananda Shankar Chowdhury*

---

## What this project does

Implements the full hybrid pipeline from the final-year project report:

```
NIfTI MRI (4 modalities)
        │
        ▼
[1] Preprocessing       Approx. N4 bias correction, z-score normalisation, brain crop
        │
        ▼
[2] 3D U-Net            Sliding-window inference → voxel-wise probability map
        │
        ▼
[3] SLIC Supervoxels    Groups voxels into ~1000 perceptually meaningful regions (ROI only)
        │
        ▼
[4] Graph Cut           Min-cut on the Region Adjacency Graph → refined mask
        │
        ▼
Outputs: segmentation_mask.nii.gz | cnn_probability_map.nii.gz | visualisation.png | Dice / HD95 metrics
```

---

## Project structure

```
brain_tumor_seg/
├── config.py               ← SET YOUR PATHS AND HYPERPARAMETERS HERE FIRST
├── train.py                ← Step 1: train the 3D U-Net
├── predict.py              ← Step 2: run inference on new scans
├── plot_history.py         ← Optional: plot training curves
├── requirements.txt
├── models/
│   ├── unet3d.py           ← Attention 3D U-Net + loss functions + metrics
│   └── graph_cut.py        ← SLIC supervoxels + RAG + max-flow graph cut
└── utils/
    ├── preprocessing.py    ← NIfTI loading, approx. bias correction, normalisation, patching
    ├── dataset.py          ← PyTorch Dataset (BraTS auto-discovery, augmentation)
    └── visualise.py        ← Matplotlib overlays, probability maps, history plots
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118
pip install nibabel numpy matplotlib scikit-image scipy tqdm maxflow
```

> **Windows note:** if `NUM_WORKERS > 0` causes errors, set `NUM_WORKERS = 0` in `config.py`.

### 2. Download the BraTS dataset

- Register and download from: https://www.synapse.org/#!Synapse:syn51156910/wiki/621282
- BraTS 2021 is recommended (the default naming convention is supported).
- Extract so the folder looks like:

```
BraTS2021_Training_Data/
    BraTS2021_00001/
        BraTS2021_00001_flair.nii.gz
        BraTS2021_00001_t1.nii.gz
        BraTS2021_00001_t1ce.nii.gz
        BraTS2021_00001_t2.nii.gz
        BraTS2021_00001_seg.nii.gz
    BraTS2021_00002/
        ...
```

### 3. Configure paths

Open `config.py` and set:

```python
BRATS_ROOT = r"C:\Users\YourName\data\BraTS2021_Training_Data"   # ← your actual path
DEVICE     = "cuda"   # or "cpu" if no GPU
```

---

## Training

```bash
python train.py
```

- Automatically splits the dataset into train / val / test (70% / 20% / 10% by default).
- Caps the number of subjects loaded via `MAX_SUBJECTS` in `config.py` (default: 200).
- Uses **AdamW** optimiser with `ReduceLROnPlateau` scheduler (halves LR after 15 epochs without improvement).
- Loss function: **Dice + BCE** combined (60% Dice weight, 40% BCE weight).
- Gradient clipping at `max_norm=1.0` is applied every step.
- Saves the best checkpoint to `checkpoints/best_model.pth`.
- Saves the last checkpoint to `checkpoints/last_model.pth` (training resumes automatically if interrupted).
- Triggers **early stopping** after 30 epochs without improvement (`EARLY_STOP` in `config.py`).
- Prints train loss and val Dice after every epoch.
- Saves `checkpoints/history.npy` and `checkpoints/test_subjects.npy` for later use.

To plot training curves after training:

```bash
python plot_history.py
```

Output saved to `outputs/training_history.png`.

---

## Inference (predict on new scans)

### Single subject

```bash
python predict.py \
    --flair  data/BraTS21_001/BraTS21_001_flair.nii.gz \
    --t1     data/BraTS21_001/BraTS21_001_t1.nii.gz    \
    --t1ce   data/BraTS21_001/BraTS21_001_t1ce.nii.gz  \
    --t2     data/BraTS21_001/BraTS21_001_t2.nii.gz    \
    --seg    data/BraTS21_001/BraTS21_001_seg.nii.gz   # optional ground-truth
```

You can also specify a custom model checkpoint:

```bash
python predict.py --flair ... --model checkpoints/my_checkpoint.pth
```

### All test subjects (from the training split)

```bash
python predict.py --all_test
```

Requires `checkpoints/test_subjects.npy` to exist (saved automatically by `train.py`).

### Outputs (per subject, saved in `outputs/<subject_name>/`)

| File | Description |
|---|---|
| `segmentation_mask.nii.gz` | Binary tumour mask (after graph cut) |
| `cnn_probability_map.nii.gz` | Raw CNN sigmoid probabilities |
| `visualisation.png` | Multi-panel figure with tumour highlighted |

The visualisation shows:
- FLAIR with **red tumour overlay**
- T1ce with **predicted contour**
- CNN **probability heatmap**
- Predicted mask vs ground-truth mask (when available)

### Terminal output example

```
────────────────────────────────────────────────────────────
  Subject: BraTS2021_00001
────────────────────────────────────────────────────────────
[1/4] Preprocessing …
      Volume shape: (4, 138, 172, 138)  (3.2s)
[2/4] Running 3D U-Net (sliding window) …
      Prob map range: [0.001, 0.997]  (8.4s)
[3/4] Supervoxel-based Graph Cut …
  [GraphCut] Computing initial CNN threshold mask ...
  [GraphCut] ROI shape: (48, 52, 44)
  [GraphCut] Generating supervoxels in ROI ...
  [GraphCut] 312 supervoxels generated
  [GraphCut] Running min-cut on 312 nodes, 1048 edges ...
  [GraphCut] Keeping largest component ...
      Tumour voxels: 12,405  (4.1s)
[4/4] Computing metrics and saving outputs …

  ┌─────────────────────────────────┐
  │         RESULTS SUMMARY          │
  ├─────────────────────────────────┤
  │  Dice (CNN only)  : 0.8412        │
  │  Dice (GraphCut)  : 0.8731        │
  │  HD95 (voxels)    : 6.24         │
  └─────────────────────────────────┘
```

> **Note:** The graph cut runs only inside a bounding-box ROI around the CNN detection, not the full volume. If the ROI exceeds 30% of the total brain volume, or if the graph cut result is empty or implausibly large (>3× the CNN mask), the pipeline falls back to the CNN threshold mask directly.

---

## Quick data viewer (in PyCharm console)

```python
from utils.visualise import quick_view
quick_view("path/to/flair.nii.gz", "path/to/segmentation_mask.nii.gz")
```

---

## Key hyperparameters (all in `config.py`)

| Parameter | Default | Description |
|---|---|---|
| `PATCH_SIZE` | (64, 64, 64) | Training and inference patch size |
| `PATCH_STRIDE` | (64, 64, 64) | Stride for sliding-window inference |
| `N_SUPERVOXELS` | 1000 | SLIC target supervoxels (capped at 300 inside ROI) |
| `SLIC_COMPACTNESS` | 0.1 | SLIC spatial vs. colour compactness trade-off |
| `GC_LAMBDA` | 100.0 | Smoothness weight λ for graph cut |
| `GC_SIGMA` | 0.05 | Pairwise Gaussian σ for graph cut |
| `BASE_FILTERS` | 16 | U-Net base feature maps |
| `ENCODER_DEPTHS` | 3 | Number of encoder/decoder levels |
| `NUM_EPOCHS` | 150 | Max training epochs |
| `LR` | 1e-3 | Initial learning rate |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `LR_PATIENCE` | 15 | Epochs before LR is halved |
| `EARLY_STOP` | 30 | Epochs without improvement before stopping |
| `BATCH_SIZE` | 4 | Training batch size |
| `VAL_FRACTION` | 0.20 | Fraction of data used for validation |
| `TEST_FRACTION` | 0.10 | Fraction of data held out for testing |
| `MAX_SUBJECTS` | 200 | Cap on subjects loaded (set `None` to use all) |
| `NUM_WORKERS` | 0 | DataLoader workers (set 0 on Windows) |
| `DEVICE` | "cuda" | "cuda" or "cpu" |
