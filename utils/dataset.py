# =============================================================================
#  utils/dataset.py
#  PyTorch Dataset that discovers BraTS subjects and serves (patch, label) pairs.
# =============================================================================

import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.preprocessing import preprocess_subject, extract_patches
import config as cfg


def discover_subjects(brats_root: str) -> list[dict]:
    """
    Walk brats_root and return a list of path-dicts, one per subject.
    Supports both BraTS 2020/2021 naming conventions.
    """
    subjects = []
    for subject_dir in sorted(glob.glob(os.path.join(brats_root, "*"))):
        if not os.path.isdir(subject_dir):
            continue
        name = os.path.basename(subject_dir)

        def find(tag):
            # try both .nii and .nii.gz
            for ext in ('.nii.gz', '.nii'):
                p = os.path.join(subject_dir, f"{name}_{tag}{ext}")
                if os.path.exists(p):
                    return p
            return None

        paths = {
            'flair': find('flair'),
            't1':    find('t1'),
            't1ce':  find('t1ce'),
            't2':    find('t2'),
            'seg':   find('seg'),
        }
        # skip if any modality is missing
        if all(v is not None for v in paths.values()):
            subjects.append(paths)

    # print(f"[Dataset] Found {len(subjects)} complete subjects in {brats_root}")
    # return subjects
    print(f"[Dataset] Found {len(subjects)} complete subjects in {brats_root}")
    max_s = getattr(cfg, 'MAX_SUBJECTS', None)
    if max_s:
        subjects = subjects[:max_s]
        print(f"[Dataset] Using only {len(subjects)} subjects (MAX_SUBJECTS limit)")
    return subjects

def split_subjects(subjects: list, val_frac: float, test_frac: float,
                   seed: int = 42) -> tuple[list, list, list]:
    random.seed(seed)
    data = subjects.copy()
    random.shuffle(data)
    n     = len(data)
    n_val  = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    test  = data[:n_test]
    val   = data[n_test:n_test + n_val]
    train = data[n_test + n_val:]
    print(f"[Dataset] Split → train: {len(train)}  val: {len(val)}  test: {len(test)}")
    return train, val, test


# ── augmentation helpers ──────────────────────────────────────────────────────

def random_flip(volume: np.ndarray, label: np.ndarray) -> tuple:
    for axis in range(1, 4):        # axes 1,2,3 → D,H,W
        if random.random() > 0.5:
            volume = np.flip(volume, axis=axis).copy()
            label  = np.flip(label,  axis=axis - 1).copy()
    return volume, label


def random_intensity_scale(volume: np.ndarray, lo=0.9, hi=1.1) -> np.ndarray:
    scale = random.uniform(lo, hi)
    return (volume * scale).astype(np.float32)


def random_gaussian_noise(volume: np.ndarray, std=0.02) -> np.ndarray:
    if random.random() > 0.5:
        volume = volume + np.random.randn(*volume.shape).astype(np.float32) * std
    return volume


# ── Dataset ──────────────────────────────────────────────────────────────────

class BraTSDataset(Dataset):
    """
    Returns random patches from BraTS subjects during training,
    or sequential patches during validation/inference.

    mode : 'train' | 'val' | 'test'
    """

    def __init__(self, subject_paths: list, mode: str = 'train',
                 patch_size: tuple = cfg.PATCH_SIZE,
                 patches_per_volume: int = 2):
                 # patches_per_volume: int = 4):
        self.subjects          = subject_paths
        self.mode              = mode
        self.patch_size        = patch_size
        self.patches_per_vol   = patches_per_volume
        self._cache: dict      = {}

    # ─────────────────────────────────────────────────────────────────────────

    def _load_subject(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load + preprocess subject; cache the result in RAM."""
        if idx not in self._cache:
            paths = self.subjects[idx]
            volume4d, label, affine, bbox = preprocess_subject(paths)
            self._cache[idx] = (volume4d, label)
        return self._cache[idx]

    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self):
        if self.mode == 'train':
            return len(self.subjects) * self.patches_per_vol
        else:
            return len(self.subjects)

    # ─────────────────────────────────────────────────────────────────────────

    def _random_patch(self, volume4d, label):
        """Sample one random patch, biased toward tumour regions."""
        C, D, H, W   = volume4d.shape
        pd, ph, pw   = self.patch_size

        # 50% chance: centre patch on a tumour voxel (positive mining)
        tumour_coords = np.argwhere(label > 0)
        if len(tumour_coords) > 0 and random.random() > 0.5:
            idx = random.randint(0, len(tumour_coords) - 1)
            cd, ch, cw = tumour_coords[idx]
        else:
            cd = random.randint(0, max(D - 1, 0))
            ch = random.randint(0, max(H - 1, 0))
            cw = random.randint(0, max(W - 1, 0))

        d0 = max(0, min(cd - pd // 2, D - pd))
        h0 = max(0, min(ch - ph // 2, H - ph))
        w0 = max(0, min(cw - pw // 2, W - pw))

        # Handle volumes smaller than patch size
        d0 = max(0, min(d0, max(D - pd, 0)))
        h0 = max(0, min(h0, max(H - ph, 0)))
        w0 = max(0, min(w0, max(W - pw, 0)))

        vol_patch = volume4d[:, d0:d0+pd, h0:h0+ph, w0:w0+pw]
        lbl_patch = label[d0:d0+pd, h0:h0+ph, w0:w0+pw]

        # Pad if needed (volume smaller than patch size)
        def pad_to(arr, target):
            pads = [(0, max(0, t - s)) for s, t in zip(arr.shape, target)]
            return np.pad(arr, pads, mode='constant')

        vol_patch = pad_to(vol_patch, (C, pd, ph, pw))
        lbl_patch = pad_to(lbl_patch, (pd, ph, pw))

        return vol_patch, lbl_patch

    # ─────────────────────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        if self.mode == 'train':
            subj_idx   = idx // self.patches_per_vol
            volume4d, label = self._load_subject(subj_idx)

            vol_patch, lbl_patch = self._random_patch(volume4d, label)

            # Augmentation
            vol_patch, lbl_patch = random_flip(vol_patch, lbl_patch)
            vol_patch = random_intensity_scale(vol_patch)
            vol_patch = random_gaussian_noise(vol_patch)

            return (torch.from_numpy(vol_patch.copy()),
                    torch.from_numpy(lbl_patch.copy()).unsqueeze(0).float())

        else:
            # For val/test return the full volume (used patch-wise in trainer)
            volume4d, label = self._load_subject(idx)
            return (torch.from_numpy(volume4d),
                    torch.from_numpy(label).unsqueeze(0).float()
                    if label is not None else torch.zeros(1))