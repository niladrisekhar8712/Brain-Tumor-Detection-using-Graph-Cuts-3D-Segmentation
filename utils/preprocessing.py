# =============================================================================
#  utils/preprocessing.py
#  Handles everything that touches raw NIfTI files before the network sees them.
# =============================================================================

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage.restoration import denoise_nl_means, estimate_sigma


# ── helpers ──────────────────────────────────────────────────────────────────

def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI volume.  Returns (data_float32, affine)."""
    img  = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img.affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str) -> None:
    """Save a numpy array as a NIfTI file."""
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)


# ── bias-field approximation ─────────────────────────────────────────────────

def n4_bias_correction_approx(volume: np.ndarray) -> np.ndarray:
    """
    Lightweight approximation of N4 bias correction:
    fits a low-frequency polynomial surface to the log-intensity field and
    divides it out.  Full N4 (SimpleITK) is more accurate but requires an
    extra dependency; this version runs with only scipy/numpy.
    """
    from scipy.ndimage import gaussian_filter
    log_vol = np.log1p(np.clip(volume, 0, None))
    bias    = gaussian_filter(log_vol, sigma=20)
    corrected = np.expm1(log_vol - bias + bias.mean())
    return corrected.astype(np.float32)


# ── normalisation ─────────────────────────────────────────────────────────────

def zscore_normalise(volume: np.ndarray, brain_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Z-score normalise: (v - µ) / σ.
    If brain_mask is supplied, statistics are computed only over the brain.
    """
    if brain_mask is not None:
        vals = volume[brain_mask > 0]
    else:
        vals = volume[volume > 0]        # skip background voxels

    if vals.size == 0:
        return volume

    mu, sigma = vals.mean(), vals.std()
    if sigma < 1e-8:
        return volume - mu
    return ((volume - mu) / sigma).astype(np.float32)


# ── brain masking ─────────────────────────────────────────────────────────────

def compute_brain_mask(flair: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Simple threshold-based brain mask from FLAIR."""
    from scipy.ndimage import binary_fill_holes, binary_closing
    mask = (flair > threshold * flair.max()).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)
    mask = binary_closing(mask, iterations=3).astype(np.uint8)
    return mask


# ── cropping ──────────────────────────────────────────────────────────────────

def crop_to_brain(volumes: list[np.ndarray], mask: np.ndarray,
                  margin: int = 5) -> tuple[list[np.ndarray], np.ndarray, tuple]:
    """
    Crop all volumes (and the mask) to a tight bounding box around the brain,
    with a small margin.  Returns (cropped_volumes, cropped_mask, bbox).
    bbox = (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    coords = np.argwhere(mask > 0)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    D, H, W = mask.shape
    z0 = max(z0 - margin, 0);  z1 = min(z1 + margin, D)
    y0 = max(y0 - margin, 0);  y1 = min(y1 + margin, H)
    x0 = max(x0 - margin, 0);  x1 = min(x1 + margin, W)
    bbox = (z0, z1, y0, y1, x0, x1)
    cropped = [v[z0:z1, y0:y1, x0:x1] for v in volumes]
    return cropped, mask[z0:z1, y0:y1, x0:x1], bbox


# ── full preprocessing pipeline ───────────────────────────────────────────────

def preprocess_subject(paths: dict[str, str]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, tuple]:
    """
    Full preprocessing for one BraTS subject.

    Parameters
    ----------
    paths : dict with keys 'flair', 't1', 't1ce', 't2', and optionally 'seg'

    Returns
    -------
    volume4d   : float32 array  (4, D, H, W)  — 4 modalities stacked
    label      : uint8  array  (D, H, W)  or None if no 'seg' key
    affine     : (4,4) affine from the FLAIR file
    bbox       : bounding-box used to crop (for un-cropping later)
    """
    modality_keys = ['flair', 't1', 't1ce', 't2']
    raw = {}
    affine = None

    for key in modality_keys:
        vol, aff = load_nifti(paths[key])
        raw[key] = vol
        if affine is None:
            affine = aff

    # 1. Bias correction
    corrected = {k: n4_bias_correction_approx(v) for k, v in raw.items()}

    # 2. Brain mask (from FLAIR)
    brain_mask = compute_brain_mask(corrected['flair'])

    # 3. Z-score per modality
    normalised = {k: zscore_normalise(v, brain_mask) for k, v in corrected.items()}

    # 4. Crop to brain
    vol_list = [normalised[k] for k in modality_keys]
    cropped_vols, cropped_mask, bbox = crop_to_brain(vol_list, brain_mask)

    # 5. Stack into (4, D, H, W)
    volume4d = np.stack(cropped_vols, axis=0).astype(np.float32)

    # 6. Load label if present
    label = None
    if 'seg' in paths:
        seg_data, _ = load_nifti(paths['seg'])
        z0, z1, y0, y1, x0, x1 = bbox
        label = seg_data[z0:z1, y0:y1, x0:x1].astype(np.uint8)
        # BraTS labels: 1=necrosis, 2=edema, 4=enhancing → binarise to whole tumour
        label = (label > 0).astype(np.uint8)

    return volume4d, label, affine, bbox


# ── patch utilities ───────────────────────────────────────────────────────────

def extract_patches(volume: np.ndarray, patch_size: tuple, stride: tuple) -> list[tuple]:
    """
    Sliding-window patch extraction.

    Parameters
    ----------
    volume     : (C, D, H, W)
    patch_size : (pd, ph, pw)
    stride     : (sd, sh, sw)

    Returns
    -------
    List of (patch, (d_start, h_start, w_start))
    """
    C, D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    patches = []
    for d in range(0, max(D - pd + 1, 1), sd):
        for h in range(0, max(H - ph + 1, 1), sh):
            for w in range(0, max(W - pw + 1, 1), sw):
                d2 = min(d + pd, D);  h2 = min(h + ph, H);  w2 = min(w + pw, W)
                d1 = d2 - pd;         h1 = h2 - ph;         w1 = w2 - pw
                patch = volume[:, d1:d2, h1:h2, w1:w2]
                patches.append((patch, (d1, h1, w1)))
    return patches


def stitch_patches(patches_and_coords: list[tuple], volume_shape: tuple,
                   patch_size: tuple) -> np.ndarray:
    """
    Reconstruct a full volume from overlapping patches by averaging.

    patches_and_coords : list of (prob_patch (D,H,W), (d,h,w))
    volume_shape       : (D, H, W) of the full volume
    patch_size         : (pd, ph, pw)
    """
    accum  = np.zeros(volume_shape, dtype=np.float64)
    count  = np.zeros(volume_shape, dtype=np.float64)
    pd, ph, pw = patch_size
    for prob, (d, h, w) in patches_and_coords:
        accum[d:d+pd, h:h+ph, w:w+pw] += prob
        count[d:d+pd, h:h+ph, w:w+pw] += 1
    count[count == 0] = 1
    return (accum / count).astype(np.float32)
