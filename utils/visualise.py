# =============================================================================
#  utils/visualise.py
#  Matplotlib-based visualisation helpers.
# =============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

TUMOR_CMAP = LinearSegmentedColormap.from_list(
    "tumor", [(0, 0, 0, 0), (1, 0.1, 0.1, 0.85)], N=256)

PROB_CMAP = plt.cm.inferno


def _mid_slice(vol: np.ndarray, axis: int) -> np.ndarray:
    idx = vol.shape[axis] // 2
    return np.take(vol, idx, axis=axis)


def _normalise_for_display(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip(img, lo, hi)
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _best_slice_idx(vol3d: np.ndarray, axis: int) -> int:
    """Return index of slice with most non-zero voxels."""
    counts = [np.take(vol3d, i, axis=axis).sum()
              for i in range(vol3d.shape[axis])]
    return int(np.argmax(counts)) if max(counts) > 0 else vol3d.shape[axis] // 2


def visualise_results(volume4d:  np.ndarray,
                      prob_map:  np.ndarray,
                      pred_mask: np.ndarray,
                      gt_mask:   np.ndarray | None = None,
                      save_path: str = "visualisation.png",
                      title:     str = "Segmentation Result",
                      show:      bool = False):

    flair = _normalise_for_display(volume4d[0])
    t1ce  = _normalise_for_display(volume4d[2])

    axes_labels = ['Axial', 'Coronal', 'Sagittal']

    # Use ground truth to find best slices if available,
    # otherwise use prob_map, otherwise use middle slice
    if gt_mask is not None and gt_mask.sum() > 0:
        reference = gt_mask
    elif prob_map.max() > 0.3:
        reference = (prob_map > 0.3).astype(np.uint8)
    else:
        reference = None

    # Get best slice indices
    slice_indices = []
    for axis in range(3):
        if reference is not None:
            idx = _best_slice_idx(reference, axis)
        else:
            idx = volume4d.shape[axis + 1] // 2
        slice_indices.append(idx)

    def slices(vol3d):
        result = []
        for axis, idx in enumerate(slice_indices):
            result.append(np.rot90(np.take(vol3d, idx, axis=axis)))
        return result

    flair_s = slices(flair)
    t1ce_s  = slices(t1ce)
    prob_s  = slices(prob_map)
    pred_s  = slices(pred_mask)
    gt_s    = slices(gt_mask) if gt_mask is not None else None

    n_rows = 4 if gt_mask is None else 5
    n_cols = 3
    fig = plt.figure(figsize=(n_cols * 4.5, n_rows * 4), facecolor='#0a0a0a')
    fig.suptitle(title, fontsize=16, color='white', fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.08, wspace=0.05)

    def ax(row, col):
        a = fig.add_subplot(gs[row, col])
        a.axis('off')
        a.set_facecolor('black')
        return a

    # ── Row 0: FLAIR + tumour overlay ─────────────────────────────────────────
    for c in range(3):
        a = ax(0, c)
        a.imshow(flair_s[c], cmap='gray', vmin=0, vmax=1)
        if pred_s[c].any():
            a.imshow(pred_s[c].astype(float), cmap=TUMOR_CMAP,
                     alpha=0.7, vmin=0, vmax=1)
        if c == 0:
            a.set_title("FLAIR + Tumour Overlay", color='white',
                        fontsize=10, pad=4)
        a.text(0.02, 0.96, axes_labels[c], transform=a.transAxes,
               color='#aaaaaa', fontsize=8, va='top')

    # ── Row 1: T1ce + predicted contour ───────────────────────────────────────
    for c in range(3):
        a = ax(1, c)
        a.imshow(t1ce_s[c], cmap='gray', vmin=0, vmax=1)
        if pred_s[c].any():
            a.contour(pred_s[c], levels=[0.5],
                      colors=['#ff4c6e'], linewidths=1.5)
        if c == 0:
            a.set_title("T1ce + Predicted Contour", color='white',
                        fontsize=10, pad=4)
        a.text(0.02, 0.96, axes_labels[c], transform=a.transAxes,
               color='#aaaaaa', fontsize=8, va='top')

    # ── Row 2: CNN probability map ────────────────────────────────────────────
    for c in range(3):
        a = ax(2, c)
        a.imshow(prob_s[c], cmap=PROB_CMAP, vmin=0, vmax=1)
        if c == 0:
            a.set_title("CNN Probability Map", color='white',
                        fontsize=10, pad=4)
        a.text(0.02, 0.96, axes_labels[c], transform=a.transAxes,
               color='#aaaaaa', fontsize=8, va='top')

    # ── Row 3: Predicted mask ─────────────────────────────────────────────────
    for c in range(3):
        a = ax(3, c)
        # Show flair as dim background so brain anatomy is visible
        a.imshow(flair_s[c], cmap='gray', vmin=0, vmax=1, alpha=0.3)
        # Paint only tumour voxels in cyan
        if pred_s[c].any():
            masked = np.ma.masked_where(
                pred_s[c] == 0, pred_s[c].astype(float))
            cyan_cmap = LinearSegmentedColormap.from_list(
                "cyan", [(0, 1, 1, 0.95), (0, 1, 1, 0.95)], N=2)
            a.imshow(masked, cmap=cyan_cmap, vmin=0, vmax=1)
        if c == 0:
            a.set_title("Predicted Mask (Graph Cut)", color='white',
                        fontsize=10, pad=4)
        a.text(0.02, 0.96, axes_labels[c], transform=a.transAxes,
               color='#aaaaaa', fontsize=8, va='top')

    # ── Row 4: Ground truth ───────────────────────────────────────────────────
    if gt_s is not None:
        for c in range(3):
            a = ax(4, c)
            a.imshow(flair_s[c], cmap='gray', vmin=0, vmax=1, alpha=0.3)
            if gt_s[c].any():
                masked_gt = np.ma.masked_where(
                    gt_s[c] == 0, gt_s[c].astype(float))
                purple_cmap = LinearSegmentedColormap.from_list(
                    "purple", [(0.64, 0.35, 1.0, 0.9),
                               (0.64, 0.35, 1.0, 0.9)], N=2)
                a.imshow(masked_gt, cmap=purple_cmap, vmin=0, vmax=1)
            if c == 0:
                a.set_title("Ground-Truth Mask", color='white',
                            fontsize=10, pad=4)
            a.text(0.02, 0.96, axes_labels[c], transform=a.transAxes,
                   color='#aaaaaa', fontsize=8, va='top')

    # ── Colourbar ─────────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.45, 0.01, 0.15])
    sm = plt.cm.ScalarMappable(cmap=PROB_CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("P(tumour)", color='white', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white', fontsize=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor='#ff4c6e', alpha=0.85, label='Tumour overlay'),
        Patch(facecolor='#ff4c6e', fill=False,
              edgecolor='#ff4c6e', label='Predicted contour'),
        Patch(facecolor='#00e5ff', alpha=0.85, label='Predicted mask'),
    ]
    if gt_mask is not None:
        legend_elements.append(
            Patch(facecolor='#a259ff', alpha=0.85, label='Ground truth'))
    fig.legend(handles=legend_elements, loc='lower left',
               bbox_to_anchor=(0.01, 0.01), ncol=2,
               framealpha=0.2, labelcolor='white', fontsize=8)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Visualisation saved → {save_path}")
    if show:
        plt.show()


def plot_training_history(history_path: str, save_path: str):
    history    = np.load(history_path, allow_pickle=True).item()
    train_loss = history.get('train_loss', [])
    val_dice   = history.get('val_dice',   [])
    epochs     = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#0a0a0a')
    for a in axes:
        a.set_facecolor('#111827')
        a.tick_params(colors='white')
        for spine in a.spines.values():
            spine.set_edgecolor('#334155')

    axes[0].plot(epochs, train_loss, color='#00e5ff',
                 linewidth=2, label='Train Loss')
    axes[0].set_title('Training Loss (Dice+BCE)', color='white')
    axes[0].set_xlabel('Epoch', color='white')
    axes[0].set_ylabel('Loss',  color='white')
    axes[0].legend(facecolor='#1e293b', labelcolor='white')

    axes[1].plot(epochs, val_dice, color='#a259ff',
                 linewidth=2, label='Val Dice')
    axes[1].set_title('Validation Dice Score', color='white')
    axes[1].set_xlabel('Epoch', color='white')
    axes[1].set_ylabel('Dice',  color='white')
    axes[1].set_ylim(0, 1)
    axes[1].legend(facecolor='#1e293b', labelcolor='white')

    fig.suptitle('Training History', color='white', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Training history plot saved → {save_path}")


def quick_view(nifti_path: str, mask_path: str | None = None):
    import nibabel as nib
    vol  = np.asarray(nib.load(nifti_path).dataobj, dtype=np.float32)
    mask = None
    if mask_path:
        mask = np.asarray(nib.load(mask_path).dataobj, dtype=np.float32)

    vol_n = _normalise_for_display(vol)
    mid_d, mid_h, mid_w = [s // 2 for s in vol.shape]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='black')
    views = [
        (np.rot90(vol_n[mid_d, :, :]), "Axial"),
        (np.rot90(vol_n[:, mid_h, :]), "Coronal"),
        (np.rot90(vol_n[:, :, mid_w]), "Sagittal"),
    ]
    mask_views = None
    if mask is not None:
        mask_views = [
            np.rot90(mask[mid_d, :, :]),
            np.rot90(mask[:, mid_h, :]),
            np.rot90(mask[:, :, mid_w]),
        ]

    for a, (img, lbl) in zip(axes, views):
        a.imshow(img, cmap='gray')
        a.axis('off')
        a.set_title(lbl, color='white')
        a.set_facecolor('black')

    if mask_views:
        for a, mv in zip(axes, mask_views):
            a.imshow(mv, cmap=TUMOR_CMAP, alpha=0.6, vmin=0, vmax=1)

    fig.suptitle(os.path.basename(nifti_path), color='white')
    plt.tight_layout()
    plt.show()
