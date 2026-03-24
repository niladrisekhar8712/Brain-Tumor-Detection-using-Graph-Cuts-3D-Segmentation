# =============================================================================
#  predict.py  —  Run the full pipeline on one or more NIfTI MRI subjects.
#
#  Usage (single subject):
#      python predict.py \
#          --flair  path/to/subject_flair.nii.gz \
#          --t1     path/to/subject_t1.nii.gz    \
#          --t1ce   path/to/subject_t1ce.nii.gz  \
#          --t2     path/to/subject_t2.nii.gz    \
#          [--seg   path/to/subject_seg.nii.gz]  # optional ground-truth
#
#  Usage (all test subjects discovered from BraTS root):
#      python predict.py --all_test
# =============================================================================

import os, argparse, time
import numpy as np
import torch

import config as cfg
from utils.preprocessing import (preprocess_subject, extract_patches,
                                  stitch_patches, save_nifti, load_nifti)
from models.unet3d       import UNet3D, dice_score, hausdorff_distance_95
from models.graph_cut    import refine_with_graph_cut
from utils.visualise     import visualise_results


# ── sliding-window CNN inference ─────────────────────────────────────────────

def predict_cnn(model, volume4d: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Slide a patch window over the full volume, gather sigmoid probabilities,
    and stitch them back into a (D, H, W) probability map.
    """
    model.eval()
    patches_info = extract_patches(volume4d, cfg.PATCH_SIZE, cfg.PATCH_STRIDE)
    prob_patches = []
    D, H, W = volume4d.shape[1:]

    with torch.no_grad():
        for patch_np, coords in patches_info:
            patch_t = torch.from_numpy(patch_np).unsqueeze(0).to(device)
            prob    = torch.sigmoid(model(patch_t))
            prob_patches.append((prob.squeeze().cpu().numpy(), coords))

    return stitch_patches(prob_patches, (D, H, W), cfg.PATCH_SIZE)


# ── per-subject runner ────────────────────────────────────────────────────────

def run_subject(paths: dict, model, device: torch.device,
                subject_name: str = "subject") -> dict:
    """
    Full pipeline for one subject.  Returns a dict of result metrics.
    """
    print(f"\n{'─'*60}")
    print(f"  Subject: {subject_name}")
    print(f"{'─'*60}")

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    print("[1/4] Preprocessing …")
    t0 = time.time()
    volume4d, gt_label, affine, bbox = preprocess_subject(paths)
    print(f"      Volume shape: {volume4d.shape}  ({time.time()-t0:.1f}s)")

    # ── Step 2: CNN probability map ───────────────────────────────────────────
    print("[2/4] Running 3D U-Net (sliding window) …")
    t0 = time.time()
    prob_map = predict_cnn(model, volume4d, device)
    print(f"      Prob map range: [{prob_map.min():.3f}, {prob_map.max():.3f}]  "
          f"({time.time()-t0:.1f}s)")

    # ── Step 3: Supervoxel-based graph cut ────────────────────────────────────
    print("[3/4] Supervoxel-based Graph Cut …")
    t0 = time.time()
    flair_vol = volume4d[0]   # channel 0 = FLAIR (after preprocessing)
    final_mask = refine_with_graph_cut(
        prob_map   = prob_map,
        flair_vol  = flair_vol,
        n_supervoxels = cfg.N_SUPERVOXELS,
        compactness   = cfg.SLIC_COMPACTNESS,
        lambda_       = cfg.GC_LAMBDA,
        sigma         = cfg.GC_SIGMA,
    )
    print(f"      Tumour voxels: {final_mask.sum():,}  ({time.time()-t0:.1f}s)")

    # ── Step 4: Metrics & output ──────────────────────────────────────────────
    print("[4/4] Computing metrics and saving outputs …")
    results = {"subject": subject_name}

    pred_t = torch.from_numpy(final_mask).unsqueeze(0).unsqueeze(0).float()
    cnn_t  = torch.from_numpy((prob_map > 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0)

    if gt_label is not None:
        gt_t   = torch.from_numpy(gt_label).unsqueeze(0).unsqueeze(0).float()
        dice_gc  = dice_score(pred_t, gt_t)
        dice_cnn = dice_score(cnn_t,  gt_t)
        hd95     = hausdorff_distance_95(final_mask, gt_label)

        results.update({
            "dice_cnn_only":  dice_cnn,
            "dice_graphcut":  dice_gc,
            "hd95_graphcut":  hd95,
        })

        print(f"\n  ┌─────────────────────────────────┐")
        print(f"  │         RESULTS SUMMARY          │")
        print(f"  ├─────────────────────────────────┤")
        print(f"  │  Dice (CNN only)  : {dice_cnn:.4f}        │")
        print(f"  │  Dice (GraphCut)  : {dice_gc:.4f}        │")
        print(f"  │  HD95 (voxels)    : {hd95:.2f}         │")
        print(f"  └─────────────────────────────────┘")
    else:
        print("  (No ground-truth provided — skipping metrics)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = os.path.join(cfg.OUTPUT_DIR, subject_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save segmentation mask as NIfTI
    mask_path = os.path.join(out_dir, "segmentation_mask.nii.gz")
    save_nifti(final_mask.astype(np.float32), affine, mask_path)
    print(f"\n  Saved mask → {mask_path}")

    # Save probability map
    prob_path = os.path.join(out_dir, "cnn_probability_map.nii.gz")
    save_nifti(prob_map, affine, prob_path)
    print(f"  Saved prob → {prob_path}")

    # Visualisation
    vis_path = os.path.join(out_dir, "visualisation.png")
    visualise_results(
        volume4d  = volume4d,
        prob_map  = prob_map,
        pred_mask = final_mask,
        gt_mask   = gt_label,
        save_path = vis_path,
        title     = subject_name,
    )
    print(f"  Saved vis  → {vis_path}")

    results["output_dir"] = out_dir
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3D Brain Tumor Segmentation — Inference")
    parser.add_argument("--flair",    type=str, help="Path to FLAIR .nii.gz")
    parser.add_argument("--t1",       type=str, help="Path to T1 .nii.gz")
    parser.add_argument("--t1ce",     type=str, help="Path to T1ce .nii.gz")
    parser.add_argument("--t2",       type=str, help="Path to T2 .nii.gz")
    parser.add_argument("--seg",      type=str, default=None,
                        help="Path to ground-truth segmentation (optional)")
    parser.add_argument("--all_test", action='store_true',
                        help="Run on all test subjects saved during training")
    parser.add_argument("--model",    type=str,
                        default=os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth"),
                        help="Path to trained model checkpoint")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.model):
        print(f"\n[ERROR] Model not found: {args.model}")
        print("  → Run train.py first to train the model.")
        return

    model = UNet3D(
        in_channels   = cfg.IN_CHANNELS,
        out_channels  = cfg.OUT_CHANNELS,
        base_filters  = cfg.BASE_FILTERS,
        depth         = cfg.ENCODER_DEPTHS,
        use_attention = True,
    ).to(device)

    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded model from {args.model}  (epoch {ckpt.get('epoch', '?')})")

    # ── Run inference ─────────────────────────────────────────────────────────
    all_results = []

    if args.all_test:
        test_path = os.path.join(cfg.CHECKPOINT_DIR, "test_subjects.npy")
        if not os.path.exists(test_path):
            print("[ERROR] test_subjects.npy not found. Run train.py first.")
            return
        test_subs = np.load(test_path, allow_pickle=True).tolist()
        print(f"\nRunning on {len(test_subs)} test subjects …")
        for paths in test_subs:
            name = os.path.basename(os.path.dirname(paths['flair']))
            results = run_subject(paths, model, device, subject_name=name)
            all_results.append(results)

    elif args.flair:
        paths = {'flair': args.flair, 't1': args.t1,
                 't1ce': args.t1ce,   't2': args.t2}
        if args.seg:
            paths['seg'] = args.seg
        # Derive subject name from path
        name = os.path.basename(os.path.dirname(args.flair)) or "prediction"
        results = run_subject(paths, model, device, subject_name=name)
        all_results.append(results)

    else:
        parser.print_help()
        print("\n  Example:\n"
              "  python predict.py \\\n"
              "      --flair data/BraTS21_001/BraTS21_001_flair.nii.gz \\\n"
              "      --t1    data/BraTS21_001/BraTS21_001_t1.nii.gz    \\\n"
              "      --t1ce  data/BraTS21_001/BraTS21_001_t1ce.nii.gz  \\\n"
              "      --t2    data/BraTS21_001/BraTS21_001_t2.nii.gz    \\\n"
              "      --seg   data/BraTS21_001/BraTS21_001_seg.nii.gz")
        return

    # ── Summary across all subjects ───────────────────────────────────────────
    if len(all_results) > 1:
        dices = [r['dice_graphcut'] for r in all_results if 'dice_graphcut' in r]
        if dices:
            print(f"\n{'='*60}")
            print(f"  SUMMARY: {len(dices)} subjects")
            print(f"  Mean Dice (GraphCut): {np.mean(dices):.4f} ± {np.std(dices):.4f}")
            print(f"  Min: {np.min(dices):.4f}  Max: {np.max(dices):.4f}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
