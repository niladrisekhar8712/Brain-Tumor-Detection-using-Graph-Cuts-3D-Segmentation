# =============================================================================
#  train.py  —  Run this file to train the 3D U-Net on your BraTS dataset.
# =============================================================================

import os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config as cfg
from utils.dataset       import discover_subjects, split_subjects, BraTSDataset
from models.unet3d       import UNet3D, DiceBCELoss, dice_score
from utils.preprocessing import extract_patches, stitch_patches


# ── helpers ───────────────────────────────────────────────────────────────────

def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [to_device(i, device) for i in x]
    return x.to(device, non_blocking=True)


def validate(model, val_dataset, device, patch_size, stride):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            volume4d, label = val_dataset[idx]
            volume_np = volume4d.numpy()
            label_np  = label.numpy()[0]          # (D, H, W)

            # Full sliding-window inference — same as predict.py
            patches_info = extract_patches(volume_np, patch_size, stride)
            prob_patches  = []
            D, H, W = volume_np.shape[1:]

            for patch_np, coords in patches_info:
                patch_t = torch.from_numpy(patch_np).unsqueeze(0).to(device)
                prob    = torch.sigmoid(model(patch_t))
                prob_patches.append(
                    (prob.squeeze().cpu().numpy(), coords))

            prob_map = stitch_patches(prob_patches, (D, H, W), patch_size)

            pred_t  = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0)
            label_t = torch.from_numpy(label_np).unsqueeze(0).unsqueeze(0)
            dice_scores.append(dice_score(pred_t, label_t))

    return float(np.mean(dice_scores))


# ── main training loop ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  3D Brain Tumor Segmentation — Training")
    print("=" * 60)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ── Data ─────────────────────────────────────────────────────────────────
    subjects = discover_subjects(cfg.BRATS_ROOT)
    if len(subjects) == 0:
        print("\n[ERROR] No BraTS subjects found!")
        print(f"  Check that BRATS_ROOT in config.py points to: {cfg.BRATS_ROOT}")
        return

    train_subs, val_subs, test_subs = split_subjects(
        subjects, cfg.VAL_FRACTION, cfg.TEST_FRACTION, cfg.RANDOM_SEED)

    train_ds = BraTSDataset(train_subs, mode='train',
                             patch_size=cfg.PATCH_SIZE, patches_per_volume=2)
    val_ds   = BraTSDataset(val_subs,   mode='val',
                             patch_size=cfg.PATCH_SIZE)

    np.save(os.path.join(cfg.CHECKPOINT_DIR, "test_subjects.npy"),
            np.array(test_subs, dtype=object), allow_pickle=True)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.NUM_WORKERS,
                              pin_memory=cfg.PIN_MEMORY)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels   = cfg.IN_CHANNELS,
        out_channels  = cfg.OUT_CHANNELS,
        base_filters  = cfg.BASE_FILTERS,
        depth         = cfg.ENCODER_DEPTHS,
        use_attention = True,
        dropout       = 0.3,
    ).to(device)

    print(f"\n  Parameters: {model.count_parameters():,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=cfg.LR_PATIENCE,
                                   factor=0.5)
    criterion = DiceBCELoss(dice_weight=0.6, bce_weight=0.4)

    # ── Resume if checkpoint exists ───────────────────────────────────────────
    best_ckpt   = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
    last_ckpt   = os.path.join(cfg.CHECKPOINT_DIR, "last_model.pth")
    start_epoch = 1
    best_dice   = 0.0
    no_improve  = 0

    if os.path.exists(last_ckpt):
        print(f"\n  Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_dice   = ckpt.get('best_dice', 0.0)
        no_improve  = ckpt.get('no_improve', 0)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  Starting training from epoch {start_epoch} …\n")
    history = {'train_loss': [], 'val_dice': []}

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (patches, labels) in enumerate(train_loader):
            patches = to_device(patches, device)
            labels  = to_device(labels,  device)

            optimizer.zero_grad()
            logits = model(patches)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch [{epoch}/{cfg.NUM_EPOCHS}]  "
                      f"Step [{batch_idx+1}/{len(train_loader)}]  "
                      f"Loss: {loss.item():.4f}", end='\r')

        avg_loss = epoch_loss / max(len(train_loader), 1)
        history['train_loss'].append(avg_loss)

        val_dice = validate(model, val_ds, device, cfg.PATCH_SIZE, cfg.PATCH_STRIDE)
        history['val_dice'].append(val_dice)
        scheduler.step(val_dice)

        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch:3d}/{cfg.NUM_EPOCHS}  "
              f"| Train Loss: {avg_loss:.4f}  "
              f"| Val Dice: {val_dice:.4f}  "
              f"| LR: {optimizer.param_groups[0]['lr']:.2e}  "
              f"| Time: {elapsed:.1f}s")

        state = {
            'epoch':      epoch,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'best_dice':  best_dice,
            'no_improve': no_improve,
        }
        torch.save(state, last_ckpt)

        if val_dice > best_dice:
            best_dice  = val_dice
            no_improve = 0
            torch.save(state, best_ckpt)
            print(f"  ✓ New best model saved  (Dice = {best_dice:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{cfg.EARLY_STOP} epochs")

        if no_improve >= cfg.EARLY_STOP:
            print(f"\n  Early stopping triggered after epoch {epoch}.")
            break

    np.save(os.path.join(cfg.CHECKPOINT_DIR, "history.npy"),
            history, allow_pickle=True)

    print(f"\n  Training complete. Best Val Dice = {best_dice:.4f}")
    print(f"  Best model: {best_ckpt}")
    print(f"\n  Next step → run: python predict.py")


if __name__ == "__main__":
    main()