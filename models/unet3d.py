# =============================================================================
#  models/unet3d.py
#  Smaller 3D U-Net — better suited for limited training data.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout3d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            pd = skip.size(2) - x.size(2)
            ph = skip.size(3) - x.size(3)
            pw = skip.size(4) - x.size(4)
            x = F.pad(x, [0, pw, 0, ph, 0, pd])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg  = nn.Conv3d(g_ch,    inter_ch, kernel_size=1, bias=False)
        self.Wx  = nn.Conv3d(x_ch,    inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:],
                               mode='trilinear', align_corners=False)
        alpha = self.psi(self.relu(g1 + x1))
        return x * alpha


class UNet3D(nn.Module):
    def __init__(self,
                 in_channels:   int   = 4,
                 out_channels:  int   = 1,
                 base_filters:  int   = 16,
                 depth:         int   = 3,
                 use_attention: bool  = True,
                 dropout:       float = 0.3):
        super().__init__()
        self.depth = depth
        f = base_filters

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = f * (2 ** i)
            self.encoders.append(
                EncoderBlock(in_ch, out_ch, dropout=dropout if i >= 1 else 0.0))
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = f * (2 ** depth)
        self.bottleneck = ConvBlock(in_ch, bottleneck_ch, dropout=dropout)

        # Decoder
        self.decoders   = nn.ModuleList()
        self.attn_gates = nn.ModuleList() if use_attention else None
        in_ch = bottleneck_ch
        for i in reversed(range(depth)):
            skip_ch = f * (2 ** i)
            out_ch  = skip_ch
            if use_attention:
                self.attn_gates.append(
                    AttentionGate(in_ch, skip_ch, max(skip_ch // 2, 1)))
            self.decoders.append(
                DecoderBlock(in_ch, skip_ch, out_ch,
                             dropout=dropout if i >= 1 else 0.0))
            in_ch = out_ch

        self.head = nn.Conv3d(in_ch, out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            if self.attn_gates is not None:
                skip = self.attn_gates[i](x, skip)
            x = dec(x, skip)
        return self.head(x)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num   = 2.0 * (probs * targets).sum()
        den   = probs.sum() + targets.sum() + self.smooth
        return 1.0 - num / den


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dw   = dice_weight
        self.bw   = bce_weight

    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + \
               self.bw * self.bce(logits, targets)


def dice_score(pred_mask, true_mask, threshold=0.5, smooth=1e-6):
    pred  = (pred_mask > threshold).float()
    inter = (pred * true_mask).sum()
    return float((2.0 * inter + smooth) /
                 (pred.sum() + true_mask.sum() + smooth))


def hausdorff_distance_95(pred, gt):
    from scipy.ndimage import distance_transform_edt
    pred_surf = pred ^ (pred & np.roll(pred, 1, axis=0))
    gt_surf   = gt   ^ (gt   & np.roll(gt,   1, axis=0))
    if pred_surf.sum() == 0 or gt_surf.sum() == 0:
        return float('nan')
    dt_pred = distance_transform_edt(~pred.astype(bool))
    dt_gt   = distance_transform_edt(~gt.astype(bool))
    d1 = dt_gt[pred_surf.astype(bool)]
    d2 = dt_pred[gt_surf.astype(bool)]
    return float(np.percentile(np.concatenate([d1, d2]), 95))

