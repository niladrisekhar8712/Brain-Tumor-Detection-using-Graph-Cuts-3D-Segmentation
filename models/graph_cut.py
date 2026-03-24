# =============================================================================
#  models/graph_cut.py
#  Stage 2 + 3 of the pipeline:
#    • SLIC 3D supervoxel generation
#    • Region Adjacency Graph construction
#    • Max-flow / min-cut energy minimisation (PyMaxflow)
# =============================================================================

import numpy as np
from skimage.segmentation import slic
from scipy.ndimage import label as ndimage_label

try:
    import maxflow
    MAXFLOW_AVAILABLE = True
except ImportError:
    MAXFLOW_AVAILABLE = False
    print("[GraphCut] WARNING: 'maxflow' not installed. "
          "Install with: pip install PyMaxflow\n"
          "Falling back to threshold-only segmentation.")


def generate_supervoxels(volume: np.ndarray,
                         n_supervoxels: int,
                         compactness: float) -> np.ndarray:
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume, dtype=np.int32)
    norm = (volume - vmin) / (vmax - vmin)
    labels = slic(norm,
                  n_segments=n_supervoxels,
                  compactness=compactness,
                  channel_axis=None,
                  enforce_connectivity=True,
                  start_label=0)
    return labels.astype(np.int32)


def aggregate_probabilities(prob_map: np.ndarray,
                             sv_labels: np.ndarray) -> dict:
    sv_ids   = np.unique(sv_labels)
    sv_probs = {}
    for sv_id in sv_ids:
        mask = (sv_labels == sv_id)
        sv_probs[sv_id] = float(prob_map[mask].mean())
    return sv_probs


def build_rag(volume: np.ndarray,
              sv_labels: np.ndarray,
              sv_probs: dict):
    sv_ids   = sorted(sv_probs.keys())
    sv_means = {}
    for sv_id in sv_ids:
        mask = (sv_labels == sv_id)
        sv_means[sv_id] = float(volume[mask].mean())

    adjacencies = set()
    for axis in range(3):
        slices_a = [slice(None)] * 3
        slices_b = [slice(None)] * 3
        slices_a[axis] = slice(0, -1)
        slices_b[axis] = slice(1, None)
        a = sv_labels[tuple(slices_a)]
        b = sv_labels[tuple(slices_b)]
        pairs = np.stack([a.ravel(), b.ravel()], axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        for p in pairs:
            key = (min(p[0], p[1]), max(p[0], p[1]))
            adjacencies.add(key)

    edges = []
    for (i, j) in adjacencies:
        diff = abs(sv_means[i] - sv_means[j])
        edges.append((i, j, diff))

    return sv_ids, edges, sv_means


def run_graph_cut(sv_ids: list,
                  edges: list,
                  sv_probs: dict,
                  sv_means: dict,
                  lambda_: float,
                  sigma: float) -> dict:
    if not MAXFLOW_AVAILABLE:
        return {sv: int(p >= 0.5) for sv, p in sv_probs.items()}

    n         = len(sv_ids)
    id_to_idx = {sv: i for i, sv in enumerate(sv_ids)}

    g     = maxflow.Graph[float](n, len(edges))
    nodes = g.add_nodes(n)

    eps = 1e-8
    for sv in sv_ids:
        idx     = id_to_idx[sv]
        p       = float(np.clip(sv_probs[sv], eps, 1 - eps))
        p_sharp = float(np.clip((p - 0.4) / 0.3, eps, 1 - eps))
        cap_source = -np.log(p_sharp)
        cap_sink   = -np.log(1.0 - p_sharp)
        g.add_tedge(nodes[idx], cap_source, cap_sink)

    for (i, j, _) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        diff  = sv_means[i] - sv_means[j]
        w     = lambda_ * np.exp(-(diff * 2) / (2 * sigma * 2))
        g.add_edge(nodes[idx_i], nodes[idx_j], w, w)

    g.maxflow()

    labels = {}
    for sv in sv_ids:
        idx = id_to_idx[sv]
        labels[sv] = 1 - g.get_segment(nodes[idx])

    return labels


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    labeled, num_features = ndimage_label(mask)
    if num_features == 0:
        return mask
    sizes   = [(labeled == i).sum() for i in range(1, num_features + 1)]
    largest = np.argmax(sizes) + 1
    return (labeled == largest).astype(np.uint8)


def refine_with_graph_cut(prob_map: np.ndarray,
                           flair_vol: np.ndarray,
                           n_supervoxels: int,
                           compactness: float,
                           lambda_: float,
                           sigma: float,
                           post_process: bool = True) -> np.ndarray:
    """
    Hybrid pipeline:
    1. CNN threshold at 0.5 → initial mask + largest component
    2. If ROI is reasonable size → run graph cut inside ROI only
    3. If graph cut result is bad → fall back to CNN threshold
    """
    print("  [GraphCut] Computing initial CNN threshold mask ...")
    cnn_mask = (prob_map >= 0.5).astype(np.uint8)

    if cnn_mask.sum() == 0:
        print("  [GraphCut] No tumour detected — returning empty mask.")
        return cnn_mask

    # Keep largest component of CNN mask
    cnn_mask = keep_largest_component(cnn_mask)
    if cnn_mask.sum() == 0:
        return cnn_mask

    # Get bounding box around CNN detection
    coords = np.argwhere(cnn_mask > 0)
    margin = 10
    D, H, W = prob_map.shape
    z0 = max(0, int(coords[:, 0].min()) - margin)
    z1 = min(D, int(coords[:, 0].max()) + margin)
    y0 = max(0, int(coords[:, 1].min()) - margin)
    y1 = min(H, int(coords[:, 1].max()) + margin)
    x0 = max(0, int(coords[:, 2].min()) - margin)
    x1 = min(W, int(coords[:, 2].max()) + margin)

    roi_size   = (z1 - z0) * (y1 - y0) * (x1 - x0)
    total_size = D * H * W

    # If ROI is more than 30% of brain, CNN is too uncertain → skip graph cut
    if roi_size > 0.30 * total_size:
        print("  [GraphCut] ROI too large — using CNN threshold directly.")
        return cnn_mask

    # Run graph cut only inside ROI
    prob_roi  = prob_map[z0:z1, y0:y1, x0:x1]
    flair_roi = flair_vol[z0:z1, y0:y1, x0:x1]

    print(f"  [GraphCut] ROI shape: {prob_roi.shape}")
    print("  [GraphCut] Generating supervoxels in ROI ...")
    sv_labels = generate_supervoxels(flair_roi, min(n_supervoxels, 300),
                                      compactness)
    print(f"  [GraphCut] {len(np.unique(sv_labels))} supervoxels generated")

    sv_probs = aggregate_probabilities(prob_roi, sv_labels)
    sv_ids, edges, sv_means = build_rag(flair_roi, sv_labels, sv_probs)

    print(f"  [GraphCut] Running min-cut on {len(sv_ids)} nodes, "
          f"{len(edges)} edges ...")
    sv_labels_cut = run_graph_cut(sv_ids, edges, sv_probs, sv_means,
                                   lambda_=lambda_, sigma=sigma)

    roi_mask = np.zeros_like(sv_labels, dtype=np.uint8)
    for sv, lab in sv_labels_cut.items():
        roi_mask[sv_labels == sv] = lab

    # Place result back into full volume
    final_mask = np.zeros_like(prob_map, dtype=np.uint8)
    final_mask[z0:z1, y0:y1, x0:x1] = roi_mask

    # If graph cut result is empty or 3x bigger than CNN → use CNN instead
    if final_mask.sum() == 0 or final_mask.sum() > cnn_mask.sum() * 3:
        print("  [GraphCut] Graph cut result unreliable — using CNN threshold.")
        return cnn_mask

    if post_process and final_mask.sum() > 0:
        print("  [GraphCut] Keeping largest component ...")
        final_mask = keep_largest_component(final_mask)

    return final_mask