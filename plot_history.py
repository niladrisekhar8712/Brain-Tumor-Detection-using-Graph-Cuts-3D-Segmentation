# =============================================================================
#  plot_history.py  —  Plot training curves after training finishes.
#
#  Usage:
#      python plot_history.py
# =============================================================================

import os
import config as cfg
from utils.visualise import plot_training_history

history_path = os.path.join(cfg.CHECKPOINT_DIR, "history.npy")
save_path    = os.path.join(cfg.OUTPUT_DIR,      "training_history.png")

if not os.path.exists(history_path):
    print(f"[ERROR] {history_path} not found. Run train.py first.")
else:
    plot_training_history(history_path, save_path)
    print("Done. Open outputs/training_history.png to view the curves.")
