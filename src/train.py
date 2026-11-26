"""Training helpers and the experiment log.

Owner: Experimentation Engineer. The Keras training loop is added in Phase 3.
"""
import csv
import json
from datetime import date
from pathlib import Path

import tensorflow as tf

METRICS = ["accuracy", "recall_pneumonia", "recall_normal", "precision_pneumonia", "macro_f1", "roc_auc"]
LOG_COLUMNS = (["run_id", "date", "author", "model", "hyperparams", "seed", "train_time_s", "params",
                "train_loss", "val_loss"]
               + [f"{split}_{m}" for split in ("train", "val") for m in METRICS]
               + ["notes"])


def set_seed(seed):
    """Fix the Python, NumPy and TensorFlow random seeds so a run can be repeated."""
    tf.keras.utils.set_random_seed(seed)


def log_run(row, log_path="experiments/runs.csv"):
    """Append one run to the experiment log CSV. The header is written the first time."""
    row = {"date": date.today().isoformat(), **row}
    if isinstance(row.get("hyperparams"), dict):
        row["hyperparams"] = json.dumps(row["hyperparams"])
    row = {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
    path = Path(log_path)
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
