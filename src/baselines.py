"""Train the two baselines, evaluate them on the validation set and add them to the experiment log.

Usage (from the repo root):
    python -m src.baselines

The test set is not touched here. It is used once, at the end of the project.
"""
import time

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.data import load_arrays, load_split
from src.evaluate import compute_metrics, plot_confusion, plot_roc
from src.models import logistic_baseline, majority_baseline
from src.train import log_run, set_seed


def main(config_path="configs/base.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    df = load_split(cfg["split_csv"])
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    X_train, y_train = load_arrays(train, cfg["data_dir"], cfg["baseline_img_size"])
    X_val, y_val = load_arrays(val, cfg["data_dir"], cfg["baseline_img_size"])
    n_pixels = X_train.shape[1]

    # B1b is the same model as B1 without class weights, to see what the weights do to per-class recall
    runs = {
        "B0_majority": (majority_baseline(), "always predict the majority class", {}, 0),
        "B1_logreg": (logistic_baseline(cfg["logreg_C"], "balanced", cfg["seed"]),
                      "logistic regression, 64x64 pixels, balanced class weights",
                      {"C": cfg["logreg_C"], "class_weight": "balanced", "img_size": cfg["baseline_img_size"]},
                      n_pixels + 1),
        "B1b_logreg_noweights": (logistic_baseline(cfg["logreg_C"], None, cfg["seed"]),
                                 "logistic regression, 64x64 pixels, no class weights",
                                 {"C": cfg["logreg_C"], "class_weight": None, "img_size": cfg["baseline_img_size"]},
                                 n_pixels + 1),
    }

    results, val_probs = {}, {}
    for run_id, (model, model_name, hyperparams, n_params) in runs.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        train_m = compute_metrics(y_train, model.predict_proba(X_train)[:, 1])
        val_probs[run_id] = model.predict_proba(X_val)[:, 1]
        val_m = compute_metrics(y_val, val_probs[run_id])
        results[run_id] = val_m
        log_run({
            "run_id": run_id, "author": "tech_lead", "model": model_name, "hyperparams": hyperparams,
            "seed": cfg["seed"], "train_time_s": train_time, "params": n_params,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
            "notes": "Phase 2 baseline",
        })

    table = pd.DataFrame(results).T.round(3)
    print(table[["accuracy", "recall_pneumonia", "recall_normal", "precision_pneumonia", "macro_f1", "roc_auc"]])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, run_id in zip(axes, runs):
        plot_confusion(y_val, val_probs[run_id], f"{run_id} (validation)", ax)
    fig.tight_layout()
    fig.savefig("experiments/baselines_confusion.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_roc({run_id: (y_val, p) for run_id, p in val_probs.items()}, ax)
    ax.set_title("Baselines, ROC on the validation set")
    fig.tight_layout()
    fig.savefig("experiments/baselines_roc.png", dpi=120)
    print("saved experiments/runs.csv and the baseline plots")


if __name__ == "__main__":
    main()
