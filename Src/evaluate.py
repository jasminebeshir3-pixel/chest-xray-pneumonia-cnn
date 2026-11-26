"""Metrics and plots for evaluating any model (baselines now, CNNs later).

Owner: Evaluation and Deployment Engineer.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, roc_auc_score, roc_curve)

CLASSES = ["NORMAL", "PNEUMONIA"]


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Every number we report for one model on one split. y_prob is the predicted probability of PNEUMONIA."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_pneumonia": report["PNEUMONIA"]["recall"],
        "precision_pneumonia": report["PNEUMONIA"]["precision"],
        "f1_pneumonia": report["PNEUMONIA"]["f1-score"],
        "recall_normal": report["NORMAL"]["recall"],
        "precision_normal": report["NORMAL"]["precision"],
        "f1_normal": report["NORMAL"]["f1-score"],
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def plot_confusion(y_true, y_prob, title, ax, threshold=0.5):
    """Draw the confusion matrix (rows = true class, columns = predicted class) on the given axis."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], display_labels=CLASSES,
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(title)


def plot_roc(curves, ax):
    """Draw ROC curves. curves is a dict: model name -> (y_true, y_prob)."""
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random guess")
    ax.set_xlabel("False positive rate (1 - recall NORMAL)")
    ax.set_ylabel("True positive rate (recall PNEUMONIA)")
    ax.legend(loc="lower right")


if __name__ == "__main__":
    # quick self-check with a tiny hand-made example
    m = compute_metrics([0, 0, 1, 1], [0.2, 0.6, 0.7, 0.9])
    assert m["recall_pneumonia"] == 1.0 and m["recall_normal"] == 0.5 and m["roc_auc"] == 1.0
    print("evaluate.py self-check passed:", {k: round(v, 3) for k, v in m.items()})
    plt.close("all")
