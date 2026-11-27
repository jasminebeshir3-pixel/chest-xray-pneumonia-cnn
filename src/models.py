"""Baselines and neural network architectures.

Owner: Model Architect. Contains the trivial and learned baselines;
the CNN models (A, B, C) are added in Phase 3.
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def majority_baseline():
    """Trivial baseline: always predicts the most common training class (PNEUMONIA)."""
    return DummyClassifier(strategy="most_frequent")


def logistic_baseline(C=0.01, class_weight="balanced", seed=42):
    """Simple learned baseline: logistic regression on flattened 64x64 grayscale pixels.

    The StandardScaler is inside the pipeline, so it is fitted on the training images only.
    A small C (strong L2) because we have about as many pixels as training images.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight=class_weight, max_iter=3000, random_state=seed),
    )
