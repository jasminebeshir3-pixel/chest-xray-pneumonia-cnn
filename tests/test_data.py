"""Leakage and preprocessing checks for src/data.py. Run from the repo root: python -m pytest tests"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data import check_no_leakage, class_weights, load_image, load_split, parse_patient

SPLIT_CSV = Path(__file__).parents[1] / "data" / "split.csv"


def test_parse_patient():
    assert parse_patient("person1_bacteria_1.jpeg") == "person1"
    assert parse_patient("person1000_virus_1681.jpeg") == "person1000"
    assert parse_patient("person447_virus_921_1.jpeg") == "person447"
    assert parse_patient("IM-0115-0001.jpeg") == "IM-0115"
    assert parse_patient("IM-0429-0001-0002.jpeg") == "IM-0429"
    assert parse_patient("NORMAL2-IM-1427-0001.jpeg") == "NORMAL2-IM-1427"


def test_same_patient_in_two_splits_is_caught():
    df = pd.DataFrame({"patient": ["p1", "p1"], "md5": ["a", "b"], "label": [1, 1], "split": ["train", "test"]})
    with pytest.raises(ValueError):
        check_no_leakage(df)


def test_same_file_in_two_splits_is_caught():
    df = pd.DataFrame({"patient": ["p1", "p2"], "md5": ["a", "a"], "label": [1, 1], "split": ["train", "test"]})
    with pytest.raises(ValueError):
        check_no_leakage(df)


def test_saved_split_is_clean():
    df = load_split(SPLIT_CSV)  # load_split runs check_no_leakage
    assert set(df["split"]) == {"train", "val", "test"}
    assert df["md5"].is_unique


def test_load_image_removes_colour(tmp_path):
    # all RGB files are PNEUMONIA, so colour must not reach the model
    path = tmp_path / "red.jpeg"
    Image.new("RGB", (300, 200), (200, 30, 30)).save(path)
    x = load_image(str(path), img_size=64).numpy()
    assert x.shape == (64, 64, 3)
    assert 0 <= x.min() and x.max() <= 1
    assert np.allclose(x[..., 0], x[..., 1]) and np.allclose(x[..., 1], x[..., 2])


def test_class_weights_balance_the_classes():
    labels = np.array([0, 1, 1, 1])
    w = class_weights(labels)
    assert w[0] * 1 == pytest.approx(w[1] * 3)  # weight x count is equal for both classes
