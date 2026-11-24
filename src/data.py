"""Data loading, cleaning, splitting and the tf.data input pipeline.

Owner: Data Engineer.

Run `python -m src.data` from the repo root to rebuild data/split.csv.
"""
import argparse
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["NORMAL", "PNEUMONIA"]  # label 0 = NORMAL, label 1 = PNEUMONIA
IMG_EXTS = {".jpeg", ".jpg", ".png"}
KAGGLE_INPUT_DIR = "/kaggle/input/chest-xray-pneumonia/chest_xray"


def find_data_dir(data_dir=None):
    """Return the chest_xray folder: the argument, else the CXR_DATA_DIR variable, else the Kaggle input folder."""
    for candidate in [data_dir, os.environ.get("CXR_DATA_DIR"), KAGGLE_INPUT_DIR]:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise FileNotFoundError(
        "Dataset not found. Run `python data/download_data.py` and set CXR_DATA_DIR to the chest_xray folder."
    )


def parse_patient(filename):
    """Get the patient id from a file name, so all images of one patient end up in the same split.

    PNEUMONIA files look like person1_bacteria_1.jpeg, NORMAL files like IM-0115-0001.jpeg,
    IM-0429-0001-0002.jpeg or NORMAL2-IM-1427-0001.jpeg. The numbers after the IM id are
    image numbers of the same patient.
    """
    stem = Path(filename).stem
    match = re.match(r"(person\d+)_", stem) or re.match(r"((?:NORMAL2-)?IM-\d+)", stem)
    if match:
        return match.group(1)
    return stem  # unknown pattern: the image is its own group


def parse_subtype(filename):
    """Return bacteria, virus or normal from the file name (only used for EDA and error analysis)."""
    name = filename.lower()
    if "bacteria" in name:
        return "bacteria"
    if "virus" in name:
        return "virus"
    return "normal"


def build_index(data_dir=None):
    """Make one table with a row per image: path, label, official folder, patient, subtype and file hash.

    We only read train/, val/ and test/ at the top level. The Kaggle zip also has a second
    copy under chest_xray/chest_xray and a __MACOSX folder, reading those would double count.
    """
    root = find_data_dir(data_dir)
    rows = []
    for official in ["train", "val", "test"]:
        for label, cls in enumerate(CLASSES):
            for f in sorted((root / official / cls).iterdir()):
                if f.suffix.lower() not in IMG_EXTS:
                    continue
                rows.append({
                    "path": f.relative_to(root).as_posix(),
                    "label": label,
                    "class_name": cls,
                    "official_split": official,
                    "patient": parse_patient(f.name),
                    "subtype": parse_subtype(f.name),
                    "md5": hashlib.md5(f.read_bytes()).hexdigest(),
                })
    return pd.DataFrame(rows)


def remove_duplicates(df):
    """Drop exact duplicate files (same MD5 hash) and keep the first copy.

    If the same file has two different labels we drop every copy, because we
    can't know which label is right.
    """
    conflict = df.groupby("md5")["label"].transform("nunique") > 1
    clean = df[~conflict].drop_duplicates("md5", keep="first").reset_index(drop=True)
    print(f"removed {len(df) - len(clean)} duplicate images ({int(conflict.sum())} had conflicting labels)")
    return clean


def check_no_leakage(df, max_ratio_gap=0.03):
    """Raise an error if a patient or an identical file is in more than one split, or the class ratio drifts."""
    for col in ["patient", "md5"]:
        splits_per_value = df.groupby(col)["split"].nunique()
        n_bad = int((splits_per_value > 1).sum())
        if n_bad:
            raise ValueError(f"leakage: {n_bad} values of '{col}' appear in more than one split")
    pneumonia_ratio = df.groupby("split")["label"].mean()
    if pneumonia_ratio.max() - pneumonia_ratio.min() > max_ratio_gap:
        raise ValueError(f"class ratio is too different between splits: {pneumonia_ratio.round(3).to_dict()}")


def make_split(df, val_size=0.15, test_size=0.15, seed=42):
    """Split into train/val/test, stratified by label and grouped by patient.

    The official val folder has only 16 images, so we pool all folders and split again.
    StratifiedGroupKFold with 20 folds gives chunks of about 5%, so 3 chunks = 15%.
    """
    n_folds = 20
    n_test = round(test_size * n_folds)
    n_val = round(val_size * n_folds)
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    split = np.empty(len(df), dtype=object)
    for i, (_, idx) in enumerate(folds.split(df, df["label"], groups=df["patient"])):
        split[idx] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    df = df.assign(split=split)
    check_no_leakage(df)
    return df


def load_split(csv_path="data/split.csv"):
    """Read the saved split and check it again for leakage before anyone uses it."""
    df = pd.read_csv(csv_path)
    check_no_leakage(df)
    return df


def class_weights(labels):
    """Weight n / (2 * n_c) per class, so both classes count the same in the loss (same as sklearn 'balanced')."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    return {c: float(len(labels) / (2 * counts[c])) for c in range(2)}


def load_image(path, img_size=224):
    """Read one image file as grayscale, copy it to 3 channels, resize to img_size x img_size and scale to [0, 1].

    All 283 RGB files in the dataset are PNEUMONIA, so if we kept their colour the model could
    learn "colour = pneumonia". Reading as grayscale removes that. 3 channels because VGG16 needs them.
    The same function is used for training, evaluation and predict.py, so the preprocessing
    is never written twice. Pixel scaling is a fixed /255, nothing is fitted, so nothing can leak.
    """
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.resize(img, [img_size, img_size])
    return img / 255.0


def augmentation_layers(seed=42):
    """Small changes that could happen in a real X-ray: rotation, zoom, shift and contrast.

    No horizontal flip on purpose: the heart is on the left, so a flipped X-ray is not a real X-ray.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.03, seed=seed),  # 0.03 of a full turn, about 10 degrees
        tf.keras.layers.RandomZoom(0.1, seed=seed),
        tf.keras.layers.RandomTranslation(0.05, 0.05, seed=seed),
        tf.keras.layers.RandomContrast(0.1, value_range=(0, 1), seed=seed),
    ], name="augmentation")


def make_dataset(df, data_dir=None, img_size=224, batch_size=32, shuffle=False, augment=False, seed=42):
    """Build a batched tf.data.Dataset of (image, label) from rows of the split table."""
    if augment and set(df["split"]) != {"train"}:
        raise ValueError("augmentation is only allowed on the training set")
    root = find_data_dir(data_dir)
    paths = [str(root / p) for p in df["path"]]
    ds = tf.data.Dataset.from_tensor_slices((paths, df["label"].to_numpy(dtype="float32")))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p, img_size), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment:
        aug = augmentation_layers(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_arrays(df, data_dir=None, img_size=64):
    """Load images as flat grayscale vectors for the sklearn baselines, using the same load_image as the CNNs."""
    ds = make_dataset(df, data_dir, img_size, batch_size=256)
    X = np.concatenate([x.numpy().mean(axis=-1).reshape(len(x), -1) for x, _ in ds])
    return X, df["label"].to_numpy(dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Index the dataset, remove duplicates and save the train/val/test split.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default="data/split.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = remove_duplicates(build_index(args.data_dir))
    df = make_split(df, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(pd.crosstab(df["split"], df["class_name"], margins=True))
    print("saved", args.out)


if __name__ == "__main__":
    main()
