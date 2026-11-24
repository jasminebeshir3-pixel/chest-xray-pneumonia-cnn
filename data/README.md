# Data

The raw images are not stored in git (about 1.2 GB). Only the split file is.

## Source

Chest X-Ray Images (Pneumonia) on Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Original data: Kermany, Zhang and Goldbaum, "Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images", Mendeley Data, Version 3, doi:10.17632/rscbjbr9sj.3, licence CC BY 4.0. The X-rays come from pediatric patients (1 to 5 years old) at Guangzhou Women and Children's Medical Center.

## How to get it

On Kaggle: add the dataset "chest-xray-pneumonia" to the notebook. The code finds it under `/kaggle/input` by itself.

Locally or on Colab:

```
python data/download_data.py
```

then set the `CXR_DATA_DIR` variable to the folder it prints (the one that has `train/`, `val/` and `test/` in it).

## Files

| File | What it is |
|------|------------|
| `download_data.py` | downloads the dataset with kagglehub and prints its path |
| `split.csv` | our train / val / test split, one row per image (made by `python -m src.data`) |

`split.csv` columns: `path` (relative to the chest_xray folder), `label` (0 = NORMAL, 1 = PNEUMONIA), `class_name`, `official_split` (the Kaggle folder the image came from), `patient`, `subtype` (bacteria / virus / normal), `md5` (file hash) and `split` (our split).

## How the split is made

1. Read every image in `train/`, `val/` and `test/` (the extra `chest_xray/chest_xray` copy and `__MACOSX` in the zip are ignored).
2. Remove exact duplicate files using the MD5 hash.
3. Take the patient id from the file name (`person1_bacteria_1.jpeg` is patient `person1`, `IM-0115-0001.jpeg` and `IM-0115-0001-0002.jpeg` are patient `IM-0115`).
4. Split 70 / 15 / 15 with `StratifiedGroupKFold`: stratified by label, and all images of one patient go to the same split.
5. `check_no_leakage` stops with an error if a patient or a file is in two splits or the class ratio is off by more than 3 points.

We don't use the official split because its validation folder has only 16 images, which is too small to pick models with.
