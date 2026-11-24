"""Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle with kagglehub.

Usage (from the repo root):
    python data/download_data.py

Not needed on Kaggle: add the dataset to the notebook and it is already under /kaggle/input.
"""
from pathlib import Path

import kagglehub

path = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")) / "chest_xray"
print("Dataset is in:", path)
print("Set this environment variable before running the code:")
print(f'  Windows PowerShell:  $env:CXR_DATA_DIR = "{path}"')
print(f'  Linux / Colab:       export CXR_DATA_DIR="{path}"')
