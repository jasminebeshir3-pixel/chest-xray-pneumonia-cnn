# Chest X-Ray Pneumonia Screening (AIE231)

## Problem Statement

A radiology department needs a screening tool that flags chest X-rays likely to show pneumonia so that a radiologist reviews them first. The model takes a chest X-ray image and outputs the probability of pneumonia (binary classification: `NORMAL` vs. `PNEUMONIA`). Because a missed pneumonia case (false negative) is far more costly than a false alarm, recall on the `PNEUMONIA` class is the primary metric, supported by macro-F1 and ROC-AUC.

## Task Definition

| Item | Our choice | Why |
|------|------------|-----|
| Input | one X-ray, read as grayscale, copied to 3 channels, resized to 224 x 224, pixels scaled to [0, 1] | 224 x 224 is the VGG16 input size; VGG16 needs 3 channels; all 283 RGB files are PNEUMONIA, so colour is removed |
| Output | one sigmoid unit = P(pneumonia) | two classes, so one probability is enough |
| Loss | binary cross-entropy, with class weights | the standard loss for a sigmoid output; the weights handle the 2.7 : 1 imbalance |
| Primary metric | recall on PNEUMONIA | a missed pneumonia is the most costly error |
| Supporting metrics | macro-F1 and ROC-AUC | a model that says "pneumonia" for everything gets 100% recall, these metrics catch that |
| Threshold | 0.5 for now | may be lowered later to raise recall, chosen on the validation set only |

## Dataset

- **Name:** Chest X-Ray Images (Pneumonia) [3], derived from the data of Kermany et al. [1], [2].
- **Size:** 5,856 images found in the dataset folders (4,273 `PNEUMONIA` / 1,583 `NORMAL`). The Kaggle page states 5,863; we rely on our own count.
- **Why it fits the project criteria:** more than 5,000 images, class imbalance of about 2.7:1, images of very different sizes, 32 exact duplicate files, not a toy dataset, and a licence that permits academic use.
- **Getting the data:** see [data/README.md](data/README.md). The images are not stored in git.

## Model Family

CNN: a custom CNN built from scratch, then VGG16 transfer learning (frozen base) and fine-tuning (last block unfrozen).

## Setup and Run

```
pip install -r requirements.txt
python data/download_data.py      # then set CXR_DATA_DIR to the folder it prints
python -m src.data                # rebuilds data/split.csv (already in the repo, same result with seed 42)
python -m pytest tests            # leakage and preprocessing checks
jupyter notebook notebooks/01_eda.ipynb
python -m src.baselines           # trains both baselines and logs them in experiments/runs.csv
```

On Kaggle, add the dataset to the notebook; the code finds it under `/kaggle/input` by itself.

## Data Split

We pooled the official folders (the official validation folder has only 16 images), removed 32 exact duplicates and split 70 / 15 / 15 by patient, so no child has X-rays in two splits. `check_no_leakage` in `src/data.py` stops with an error if a patient or a file appears in two splits.

| Split | NORMAL | PNEUMONIA | Total | Patients |
|-------|-------:|----------:|------:|---------:|
| train | 1,105 | 2,972 | 4,077 | 2,185 |
| val | 237 | 636 | 873 | 468 |
| test | 237 | 637 | 874 | 464 |

## Results So Far (validation set)

The test set is not used until the final model is chosen.

| Model | Recall PNEUMONIA | Recall NORMAL | Macro-F1 | ROC-AUC |
|-------|-----------------:|--------------:|---------:|--------:|
| B0: majority class (always PNEUMONIA) | 1.000 | 0.000 | 0.421 | 0.500 |
| B1: logistic regression, 64x64 pixels, class weights | 0.948 | 0.911 | 0.923 | 0.979 |
| B1b: same without class weights | 0.964 | 0.848 | 0.913 | 0.979 |

Every CNN must beat B1. Full log with hyperparameters, seed and training time: `experiments/runs.csv`.

## Repository Structure

```
configs/       experiment config files (base.yaml has the shared settings)
data/          download script, data README and our split (no images in git)
notebooks/     01_eda.ipynb, later 02_results.ipynb
src/           data.py, models.py, train.py, evaluate.py, baselines.py
tests/         leakage and preprocessing checks
experiments/   experiment log (runs.csv) and plots
models/        final saved model
app/           demo application
report/        report PDF and slides
```

## Team

| Member | Role | Work package | Files so far |
|--------|------|--------------|--------------|
| Amir Emil, 24100316 | Tech Lead | problem formulation, repository, config, integration, EDA and baselines, final model with 3 seeds | `requirements.txt`, `configs/base.yaml`, `.gitignore`, `notebooks/01_eda.ipynb`, `src/baselines.py`, `experiments/` |
| Youssef Islam, 24100674 | Data Engineer | loading, cleaning, leak-free split, preprocessing, class imbalance, augmentation | `src/data.py`, `data/`, `tests/test_data.py` |
| Lara, 24100605 | Baseline and Analysis Engineer | trains Model A (custom CNN) and compares it with the baselines | starts in Phase 3 |
| Jasmine Beshir, 24100515 | Model Architect | architectures, hand parameter counts, regularization, GradientTape step | `src/models.py` |
| Sarah Ahmed, 24100558 | Experimentation Engineer | experiment log, tuning studies, learning curves | `src/train.py` |
| Maryam Gomaa, [22100578] | Evaluation and Deployment Engineer | metrics, error analysis, `predict.py`, demo app | `src/evaluate.py` |

## License

Kaggle lists "Other (specified in description)". The original Mendeley Data dataset [1] is released under CC BY 4.0, which permits academic use with attribution. This is an educational prototype for the AIE231 course, not a clinical diagnostic tool.

## References

[1] D. Kermany, K. Zhang, and M. Goldbaum, "Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images," Mendeley Data, V3, 2018. https://data.mendeley.com/datasets/rscbjbr9sj/3

[2] D. S. Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," Cell, vol. 172, no. 5, 2018. https://doi.org/10.1016/j.cell.2018.02.010

[3] P. T. Mooney, "Chest X-Ray Images (Pneumonia)," Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
