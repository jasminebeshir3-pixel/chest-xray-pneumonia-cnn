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

## Team

| Member | Role | Work package | Files so far |
|--------|------|--------------|--------------|
| Amir Emil, 24100316 | Tech Lead | problem formulation, repository, config, integration, EDA and baselines, final model with 3 seeds | `requirements.txt`, `configs/base.yaml`, `.gitignore`, `notebooks/01_eda.ipynb`, `src/baselines.py`, `experiments/` |
| Youssef Islam, 24100674 | Data Engineer | loading, cleaning, leak-free split, preprocessing, class imbalance, augmentation | `src/data.py`, `data/`, `tests/test_data.py` |
| Lara, 24100605 | Baseline and Analysis Engineer | trains Model A (custom CNN) and compares it with the baselines | starts in Phase 3 |
| Jasmine Beshir, 24100515 | Model Architect | architectures, hand parameter counts, regularization, GradientTape step | `src/models.py` |
| Sarah Ahmed, 24100558 | Experimentation Engineer | experiment log, tuning studies, learning curves | `src/train.py` |
| Maryam Gomaa, [ID] | Evaluation and Deployment Engineer | metrics, error analysis, `predict.py`, demo app | `src/evaluate.py` |

## License

Kaggle lists "Other (specified in description)". The original Mendeley Data dataset [1] is released under CC BY 4.0, which permits academic use with attribution. This is an educational prototype for the AIE231 course, not a clinical diagnostic tool.

## References

[1] D. Kermany, K. Zhang, and M. Goldbaum, "Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images," Mendeley Data, V3, 2018. https://data.mendeley.com/datasets/rscbjbr9sj/3

[2] D. S. Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," Cell, vol. 172, no. 5, 2018. https://doi.org/10.1016/j.cell.2018.02.010

[3] P. T. Mooney, "Chest X-Ray Images (Pneumonia)," Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
