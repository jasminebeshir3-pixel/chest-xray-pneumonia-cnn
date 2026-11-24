# Chest X-Ray Pneumonia Screening (AIE231)

## Problem statement

A radiology department needs a screening tool that flags chest X-rays likely to show pneumonia, so that a radiologist reviews them first. The model takes one chest X-ray and outputs the probability of pneumonia (binary classification: `NORMAL` vs `PNEUMONIA`). A missed pneumonia case (false negative) costs far more than a false alarm, which only means one extra review. So recall on the `PNEUMONIA` class is our primary metric, supported by macro-F1 and ROC-AUC.

## Task definition

| Item | Our choice | Why |
|------|------------|-----|
| Input | one X-ray resized to 224 x 224 x 3, pixels scaled to [0, 1] | 224 x 224 is the VGG16 / ResNet-50 input size; 3 channels because a few files are RGB and the pretrained models expect 3 |
| Output | one sigmoid unit = P(pneumonia) | two classes, so one probability is enough |
| Loss | binary cross-entropy, with class weights | the standard loss for a sigmoid output; the weights handle the 2.7 : 1 imbalance |
| Primary metric | recall on PNEUMONIA | a missed pneumonia is the most costly error |
| Supporting metrics | macro-F1 and ROC-AUC | a model that says "pneumonia" for everything gets 100% recall, these metrics catch that |
| Threshold | 0.5 for now | may be lowered later to raise recall, chosen on the validation set only |

## Dataset

- **Dataset:** Chest X-Ray Images (Pneumonia), 5,856 images in the dataset folders (4,273 `PNEUMONIA` / 1,583 `NORMAL`). The Kaggle page also mentions 5,863.
- **Why it fits AIE231:** more than the 5,000-image minimum, clear class imbalance (about 2.7 : 1), images of very different sizes, not a toy dataset, and the licence allows academic use with attribution.
- **Model family:** CNNs. A custom CNN from scratch first, then VGG16 transfer learning and fine-tuning.
- **Getting the data:** see [data/README.md](data/README.md). The images are not stored in git.

## Team

| Member | Role | Work package |
|--------|------|--------------|
| Amir emil , 24100316 | Tech Lead (also EDA and baselines, we are 5) | problem formulation, repo, config, integration, EDA, baselines, final model |
| Sarah ahmed , 24100558 | Data Engineer | `src/data.py`: loading, cleaning, leak-free split, preprocessing, imbalance, augmentation |
| Lara , 24100605 | Model Architect | `src/models.py`: architectures, hand parameter counts, regularization |
| Jasmine Beshir ,  24100515 | Experimentation Engineer | experiment log, tuning studies, learning curves |
| Youssef Islam , 24100674 | Evaluation and Deployment Engineer | `src/evaluate.py`, `src/predict.py`, demo app |

## Setup

```
pip install -r requirements.txt
python data/download_data.py
```

Set `CXR_DATA_DIR` to the folder the download script prints. On Kaggle just add the dataset to the notebook, no download needed.

## References and licence

- Kermany, D. S. et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", *Cell*, 172(5), 2018. DOI: 10.1016/j.cell.2018.02.010
- Mooney, P. T., "Chest X-Ray Images (Pneumonia)", Kaggle, kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Licence: Kaggle lists "Other (specified in description)". The original Mendeley Data release (Kermany et al., Version 3, https://data.mendeley.com/datasets/rscbjbr9sj/3) is CC BY 4.0, which allows academic use with attribution.
- This is a course project and an educational prototype, not a clinical diagnostic tool.

## Repository structure

```
configs/       experiment config files (base.yaml has the shared settings)
data/          download script, data README and our split (no images in git)
notebooks/     01_eda.ipynb, later 02_results.ipynb
src/           data.py, models.py, train.py, evaluate.py, predict.py
tests/         leakage and preprocessing checks
experiments/   experiment log (runs.csv) and plots
models/        final saved model
app/           demo application
report/        report PDF and slides
```
