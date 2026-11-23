# Chest X-Ray Pneumonia Screening (AIE231)

## Problem Statement
A radiology department needs a screening tool that flags chest X-rays likely to show pneumonia so that a radiologist reviews them first[cite: 2, 3]. The model takes a chest X-ray image and outputs the probability of pneumonia (binary classification: `NORMAL` vs. `PNEUMONIA`)[cite: 2, 3]. Because a missed pneumonia case (false negative) is far more costly than a false alarm (which only requires an extra review), recall on the `PNEUMONIA` class is the primary metric, supported by macro-F1 and ROC-AUC[cite: 2, 3, 5].

## Dataset & Phase 1 Summary
- **Dataset:** Chest X-Ray Images (Pneumonia), with **5,856** images found in the dataset folders (4,273 `PNEUMONIA` / 1,583 `NORMAL`; the Kaggle page also notes 5,863)[cite: 2, 3].
- **Why It Fits AIE231:** Exceeds the 5,000-image minimum for image datasets, has clear class imbalance (~2.7:1), includes varying image dimensions and duplicate files, is not a toy dataset, and permits academic use with attribution[cite: 1, 2, 3, 8, 10].
- **Model Family:** Convolutional Neural Networks (CNN) — starting with a custom CNN built from scratch, followed by VGG16 or ResNet-50 transfer learning and fine-tuning[cite: 2, 4, 10].
- **Primary Metric:** Recall on `PNEUMONIA`, alongside macro-F1 and ROC-AUC[cite: 3, 5].

## References & License
- **Original Paper:** Kermany, D. S. et al., *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"*, *Cell*, 172(5), 2018. DOI: `10.1016/j.cell.2018.02.010`
- **Kaggle Source:** Mooney, P. T., *"Chest X-Ray Images (Pneumonia)"*, Kaggle (`kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia`)[cite: 2].
- **License:** Kaggle lists *"Other (specified in description)"*. The underlying Mendeley Data dataset (Kermany et al., Version 3, `https://data.mendeley.com/datasets/rscbjbr9sj/3`) is under CC BY 4.0, permitting academic use with attribution[cite: 8, 10].
- **Academic Scope:** Built strictly for the AIE231 course project as an educational prototype, not a clinical diagnostic tool[cite: 10].

## Repository Structure (Planned)
- `src/` — modular data, model, training, evaluation, and inference code[cite: 7]
- `notebooks/` — EDA and results exploration notebooks[cite: 7]
- `configs/` — experiment configuration files[cite: 7]
- `data/` — dataset notes (no raw dataset files committed)[cite: 7]
- `experiments/` — experiment log CSV and plots[cite: 7]
- `models/` — saved models and preprocessing artifacts[cite: 7]
- `app/` — demo application[cite: 7]
- `report/` — final PDF report and presentation slides
