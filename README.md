# Chest X-Ray Pneumonia Screening (AIE231)

## Problem Statement
A radiology department needs a screening tool that flags chest X-rays likely to show pneumonia so that a radiologist reviews them first. The model takes a chest X-ray image and outputs the probability of pneumonia (binary classification: `NORMAL` vs. `PNEUMONIA`). Because a missed pneumonia case (false negative) is far more costly than a false alarm, recall on the `PNEUMONIA` class is the primary metric, supported by macro-F1 and ROC-AUC.
## Dataset (source)
Chest X-Ray Images (Pneumonia) [3], derived from Kermany et al. [1], [2]. Licence: CC BY 4.0 (original Mendeley data [1]).
## Model Family
CNN: a custom CNN built from scratch, then VGG16 or ResNet-50 transfer learning and fine-tuning.

## License
Kaggle lists "Other (specified in description)". The original Mendeley Data dataset [1] is released under CC BY 4.0, which permits academic use with attribution. This is an educational prototype, not a clinical diagnostic tool.

## References
[1] D. Kermany, K. Zhang, and M. Goldbaum, "Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images," Mendeley Data, V3, 2018. https://data.mendeley.com/datasets/rscbjbr9sj/3

[2] D. S. Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," Cell, vol. 172, no. 5, 2018. doi: 10.1016/j.cell.2018.02.010

[3] P. T. Mooney, "Chest X-Ray Images (Pneumonia)," Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
