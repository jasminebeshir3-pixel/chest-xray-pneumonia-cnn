# Model Architect

Read `HOW_TO_PUSH.md` (in this folder) first for the git steps.

Branch: `feature/baseline-models`
Copy `src/models.py` into the repo's `src` folder.

Commit message: `Add baseline models (majority class and logistic regression)`

```
git add src/models.py
```

Reviewer: the Evaluation and Deployment Engineer.
You can push this any time after the Data Engineer's pull request is merged.

## What your file does (be ready to explain)

`src/models.py` holds every model of the project. In Phase 2 it has the two baselines:

- `majority_baseline()`: scikit-learn `DummyClassifier(strategy="most_frequent")`. It learns nothing except which class is most common in the training set (PNEUMONIA, about 73%), and always predicts it.
- `logistic_baseline()`: a pipeline of `StandardScaler` then `LogisticRegression`.
  - Input: each image as 64 x 64 grayscale = 4096 numbers.
  - The scaler is inside the pipeline, so `fit` learns the mean and standard deviation from the training images only. Validation and test images are scaled with those training numbers, so nothing leaks.
  - `C=0.01` is strong L2 regularization. We have about 4,000 training images and 4,096 features, so without it the model just memorises the training set.
  - `class_weight="balanced"` gives NORMAL images more weight because there are fewer of them.
  - It has 4096 weights + 1 bias = 4,097 parameters.

In Phase 3 you add the three CNNs (custom CNN, frozen VGG16, fine-tuned VGG16) to this same file, with the hand parameter counts. That part is ready and waiting, it will be sent later.
