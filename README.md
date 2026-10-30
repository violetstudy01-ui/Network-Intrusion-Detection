# Network Intrusion Detection – ML Challenge

A Python machine-learning project that compares five classifiers for the four-class network-traffic classification task:

- Benign
- Availability
- Reconnaissance
- Spoofing

The project trains LightGBM, XGBoost, Random Forest, Extra Trees, and CatBoost, evaluates them using **Macro F1**, and writes Kaggle-format prediction files.

## Important validation warning

The competition description says the hidden evaluation data contains network-traffic behaviour profiles deliberately separated from the training data. The included `train_test_split(..., stratify=y)` is a **random, stratified baseline split**. It preserves class proportions, but it does not ensure that related traffic profiles are separated between training and validation.

Consequently, a high local random-validation Macro F1 can be misleading and should not be presented as an estimate of the Kaggle public/private score. The public leaderboard score is calculated by Kaggle on its hidden public subset. The private leaderboard score uses a separate hidden subset.

For a stronger evaluation, investigate whether the supplied features support meaningful profile/group identification and design a group-aware validation split that approximates the competition's unseen-profile setup. Do not claim that a group split is valid unless the grouping logic is supported by the data.

## Project structure

```text
Network-Intrusion-Detection/
├── README.md
├── train.py
├── requirements.txt
├── .gitignore
├── models/
│   └── README.md
└── results/
    └── README.md
```

The Kaggle CSV files and generated prediction files are intentionally not included in this repository. Place the three competition files in the project root locally.

## Dataset files required

Download these files from the Kaggle competition page:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Expected columns:
- `train.csv`: input features, `id`, and target column `Target`
- `test.csv`: input features and `id`
- `sample_submission.csv`: `id` and `Target`

The script excludes `id` from model features, aligns test columns to the training feature columns, imputes missing values, and one-hot encodes categorical input features.

## Algorithms used

| Algorithm | Main idea |
|---|---|
| LightGBM | Gradient-boosted decision trees with efficient histogram-based learning |
| XGBoost | Regularized gradient-boosted decision trees |
| Random Forest | Ensemble of independently trained decision trees |
| Extra Trees | Randomized tree ensemble |
| CatBoost | Gradient boosting designed to work effectively with categorical data |

For a consistent comparison in this starter pipeline, categorical columns are one-hot encoded before all five models. CatBoost's native categorical-feature handling is therefore **not** being used in this particular script.

Class-balanced sample weights are applied to the boosting models; Random Forest and Extra Trees use their built-in class-weight options. These are experiment choices, not a guarantee of improved leaderboard performance.

## Setup

Python 3.10 or newer is recommended.

Create and activate a virtual environment (optional but recommended):

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Put the three CSV files in the same folder as `train.py`, then run:

```bash
python train.py
```

The script:
1. Loads the CSV files.
2. Separates features and target.
3. Encodes the target labels.
4. Creates a stratified random 80/20 train-validation split.
5. Trains each available model and prints its validation Macro F1 and classification report.
6. Refits each model on all labeled training records.
7. Generates one Kaggle-format CSV per model.
8. Saves the local model comparison to `results/model_comparison.csv`.

Generated submissions:
- `submission_lightgbm.csv`
- `submission_xgboost.csv`
- `submission_randomforest.csv`
- `submission_extratrees.csv`
- `submission_catboost.csv`

Each submission contains exactly two columns: `id` and `Target`.

## Evaluation metric: Macro F1

Macro F1 calculates the F1 score independently for each class, then takes the unweighted average:

\[
F1_{\text{macro}} = \frac{F1_{\text{Benign}} + F1_{\text{Availability}} + F1_{\text{Reconnaissance}} + F1_{\text{Spoofing}}}{4}
\]

This gives each class equal importance, regardless of its frequency. It is therefore more informative than accuracy when classes are imbalanced.

## Reproducibility and limitations

- A fixed random seed (`42`) is used for the baseline split and model initialisation.
- Results may vary with library versions, hardware, and execution environment.
- The script's random validation scores are not directly comparable to Kaggle's public/private scores.
- No external labeled intrusion-detection datasets are used.
- Do not upload hidden test labels, private data, credentials, or large competition CSVs to GitHub.
- Check the competition rules before sharing any competition-provided data.

## Competition reference

Kaggle: [Network Intrusion Detection ML Challenge](https://www.kaggle.com/competitions/network-intrusion-detection2-ml-challenge/overview)

Dataset attribution stated in the competition: B. S. and Rohini Nagapadma, The National Institute of Engineering, Mysuru. The competition page states a CC BY 4.0 dataset license; competition files remain subject to the competition rules.
