# Model Results

This folder contains the results generated during model training and evaluation.

The project compares five machine learning algorithms:

- LightGBM
- XGBoost
- Random Forest
- Extra Trees
- CatBoost

## Evaluation Metric

The models are evaluated using **Macro F1-score**, which calculates the F1-score separately for each class and takes their average.

This is useful for network intrusion detection because the model should perform well across all attack categories, not just the most frequent one.

## Generated Files

After running `train.py`, the following files are generated:

| File | Description |
|---|---|
| `model_comparison.csv` | Comparison of model performance on the validation dataset |
| `submission_lightgbm.csv` | Predictions from LightGBM |
| `submission_xgboost.csv` | Predictions from XGBoost |
| `submission_random_forest.csv` | Predictions from Random Forest |
| `submission_extratrees.csv` | Predictions from Extra Trees |
| `submission_catboost.csv` | Predictions from CatBoost |

## Note

The validation results are based on a random train-validation split. Since the competition test data may contain different network traffic patterns, validation scores may differ from the actual Kaggle leaderboard scores.

The final submission performance is determined by the competition's hidden test labels.
