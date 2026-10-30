"""
Network Intrusion Detection - five-model comparison

Expected input files in the project root:
    train.csv
    test.csv
    sample_submission.csv

Run:
    python train.py

This script creates:
    results/model_comparison.csv
    submission_lightgbm.csv
    submission_xgboost.csv
    submission_randomforest.csv
    submission_extratrees.csv
    submission_catboost.csv

IMPORTANT:
The stratified random split is only a baseline. The competition describes
unseen traffic behaviour profiles in its hidden test set, so random-split
validation may substantially overestimate leaderboard performance.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "Target"
ID_COL = "id"
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Read the three competition CSV files."""
    train_path = ROOT / "train.csv"
    test_path = ROOT / "test.csv"
    sample_path = ROOT / "sample_submission.csv"

    for path in (train_path, test_path, sample_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Download it from Kaggle and place it "
                f"in: {ROOT}"
            )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    if TARGET not in train.columns:
        raise ValueError(f"Training data must contain '{TARGET}'.")

    if ID_COL not in test.columns or ID_COL not in sample.columns:
        raise ValueError("Both test.csv and sample_submission.csv need an id column.")

    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.drop(columns=[ID_COL], errors="ignore").reindex(columns=X.columns)
    y_text = train[TARGET].astype(str)

    return X, y_text, X_test, test, sample


def make_preprocessor(X):
    """Impute numeric values and one-hot encode categorical values."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn versions before 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ], remainder="drop")


def build_models(num_classes):
    """Create the five classifiers. Optional libraries are checked explicitly."""
    models = {}

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
        )
    except ImportError:
        print("LightGBM unavailable; install with: pip install lightgbm")

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=350, max_depth=7, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            objective="multi:softprob", num_class=num_classes,
            eval_metric="mlogloss", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    except ImportError:
        print("XGBoost unavailable; install with: pip install xgboost")

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RANDOM_STATE,
        n_jobs=-1
    )
    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=300, depth=7, learning_rate=0.05,
            loss_function="MultiClass", verbose=False,
            random_seed=RANDOM_STATE, thread_count=-1
        )
    except ImportError:
        print("CatBoost unavailable; install with: pip install catboost")

    return models


def main():
    X, y_text, X_test, test, sample = load_data()

    # Encode target labels consistently for all five libraries.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    num_classes = len(label_encoder.classes_)

    print(f"Training rows: {len(X):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Features: {X.shape[1]}")
    print("Classes:", list(label_encoder.classes_))

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    train_weights = compute_sample_weight("balanced", y_train)
    all_weights = compute_sample_weight("balanced", y)
    models = build_models(num_classes)

    if not models:
        raise RuntimeError("No models are available. Install the dependencies.")

    results = []

    for name, classifier in models.items():
        print(f"\n{'=' * 18} {name} {'=' * 18}")
        preprocessor = make_preprocessor(X)

        validation_model = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ])

        fit_args = {}
        if name in {"LightGBM", "XGBoost", "CatBoost"}:
            fit_args["classifier__sample_weight"] = train_weights

        validation_model.fit(X_train, y_train, **fit_args)
        valid_pred = validation_model.predict(X_valid).astype(int)

        score = f1_score(y_valid, valid_pred, average="macro")
        print(f"Random-split validation Macro F1: {score:.5f}")
        print(classification_report(
            y_valid, valid_pred,
            labels=np.arange(num_classes),
            target_names=label_encoder.classes_,
            zero_division=0
        ))
        results.append({"model": name, "validation_macro_f1": score})

        # Refit on all labeled records to create the Kaggle test predictions.
        full_model = Pipeline([
            ("preprocess", make_preprocessor(X)),
            ("classifier", classifier),
        ])
        full_args = {}
        if name in {"LightGBM", "XGBoost", "CatBoost"}:
            full_args["classifier__sample_weight"] = all_weights

        full_model.fit(X, y, **full_args)
        test_pred_encoded = full_model.predict(X_test).astype(int)
        test_pred = label_encoder.inverse_transform(test_pred_encoded)

        # Preserve the official sample-submission ID order.
        pred_frame = pd.DataFrame({
            ID_COL: test[ID_COL],
            TARGET: test_pred
        })
        submission = sample[[ID_COL]].merge(
            pred_frame, on=ID_COL, how="left", validate="one_to_one"
        )

        if submission[TARGET].isna().any() or len(submission) != len(test):
            raise ValueError(f"Invalid prediction output for {name}.")

        allowed = set(label_encoder.classes_)
        if not set(submission[TARGET].unique()).issubset(allowed):
            raise ValueError(f"Unexpected class label in {name} submission.")

        filename = f"submission_{name.lower()}.csv"
        submission.to_csv(ROOT / filename, index=False)
        print(f"Saved {filename}")

    results_df = pd.DataFrame(results).sort_values(
        "validation_macro_f1", ascending=False
    )
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("\nModel comparison (random validation only):")
    print(results_df.to_string(index=False))
    print("\nDone. Do not treat random-split validation as a leaderboard estimate.")


if __name__ == "__main__":
    main()
