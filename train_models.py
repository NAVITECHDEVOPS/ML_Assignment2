"""Train and evaluate the five required classifiers on the UCI diagnostic data."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "diagnosis"
POSITIVE_CLASS = "malignant"
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the public Wisconsin diagnostic dataset with readable class labels."""
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    labels = dataset.target.map({0: "malignant", 1: "benign"}).rename(TARGET_COLUMN)
    return features, labels


def build_models() -> dict[str, object]:
    """Create all five classifiers; scale features inside relevant pipelines."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE)),
            method="sigmoid",
            cv=5,
        ),
    }


def calculate_metrics(model: object, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """Calculate binary metrics with malignant as the clinically important positive class."""
    predictions = model.predict(features)
    class_index = list(model.classes_).index(POSITIVE_CLASS)
    malignant_scores = model.predict_proba(features)[:, class_index]
    malignant_truth = labels.eq(POSITIVE_CLASS).astype(int)
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": roc_auc_score(malignant_truth, malignant_scores),
        "Precision": precision_score(labels, predictions, pos_label=POSITIVE_CLASS),
        "Recall": recall_score(labels, predictions, pos_label=POSITIVE_CLASS),
        "F1": f1_score(labels, predictions, pos_label=POSITIVE_CLASS),
        "MCC": matthews_corrcoef(labels, predictions),
    }


def main() -> None:
    features, labels = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    full_data = features.copy()
    full_data[TARGET_COLUMN] = labels
    full_data.to_csv(DATA_DIR / "breast_cancer.csv", index=False)

    result_rows = []
    model_files = {}
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        result_rows.append({"Model": model_name, **calculate_metrics(model, x_test, y_test)})
        model_file = f"{model_name.lower().replace(' ', '_').replace('-', '')}.joblib"
        joblib.dump(model, MODEL_DIR / model_file)
        model_files[model_name] = model_file

    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(PROJECT_DIR / "test_data.csv", index=False)

    results = pd.DataFrame(result_rows)
    results.to_csv(MODEL_DIR / "metrics.csv", index=False)
    metadata = {
        "dataset": "Wisconsin Breast Cancer Diagnostic",
        "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "target_column": TARGET_COLUMN,
        "positive_class": POSITIVE_CLASS,
        "class_names": ["malignant", "benign"],
        "feature_names": features.columns.tolist(),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "model_files": model_files,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Dataset shape: {full_data.shape[0]} rows, {features.shape[1]} features")
    print(f"Held-out test rows: {len(test_data)}")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Saved models, metrics, metadata, source CSV, and test_data.csv.")


if __name__ == "__main__":
    main()
