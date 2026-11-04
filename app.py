"""Interactive evaluation of saved classifiers on an uploaded held-out CSV."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Classifier Evaluation Lab", layout="wide")


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def load_metadata() -> dict:
    metadata_path = MODEL_DIR / "metadata.json"
    with metadata_path.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def markdown_table(dataframe: pd.DataFrame) -> str:
    headers = [str(column) for column in dataframe.columns]
    rows = [
        [str(index), *(str(value) for value in row)]
        for index, row in dataframe.iterrows()
    ]
    header_row = ["", *headers]
    table_rows = [header_row, ["---"] * len(header_row), *rows]
    return "\n".join("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in table_rows)


def evaluate_model(model, features: pd.DataFrame, labels: pd.Series, metadata: dict):
    predictions = model.predict(features)
    class_names = metadata["class_names"]
    positive_class = metadata["positive_class"]
    positive_index = list(model.classes_).index(positive_class)
    positive_scores = model.predict_proba(features)[:, positive_index]
    positive_truth = labels.eq(positive_class).astype(int)
    auc = roc_auc_score(positive_truth, positive_scores) if labels.nunique() == 2 else None
    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": auc,
        "Precision": precision_score(labels, predictions, pos_label=positive_class),
        "Recall": recall_score(labels, predictions, pos_label=positive_class),
        "F1": f1_score(labels, predictions, pos_label=positive_class),
        "MCC": matthews_corrcoef(labels, predictions),
    }
    matrix = confusion_matrix(labels, predictions, labels=class_names)
    report = classification_report(
        labels,
        predictions,
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return metrics, matrix, pd.DataFrame(report).transpose()


st.title("Classifier Evaluation Lab")
st.caption("Wisconsin Breast Cancer Diagnostic dataset | Held-out test evaluation")

metadata_path = MODEL_DIR / "metadata.json"
if not metadata_path.exists():
    st.error("Model artifacts are missing. Run `python train_models.py` from this project folder first.")
    st.stop()

metadata = load_metadata()
model_files = metadata["model_files"]
model_name = st.selectbox("Classifier", list(model_files))
uploaded_file = st.file_uploader(
    "Test dataset (CSV)",
    type=["csv"],
    help="Include all 30 feature columns and the diagnosis target. No model is retrained.",
)

if uploaded_file is None:
    st.info("Upload `test_data.csv` from this project, or another labeled test CSV with the same columns.")
    st.stop()

try:
    test_data = pd.read_csv(uploaded_file)
except (UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

target_column = metadata["target_column"]
feature_names = metadata["feature_names"]
required_columns = feature_names + [target_column]
missing_columns = [column for column in required_columns if column not in test_data.columns]
if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()
if test_data.empty:
    st.error("The uploaded CSV has no rows to evaluate.")
    st.stop()

features = test_data.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
labels = test_data[target_column].astype(str)
if features.isna().any().any():
    st.error("Feature columns must contain numeric values without missing cells.")
    st.stop()
unknown_labels = sorted(set(labels) - set(metadata["class_names"]))
if unknown_labels:
    st.error(f"Unexpected target labels: {', '.join(unknown_labels)}")
    st.stop()

model_path = MODEL_DIR / model_files[model_name]
model = load_model(str(model_path))
metrics, matrix, report = evaluate_model(model, features, labels, metadata)

st.subheader(model_name)
st.write(f"Evaluated **{len(test_data)}** rows. The positive class for AUC, precision, recall, and F1 is **{metadata['positive_class']}**.")
metric_columns = st.columns(3)
for index, metric_name in enumerate(METRIC_NAMES):
    value = metrics[metric_name]
    display_value = f"{value:.3f}" if value is not None else "N/A"
    metric_columns[index % 3].metric(metric_name, display_value)

left, right = st.columns([1, 1.4])
with left:
    st.markdown("**Confusion matrix**")
    st.caption("Rows are actual labels; columns are predicted labels.")
    confusion_table = pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in metadata["class_names"]],
        columns=[f"Predicted {label}" for label in metadata["class_names"]],
    )
    st.markdown(markdown_table(confusion_table))
with right:
    st.markdown("**Classification report**")
    st.markdown(markdown_table(report.round(3)))

with st.expander("Preview uploaded rows"):
    preview = test_data.loc[:, feature_names[:5] + [target_column]].head(10)
    st.markdown(markdown_table(preview))
