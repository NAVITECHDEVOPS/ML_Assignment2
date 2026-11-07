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

st.set_page_config(page_title="Model Review Studio", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #173847;
        --teal: #167a78;
        --amber: #c98231;
        --paper: #f4f6f3;
        --line: #d7e0dc;
        --muted: #647780;
    }
    [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1460px; padding-top: 1.5rem; padding-bottom: 3rem; }
    .masthead { border-bottom: 1px solid var(--line); padding: 0.4rem 0 1.2rem; margin-bottom: 1.35rem; }
    .masthead .eyebrow { color: var(--teal); font: 700 0.72rem/1.2 'Segoe UI', sans-serif; text-transform: uppercase; }
    .masthead h1 { color: var(--ink); font: 600 2.25rem/1.12 Georgia, serif; margin: 0.35rem 0; }
    .masthead p { color: var(--muted); font: 0.96rem/1.45 'Segoe UI', sans-serif; margin: 0; }
    .edition { border-left: 3px solid var(--amber); padding: 0.15rem 0 0.15rem 0.8rem; margin-top: 0.45rem; color: var(--muted); font-size: 0.85rem; }
    .edition strong { display: block; color: var(--ink); font-size: 1.08rem; margin: 0.15rem 0; }
    h2, h3 { color: var(--ink); }
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-top: 2px solid var(--teal); border-radius: 4px; padding: 0.7rem 0.8rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.55rem; }
    [data-testid="stFileUploaderDropzone"] { background: #fff; border-color: #9eb8b1; border-radius: 4px; }
    .section-note { color: var(--muted); font-size: 0.83rem; margin: -0.25rem 0 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def load_metadata() -> dict:
    metadata_path = MODEL_DIR / "metadata.json"
    with metadata_path.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def markdown_table(dataframe: pd.DataFrame) -> str:
    headers = [str(column) for column in dataframe.columns]

    def format_cell(value):
        if pd.isna(value):
            return "N/A"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    rows = [
        [str(index), *(format_cell(value) for value in row)]
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


header_left, header_right = st.columns([4, 1.15], vertical_alignment="center")
with header_left:
    st.markdown(
        """
        <div class="masthead">
          <div class="eyebrow">MODEL VALIDATION / WORKSPACE 01</div>
          <h1>Model Review Studio</h1>
          <p>Compare saved classifiers on labeled, held-out data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        """
        <div class="edition">
          EDUCATIONAL PROTOTYPE
          <strong>WDBC</strong>
          30 predictors / binary target
        </div>
        """,
        unsafe_allow_html=True,
    )

metadata_path = MODEL_DIR / "metadata.json"
if not metadata_path.exists():
    st.error("Model artifacts are missing. Run `python train_models.py` from this project folder first.")
    st.stop()

metadata = load_metadata()
model_files = metadata["model_files"]
feature_names = metadata["feature_names"]
target_column = metadata["target_column"]

setup_column, workspace_column = st.columns([0.9, 2.1], gap="large")
with setup_column:
    st.markdown("### Evaluation setup")
    model_name = st.selectbox("Focus model", list(model_files))
    uploaded_file = st.file_uploader(
        "Labeled test data",
        type=["csv"],
        help="CSV must contain the 30 model features and the diagnosis target.",
    )
    st.caption("Models are loaded from saved artifacts; uploaded data is not used for training.")

with workspace_column:
    st.markdown("### Benchmark workspace")
    st.markdown(
        '<p class="section-note">All five saved models are scored against the same uploaded rows.</p>',
        unsafe_allow_html=True,
    )
    if uploaded_file is None:
        st.info("Upload a labeled CSV to generate the model comparison and error analysis.")
        sample_path = PROJECT_DIR / "test_data.csv"
        if sample_path.exists():
            st.download_button(
                "Download sample test CSV",
                data=sample_path.read_bytes(),
                file_name="test_data.csv",
                mime="text/csv",
            )
        st.stop()

try:
    test_data = pd.read_csv(uploaded_file)
except (UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

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

models = {
    name: load_model(str(MODEL_DIR / model_file))
    for name, model_file in model_files.items()
}
comparison_rows = []
for name, candidate in models.items():
    candidate_metrics, _, _ = evaluate_model(candidate, features, labels, metadata)
    comparison_rows.append({"Model": name, **candidate_metrics})
comparison = pd.DataFrame(comparison_rows).set_index("Model")

st.markdown("### Model comparison")
st.write(f"Scored **{len(test_data)}** rows. Positive class: **{metadata['positive_class']}**.")
st.markdown(markdown_table(comparison))

model = models[model_name]
metrics, matrix, report = evaluate_model(model, features, labels, metadata)
st.markdown(f"### Error analysis / {model_name}")
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
