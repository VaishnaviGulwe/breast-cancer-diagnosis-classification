import os
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
)


# CONFIGURATION


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

METRIC_COLUMNS = [
    "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"
]


# PAGE DESIGN


st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1 {
    font-size: 2rem !important;
    font-weight: 750 !important;
}
h2 {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
}
h3 {
    font-size: 1.15rem !important;
    font-weight: 650 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.95rem !important;
}
</style>
""", unsafe_allow_html=True)

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.2
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


# LOAD SAVED MODELS AND TEST DATA


@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(
        MODEL_DIR,
        MODEL_FILES[model_name]
    )
    return joblib.load(model_path)


@st.cache_resource
def load_features():
    feature_path = os.path.join(
        MODEL_DIR,
        "feature_columns.joblib"
    )
    return list(joblib.load(feature_path))


@st.cache_data
def load_default_test_data():
    return pd.read_csv(TEST_DATA_PATH)



# PREPARE TEST DATA


def prepare_test_data(df, feature_columns):

    if df.empty:
        raise ValueError("The test CSV is empty.")

    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    if "diagnosis" not in df.columns:
        raise ValueError(
            "The test CSV must include a diagnosis column."
        )

    X = df.loc[:, feature_columns].copy()

    for col in feature_columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    y = df["diagnosis"].replace({
        "B": 0,
        "M": 1,
        "b": 0,
        "m": 1
    })

    y = pd.to_numeric(y, errors="coerce")

    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError(
            "diagnosis must contain only B/M or 0/1."
        )

    return X, y.astype(int)



# EVALUATE MODEL


def evaluate(model, X, y):

    predictions = model.predict(X)

    class_labels = list(model.classes_)

    if 1 not in class_labels:
        raise ValueError(
            "The saved model does not have malignant class 1."
        )

    probabilities = model.predict_proba(X)[
        :, class_labels.index(1)
    ]

    scores = {
        "Accuracy": accuracy_score(y, predictions),

        "AUC": (
            roc_auc_score(y, probabilities)
            if y.nunique() == 2
            else np.nan
        ),

        "Precision": precision_score(
            y, predictions, zero_division=0
        ),

        "Recall": recall_score(
            y, predictions, zero_division=0
        ),

        "F1": f1_score(
            y, predictions, zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y, predictions
        ),
    }

    return predictions, probabilities, scores



# DOWNLOAD CHARTS


def download_figure(fig, label, filename):

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=300,
        bbox_inches="tight"
    )

    st.download_button(
        label,
        buffer.getvalue(),
        file_name=filename,
        mime="image/png"
    )



# MAIN DASHBOARD


st.title("🧠 Machine Learning Classification Dashboard")

st.caption(
    "BITS Pilani | M.Tech AI & ML | Assignment 2"
)

st.divider()


# SIDEBAR


st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES)
)

st.sidebar.subheader("Test Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload a different test CSV (optional)",
    type=["csv"]
)


# AUTOMATIC TEST DATA LOADING


try:

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.sidebar.success(
            "Uploaded CSV loaded successfully"
        )

        data_source = "Uploaded test dataset"

    else:

        df = load_default_test_data()

        st.sidebar.success(
            "Default test_data.csv loaded automatically"
        )

        data_source = "Bundled test_data.csv"

    feature_columns = load_features()

    X_test, y_test = prepare_test_data(
        df,
        feature_columns
    )

except (
    FileNotFoundError,
    ValueError,
    pd.errors.ParserError,
    OSError
) as error:

    st.error(
        f"Unable to load test data: {error}"
    )

    st.info(
        "Ensure test_data.csv and "
        "model/feature_columns.joblib exist."
    )

    st.stop()



# DASHBOARD TABS


tab1, tab2 = st.tabs([
    "Model Evaluation",
    "All Models Comparison"
])



# TAB 1 - MODEL EVALUATION


with tab1:

    st.header("1. Test Dataset")

    st.success(
        f"Dataset loaded: {data_source}"
    )

    st.write(
        f"Dataset shape: {df.shape[0]} rows "
        f"× {df.shape[1]} columns"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    
    # LOAD SELECTED MODEL
    

    try:

        model = load_model(selected_model)

        y_pred, y_prob, scores = evaluate(
            model,
            X_test,
            y_test
        )

    except Exception as error:

        st.error(
            f"Could not evaluate {selected_model}: {error}"
        )

        st.stop()

    st.header("2. Selected Model")

    st.info(selected_model)

    
    # EVALUATION METRICS
    

    st.header("3. Evaluation Metrics")

    for metric_row in (
        METRIC_COLUMNS[:3],
        METRIC_COLUMNS[3:]
    ):

        columns = st.columns(3)

        for col, metric_name in zip(
            columns,
            metric_row
        ):

            value = scores[metric_name]

            col.metric(
                "F1 Score"
                if metric_name == "F1"
                else metric_name,

                f"{value:.4f}"
                if pd.notna(value)
                else "N/A"
            )

    
    # CONFUSION MATRIX
    

    st.header("4. Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(
        figsize=(6.4, 4.6),
        constrained_layout=True
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        annot_kws={
            "size": 13,
            "weight": "bold"
        },
        xticklabels=[
            "Benign",
            "Malignant"
        ],
        yticklabels=[
            "Benign",
            "Malignant"
        ]
    )

    ax.set(
        title=f"Confusion Matrix - {selected_model}",
        xlabel="Predicted",
        ylabel="Actual"
    )

    left, middle, right = st.columns(
        [1, 2, 1]
    )

    with middle:
        st.pyplot(
            fig,
            use_container_width=True
        )

    download_figure(
        fig,
        "Download Confusion Matrix (300 DPI PNG)",
        "confusion_matrix.png"
    )

    plt.close(fig)

    
    # CLASSIFICATION REPORT
    

    st.header("5. Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Benign",
            "Malignant"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    
    # ROC CURVE
    

    st.header("6. ROC Curve")

    if y_test.nunique() == 2:

        fpr, tpr, _ = roc_curve(
            y_test,
            y_prob
        )

        fig2, ax2 = plt.subplots(
            figsize=(7, 4.6),
            constrained_layout=True
        )

        ax2.plot(
            fpr,
            tpr,
            label=f"AUC = {scores['AUC']:.4f}"
        )

        ax2.plot(
            [0, 1],
            [0, 1],
            "k--"
        )

        ax2.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"ROC Curve - {selected_model}"
        )

        ax2.legend()

        st.pyplot(
            fig2,
            use_container_width=True
        )

        download_figure(
            fig2,
            "Download ROC Curve (300 DPI PNG)",
            "roc_curve.png"
        )

        plt.close(fig2)

    else:

        st.warning(
            "ROC/AUC requires both diagnosis classes."
        )

    
    # PREDICTION RESULTS
    

    st.header("7. Prediction Results")

    output_df = df.copy()

    output_df["Predicted Diagnosis"] = y_pred

    output_df["Predicted Label"] = np.where(
        y_pred == 1,
        "Malignant",
        "Benign"
    )

    output_df["Malignant Probability"] = y_prob

    st.dataframe(
        output_df.head(20),
        use_container_width=True
    )

    st.download_button(
        "Download Predictions CSV",
        output_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )



# TAB 2 - ALL MODELS COMPARISON


with tab2:

    st.header(
        "Comparison of Five Classification Models"
    )

    st.caption(
        f"Results on {len(y_test)} test rows "
        f"from {data_source}."
    )

    
    # EVALUATE ALL FIVE SAVED MODELS
    

    results = []

    for model_name in MODEL_FILES:

        try:

            saved_model = load_model(
                model_name
            )

            _, _, model_scores = evaluate(
                saved_model,
                X_test,
                y_test
            )

            results.append({
                "ML Model Name": model_name,
                **model_scores
            })

        except Exception as error:

            st.error(
                f"Unable to evaluate "
                f"{model_name}: {error}"
            )

    
    # DISPLAY COMPARISON TABLE
    

    if results:

        metrics_df = pd.DataFrame(results)

        st.subheader(
            "Evaluation Metrics Table"
        )

        st.dataframe(
            metrics_df.round(4),
            use_container_width=True
        )

        st.download_button(
            "Download Metrics CSV",
            metrics_df.to_csv(index=False),
            file_name="evaluated_metrics.csv",
            mime="text/csv"
        )

        
        # MODEL COMPARISON CHART
        

        st.subheader(
            "Performance Comparison Chart"
        )

        fig3, ax3 = plt.subplots(
            figsize=(9, 5.4),
            constrained_layout=True
        )

        metrics_df.set_index(
            "ML Model Name"
        )[METRIC_COLUMNS].plot(
            kind="bar",
            ax=ax3,
            width=0.78
        )

        ax3.set(
            title="Test Set: Model Comparison",
            xlabel="Classifier",
            ylabel="Score",
            ylim=(0, 1.05)
        )

        ax3.tick_params(
            axis="x",
            rotation=20
        )

        for tick in ax3.get_xticklabels():
            tick.set_ha("right")

        ax3.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False
        )

        st.pyplot(
            fig3,
            use_container_width=True
        )

        download_figure(
            fig3,
            "Download Comparison Chart (300 DPI PNG)",
            "model_comparison.png"
        )

        plt.close(fig3)

        
        # BEST PERFORMING MODEL
        

        st.subheader(
            "Best Performing Model"
        )

        best = metrics_df.loc[
            metrics_df["MCC"].idxmax()
        ]

        st.success(
            "Best Model Based on MCC: "
            f"{best['ML Model Name']}"
        )

        st.write(
            f"Accuracy: {best['Accuracy']:.4f} | "
            f"F1: {best['F1']:.4f} | "
            f"MCC: {best['MCC']:.4f}"
        )



# FOOTER


st.divider()

st.caption(
    "BITS Pilani | M.Tech AI & ML | "
    "Machine Learning Assignment 2"
)
