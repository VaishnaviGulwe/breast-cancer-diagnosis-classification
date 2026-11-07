import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)


# 1. CONFIGURATION


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧠",
    layout="wide"
)


# 2. PROFESSIONAL STYLING


st.markdown("""
<style>
    .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 1.45rem !important;
        font-weight: 650 !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }

    p, label, .stMarkdown {
        font-size: 0.95rem;
        line-height: 1.55;
    }

    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.55rem;
        font-weight: 700;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 12px;
    }

    [data-testid="stSidebar"] {
        min-width: 250px;
    }
</style>
""", unsafe_allow_html=True)


# 3. CHART FORMATTING


sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def figure_to_png(fig):
    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    buffer.seek(0)
    return buffer.getvalue()



# 4. LOAD SAVED MODELS


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
    return joblib.load(feature_path)



# 5. DASHBOARD HEADER


st.title("Machine Learning Classification Dashboard")

st.caption(
    "BITS Pilani | M.Tech Artificial Intelligence "
    "and Machine Learning | Assignment 2"
)

st.markdown(
    "Evaluate five trained classification models "
    "using a labelled test dataset."
)

st.divider()


# 6. SIDEBAR


st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

st.sidebar.markdown(
    "**Available classifiers**\n"
    "- Logistic Regression\n"
    "- Decision Tree\n"
    "- K-Nearest Neighbors\n"
    "- Naive Bayes\n"
    "- Support Vector Machine"
)

st.sidebar.info(
    "Upload the generated test_data.csv file "
    "to evaluate the selected saved model."
)


# 7. DASHBOARD TABS


tab1, tab2 = st.tabs([
    "Model Evaluation",
    "All Models Comparison"
])


# 8. MODEL EVALUATION


with tab1:

    st.header("1. Upload Test Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file containing features and diagnosis",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info(
            "Upload test_data.csv to display predictions "
            "and evaluation metrics."
        )

    else:

        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("Uploaded CSV is empty.")
                st.stop()

            st.success("Dataset uploaded successfully.")

            st.subheader("Dataset Overview")

            col_a, col_b = st.columns(2)

            col_a.metric("Total Records", len(df))
            col_b.metric("Total Columns", len(df.columns))

            with st.expander("View Dataset Preview"):
                st.dataframe(
                    df.head(10),
                    use_container_width=True
                )

            # ==============================================
            # 9. VALIDATE DATASET
            # ==============================================

            feature_columns = load_features()

            missing_features = [
                col for col in feature_columns
                if col not in df.columns
            ]

            if missing_features:
                st.error(
                    f"Missing input features: {missing_features}"
                )
                st.stop()

            if "diagnosis" not in df.columns:
                st.error(
                    "The uploaded CSV must contain "
                    "the diagnosis target column."
                )
                st.stop()

            X_test = df[feature_columns].copy()

            for col in feature_columns:
                X_test[col] = pd.to_numeric(
                    X_test[col],
                    errors="coerce"
                )

            y_test = df["diagnosis"].replace({
                "B": 0,
                "M": 1
            })

            y_test = pd.to_numeric(
                y_test,
                errors="coerce"
            )

            if (
                y_test.isna().any()
                or not y_test.isin([0, 1]).all()
            ):
                st.error(
                    "Diagnosis values must be B/M or 0/1."
                )
                st.stop()

            y_test = y_test.astype(int)

            # ==============================================
            # 10. LOAD MODEL AND PREDICT
            # ==============================================

            model = load_model(selected_model)

            y_pred = model.predict(X_test)

            probabilities = model.predict_proba(X_test)

            positive_index = list(
                model.classes_
            ).index(1)

            y_prob = probabilities[:, positive_index]

            st.header("2. Selected Classification Model")

            st.info(selected_model)

            # ==============================================
            # 11. CALCULATE SIX METRICS
            # ==============================================

            accuracy = accuracy_score(y_test, y_pred)

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y_test,
                y_pred
            )

            if y_test.nunique() == 2:
                auc = roc_auc_score(y_test, y_prob)
            else:
                auc = np.nan

            st.header("3. Model Performance Metrics")

            metric_col1, metric_col2, metric_col3 = (
                st.columns(3)
            )

            metric_col1.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

            metric_col2.metric(
                "AUC",
                f"{auc:.4f}"
                if not np.isnan(auc) else "N/A"
            )

            metric_col3.metric(
                "Precision",
                f"{precision:.4f}"
            )

            metric_col4, metric_col5, metric_col6 = (
                st.columns(3)
            )

            metric_col4.metric(
                "Recall",
                f"{recall:.4f}"
            )

            metric_col5.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            metric_col6.metric(
                "MCC",
                f"{mcc:.4f}"
            )

            # ==============================================
            # 12. CONFUSION MATRIX
            # ==============================================

            st.header("4. Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )

            fig, ax = plt.subplots(
                figsize=(6, 4.5)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                linewidths=0.5,
                linecolor="white",
                cbar=False,
                ax=ax
            )

            ax.set_title(
                f"Confusion Matrix - {selected_model}",
                pad=15
            )

            ax.set_xlabel("Predicted Class", labelpad=10)
            ax.set_ylabel("Actual Class", labelpad=10)

            ax.tick_params(axis="x", rotation=0)
            ax.tick_params(axis="y", rotation=0)

            fig.tight_layout()

            st.pyplot(fig, use_container_width=False)

            st.download_button(
                "Download Confusion Matrix (300 DPI)",
                data=figure_to_png(fig),
                file_name="confusion_matrix.png",
                mime="image/png"
            )

            plt.close(fig)

            # ==============================================
            # 13. CLASSIFICATION REPORT
            # ==============================================

            st.header("5. Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["Benign", "Malignant"],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df.style.format(
                    {
                        col: "{:.4f}"
                        for col in [
                            "precision",
                            "recall",
                            "f1-score"
                        ]
                        if col in report_df.columns
                    }
                ),
                use_container_width=True
            )

            st.download_button(
                "Download Classification Report",
                data=report_df.to_csv(),
                file_name="classification_report.csv",
                mime="text/csv"
            )

            # ==============================================
            # 14. ROC CURVE
            # ==============================================

            st.header("6. ROC Curve")

            if y_test.nunique() == 2:

                fpr, tpr, _ = roc_curve(
                    y_test,
                    y_prob
                )

                fig2, ax2 = plt.subplots(
                    figsize=(7, 4.5)
                )

                ax2.plot(
                    fpr,
                    tpr,
                    color="#1769AA",
                    linewidth=2.5,
                    label=f"AUC = {auc:.4f}"
                )

                ax2.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                    color="gray",
                    linewidth=1.5,
                    label="Random Classifier"
                )

                ax2.set_title(
                    f"ROC Curve - {selected_model}",
                    pad=15
                )

                ax2.set_xlabel(
                    "False Positive Rate",
                    labelpad=10
                )

                ax2.set_ylabel(
                    "True Positive Rate",
                    labelpad=10
                )

                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1.02)

                ax2.legend(loc="lower right")
                ax2.grid(alpha=0.25)

                fig2.tight_layout()

                st.pyplot(
                    fig2,
                    use_container_width=False
                )

                st.download_button(
                    "Download ROC Curve (300 DPI)",
                    data=figure_to_png(fig2),
                    file_name="roc_curve.png",
                    mime="image/png"
                )

                plt.close(fig2)

            else:
                st.warning(
                    "ROC curve requires both target classes."
                )

            # ==============================================
            # 15. PREDICTION RESULTS
            # ==============================================

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
                label="Download Predictions CSV",
                data=output_df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )

        except FileNotFoundError as error:
            st.error(
                "Saved model files are missing. "
                "Run train_models.py first."
            )
            st.exception(error)

        except Exception as error:
            st.error(
                "An error occurred during model evaluation."
            )
            st.exception(error)


# 16. ALL MODELS COMPARISON


with tab2:

    st.header("Comparison of Five Classification Models")

    metrics_path = os.path.join(
        BASE_DIR,
        "metrics.csv"
    )

    if os.path.exists(metrics_path):

        metrics_df = pd.read_csv(metrics_path)

        st.subheader("Evaluation Metrics Table")

        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}"
            }),
            use_container_width=True
        )

        st.subheader("Performance Comparison Chart")

        chart_df = metrics_df.set_index(
            "ML Model Name"
        )

        fig3, ax3 = plt.subplots(
            figsize=(9, 5.5)
        )

        chart_df.plot(
            kind="bar",
            ax=ax3,
            width=0.8
        )

        ax3.set_title(
            "Classification Model Performance Comparison",
            pad=15
        )

        ax3.set_xlabel("Classification Model")
        ax3.set_ylabel("Metric Score")
        ax3.set_ylim(-0.05, 1.05)

        ax3.tick_params(
            axis="x",
            rotation=20
        )

        for label in ax3.get_xticklabels():
            label.set_ha("right")

        ax3.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            fontsize=9
        )

        ax3.grid(
            axis="y",
            alpha=0.2
        )

        fig3.tight_layout()

        st.pyplot(
            fig3,
            use_container_width=True
        )

        st.download_button(
            "Download Comparison Chart (300 DPI)",
            data=figure_to_png(fig3),
            file_name="model_comparison.png",
            mime="image/png"
        )

        plt.close(fig3)

        st.subheader("Best Performing Model")

        best_model = metrics_df.loc[
            metrics_df["MCC"].idxmax()
        ]

        st.success(
            f"Best Model Based on MCC: "
            f"{best_model['ML Model Name']}"
        )

        best_col1, best_col2, best_col3 = (
            st.columns(3)
        )

        best_col1.metric(
            "Accuracy",
            f"{best_model['Accuracy']:.4f}"
        )

        best_col2.metric(
            "F1 Score",
            f"{best_model['F1']:.4f}"
        )

        best_col3.metric(
            "MCC",
            f"{best_model['MCC']:.4f}"
        )

        st.caption(
            "Comparison metrics are based on the "
            "original held-out test dataset."
        )

    else:
        st.warning(
            "metrics.csv is missing. "
            "Run train_models.py first."
        )


# 17. FOOTER


st.divider()

st.caption(
    "BITS Pilani | M.Tech AI & ML | "
    "Machine Learning Assignment 2"
)