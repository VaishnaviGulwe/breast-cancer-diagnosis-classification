# Machine Learning Assignment 2
## Breast Cancer Classification Using Five Machine Learning Models

**BITS Pilani – M.Tech Artificial Intelligence and Machine Learning (WILP)**

**GitHub Repository:** [Add your GitHub repository URL]  
**Live Streamlit App:** [Add your Streamlit deployment URL]

## a. Problem Statement

This project classifies breast tumors as **Benign (B)** or **Malignant (M)** using 30 numerical measurements of cell nuclei from the Breast Cancer Wisconsin Diagnostic dataset. Five supervised learning algorithms—Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM)—were trained and compared using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). An interactive Streamlit application evaluates the saved models without retraining them.

## b. Dataset Description

| Item | Description |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic) |
| Source | Kaggle / UCI Machine Learning Repository |
| Total records | 569 |
| Input features | 30 numerical features |
| Target | `diagnosis` (Benign / Malignant) |
| Training data | 455 records (80%) |
| Testing data | 114 records (20%) |
| Split | Stratified, `random_state=42` |

The features include radius, texture, perimeter, area, smoothness, and concavity. The `id` and unnecessary unnamed columns were removed, and diagnosis was encoded as Benign = 0 and Malignant = 1. The model pipelines use median imputation; StandardScaler is applied to Logistic Regression, KNN, Naive Bayes, and SVM. Five-fold stratified cross-validation was performed on the training data. Malignant is the positive class for Precision, Recall, and F1.

## c. GitHub Repository Link

**Repository:** [Insert your GitHub repository URL]

| File / folder | Purpose |
|---|---|
| `train_models.py` | Preprocessing, model training, cross-validation, and evaluation |
| `app.py` | Streamlit dashboard |
| `data.csv` | Original dataset |
| `test_data.csv` | Held-out test data |
| `metrics.csv` | Six evaluation metrics for each model |
| `model/` | Saved `.joblib` models and feature-column list |
| `outputs/` | Confusion matrices, ROC curves, and comparison charts |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

**Run locally on Linux:**

```bash
python3 -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

Open `http://localhost:8501` and upload `test_data.csv` to evaluate a selected model.

## d. Models Used and Performance Comparison

Results below are from the **114 held-out test observations**, as displayed in the Streamlit model-comparison table.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | **0.9960** | 0.9750 | **0.9286** | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| **SVM** | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

**Model observations:**

- **Logistic Regression:** Highest AUC (0.9960), with strong accuracy (96.49%) and recall (92.86%).
- **Decision Tree:** Accuracy of 92.11%, but the lowest recall (83.33%) among the five models.
- **KNN:** Accuracy of 95.61% and MCC of 0.9058, showing strong classification performance.
- **Naive Bayes:** Accuracy of 92.11% and AUC of 0.9891, although its MCC was lower than SVM's.
- **SVM:** Highest Accuracy (97.37%), Precision (100%), F1 (0.9630), and MCC (0.9442).

**Overall winner: Support Vector Machine (SVM).** It performed best on four of the six metrics. Logistic Regression achieved the highest AUC, while SVM and Logistic Regression tied for highest Recall.

## e. Streamlit Application

**Live App:** [Insert your deployed Streamlit URL]

The dashboard has two sections:

- **Model Evaluation:** Upload `test_data.csv`, select a trained classifier, view six metrics, a confusion matrix, ROC curve, and classification report, and download predictions or charts.
- **All Models Comparison:** Compare the five classifiers in a metrics table and performance chart.

The application loads the saved `.joblib` pipelines from `model/` and uses the saved feature-column list. Uploading a CSV performs inference and evaluation; it does **not** retrain the models.

## Conclusion

Five classifiers were trained and evaluated on the Breast Cancer Wisconsin Diagnostic dataset. **SVM performed best overall**, achieving 97.37% accuracy, 100% precision, and an MCC of 0.9442. Logistic Regression had the highest AUC at 0.9960. The Streamlit application demonstrates reusable model deployment and interactive evaluation. This project is for academic comparison, not clinical diagnosis.

## References

1. [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
2. [Kaggle Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
3. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
4. [Streamlit Documentation](https://docs.streamlit.io/)
