import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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
    make_scorer
)

# 1. PROJECT CONFIGURATION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, "data.csv")

print("=" * 65)
print("BITS PILANI - MACHINE LEARNING ASSIGNMENT 2")
print("=" * 65)


# 2. LOAD DATASET


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "data.csv not found. Place it in the project folder."
    )

df = pd.read_csv(DATA_PATH)

print("\nOriginal Dataset Shape:", df.shape)
print("\nFirst Five Records:")
print(df.head().to_string())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum().to_string())


# 3. DATA CLEANING


df = df.drop(
    columns=["id", "Unnamed: 32"],
    errors="ignore"
)

if "diagnosis" not in df.columns:
    raise ValueError(
        "Target column 'diagnosis' is missing."
    )

df["diagnosis"] = df["diagnosis"].replace({
    "B": 0,
    "M": 1
})

df["diagnosis"] = pd.to_numeric(
    df["diagnosis"], errors="raise"
).astype(int)

if not df["diagnosis"].isin([0, 1]).all():
    raise ValueError("Diagnosis must contain B/M or 0/1.")

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X = X.apply(pd.to_numeric, errors="coerce")

if X.shape[1] < 12 or len(X) < 500:
    raise ValueError(
        "Dataset must contain at least 500 rows "
        "and 12 input features."
    )

if y.nunique() != 2:
    raise ValueError("This implementation expects two classes.")

print("\nCleaned Dataset Shape:", df.shape)
print("Number of Features:", X.shape[1])

print("\nTarget Class Distribution:")
print(y.value_counts().to_string())


# 4. EXPLORATORY DATA ANALYSIS


print("\nGenerating EDA charts...")

plt.figure(figsize=(7, 5))

sns.countplot(
    x=y,
    hue=y,
    palette="Set2",
    legend=False
)

plt.title("Diagnosis Class Distribution")
plt.xlabel("Diagnosis (0=Benign, 1=Malignant)")
plt.ylabel("Number of Samples")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "class_distribution.png"),
    dpi=150
)
plt.close()

# Correlation analysis
eda_df = X.copy()
eda_df["diagnosis"] = y

plt.figure(figsize=(16, 12))

sns.heatmap(
    eda_df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
    dpi=150
)
plt.close()

# Top features correlated with target
correlations = (
    eda_df.corr()["diagnosis"]
    .drop("diagnosis")
    .abs()
    .sort_values(ascending=False)
)

print("\nTop 10 Features by Absolute Target Correlation:")
print(correlations.head(10).to_string())


# 5. TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# 6. DEFINE FIVE CLASSIFIERS


models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Naive Bayes": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}


# 7. CROSS VALIDATION AND TRAINING


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "mcc": make_scorer(matthews_corrcoef)
}

cv_records = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_records.append({
        "Model": name,
        "CV Accuracy Mean":
            cv_scores["test_accuracy"].mean(),
        "CV Accuracy Std":
            cv_scores["test_accuracy"].std(),
        "CV F1 Mean":
            cv_scores["test_f1"].mean(),
        "CV MCC Mean":
            cv_scores["test_mcc"].mean()
    })

    model.fit(X_train, y_train)

    print(
        "Mean CV Accuracy:",
        round(cv_scores["test_accuracy"].mean(), 4)
    )

    print(
        "Mean CV MCC:",
        round(cv_scores["test_mcc"].mean(), 4)
    )

cv_df = pd.DataFrame(cv_records)

cv_df.to_csv(
    os.path.join(OUTPUT_DIR, "cross_validation.csv"),
    index=False
)


# 8. MODEL EVALUATION - SIX METRICS


results = []

for name, model in models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    results.append(metrics)

    print("\n" + "-" * 60)
    print("EVALUATION:", name)
    print("-" * 60)

    for metric_name, value in metrics.items():
        if metric_name != "ML Model Name":
            print(f"{metric_name}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        zero_division=0
    ))

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(BASE_DIR, "metrics.csv"),
    index=False
)


# 9. MODEL COMPARISON CHART


plot_df = results_df.set_index("ML Model Name")

ax = plot_df.plot(
    kind="bar",
    figsize=(14, 7),
    width=0.8
)

ax.set_title("Performance Comparison of Five Classifiers")
ax.set_xlabel("Classification Model")
ax.set_ylabel("Metric Score")
ax.set_ylim(0, 1.05)

plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison.png"),
    dpi=150
)
plt.close()


# 10. CONFUSION MATRICES


fig, axes = plt.subplots(
    2, 3,
    figsize=(12, 8)
)

for ax, (name, model) in zip(
    axes.flat,
    models.items()
):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

axes.flat[-1].axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrices.png"),
    dpi=150
)
plt.close()


# 11. ROC CURVE COMPARISON


plt.figure(figsize=(8, 6))

for name, model in models.items():

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    "k--"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_comparison.png"),
    dpi=150
)
plt.close()


# 12. SAVE TRAINED MODELS


print("\nSaving trained models...")

for name, model in models.items():

    model_path = os.path.join(
        MODEL_DIR,
        model_files[name]
    )

    joblib.dump(model, model_path)

    print("Saved:", model_files[name])

joblib.dump(
    X.columns.tolist(),
    os.path.join(
        MODEL_DIR,
        "feature_columns.joblib"
    )
)


# 13. EXPORT TEST DATASET


test_data = X_test.copy()
test_data["diagnosis"] = y_test

test_data.to_csv(
    os.path.join(BASE_DIR, "test_data.csv"),
    index=False
)

print("\nTest dataset exported successfully.")
print("Test dataset shape:", test_data.shape)


# 14. BEST MODEL SELECTION


best_row = results_df.loc[
    results_df["MCC"].idxmax()
]

print("\n" + "=" * 60)
print("BEST MODEL BASED ON MCC")
print("=" * 60)

print("Model:", best_row["ML Model Name"])
print("Accuracy:", round(best_row["Accuracy"], 4))
print("AUC:", round(best_row["AUC"], 4))
print("F1:", round(best_row["F1"], 4))
print("MCC:", round(best_row["MCC"], 4))

print("\nAssignment training pipeline completed!")
print("Generated files are available in the project folder.")