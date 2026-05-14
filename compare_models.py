from pathlib import Path
import time
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from model_utils import build_feature_frame


BASE = Path(__file__).resolve().parent
DATA = BASE / "final_phishing_dataset.csv"
OUTPUT = BASE / "model_comparison_results.csv"


def load_data():
    df = pd.read_csv(DATA)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def build_preprocessor(feature_df):
    numeric_cols = [c for c in feature_df.columns if c != "text"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=3000), "text"),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_cols,
            ),
        ]
    )

    return preprocessor


def evaluate(model_name, pipeline, X_train, X_test, y_train, y_test):
    start = time.time()

    pipeline.fit(X_train, y_train)

    train_time = time.time() - start

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_scores = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_scores = pipeline.decision_function(X_test)
    else:
        y_scores = None

    results = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-score": round(f1_score(y_test, y_pred), 4),
        "Training Time (s)": round(train_time, 2),
    }

    if y_scores is not None:
        results["ROC-AUC"] = round(roc_auc_score(y_test, y_scores), 4)
    else:
        results["ROC-AUC"] = "N/A"

    return results


def main():
    df = load_data()

    print("Dataset shape:", df.shape)
    print("Class balance:")
    print(df["label"].value_counts())
    print()

    X_raw = df["text"].tolist()
    y = df["label"]

    feature_df = build_feature_frame(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(feature_df)

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=2000)),
            ]
        ),
        "Linear SVM": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(max_iter=5000),
                        cv=3,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    results = []

    for name, model in models.items():
        print("Training:", name)
        res = evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1-score", ascending=False)

    print("\nModel Comparison Results\n")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUTPUT, index=False)

    print("\nSaved comparison table to:", OUTPUT)


if __name__ == "__main__":
    main()