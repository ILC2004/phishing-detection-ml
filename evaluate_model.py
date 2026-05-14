from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from model_utils import build_feature_frame

BASE = Path(__file__).resolve().parent
DATA = BASE / "final_phishing_dataset.csv"
MODEL = BASE / "phishing_pipeline.joblib"
OUTPUTS = BASE / "evaluation_outputs"


def load_data() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def load_model():
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}. Run train_model.py first.")
    return joblib.load(MODEL)


def save_text_report(report_text: str, metrics: dict):
    OUTPUTS.mkdir(exist_ok=True)

    report_path = OUTPUTS / "classification_report.txt"
    metrics_path = OUTPUTS / "metrics.json"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved report to: {report_path}")
    print(f"Saved metrics to: {metrics_path}")


def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Phishing"])

    plt.figure(figsize=(6, 5))
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    out = OUTPUTS / "confusion_matrix.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to: {out}")


def save_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out = OUTPUTS / "roc_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve to: {out}")


def save_precision_recall_curve(y_test, y_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()

    out = OUTPUTS / "precision_recall_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Precision-Recall curve to: {out}")


def main():
    OUTPUTS.mkdir(exist_ok=True)

    df = load_data()
    model = load_model()

    print("Dataset shape:", df.shape)
    print("Class balance:")
    print(df["label"].value_counts())

    X_raw = df["text"].tolist()
    y = df["label"]

    feat_df = build_feature_frame(X_raw)

    # Rebuild the exact same split used in train_model.py
    _, X_test, _, y_test = train_test_split(
        feat_df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report_text = classification_report(y_test, y_pred, digits=3)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "dataset_rows": int(len(df)),
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "class_balance": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
    }

    print("\nClassification Report:\n")
    print(report_text)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC:  {roc_auc:.4f}")

    save_text_report(report_text, metrics)
    save_confusion_matrix(y_test, y_pred)
    save_roc_curve(y_test, y_proba)
    save_precision_recall_curve(y_test, y_proba)

    print("\nDone. Check the 'evaluation_outputs' folder.")


if __name__ == "__main__":
    main()