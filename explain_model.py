from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "phishing_pipeline.joblib"

def main():
    model = joblib.load(MODEL_PATH)

    # Access pipeline parts
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Extract TF-IDF vectorizer
    vectorizer = preprocessor.named_transformers_["text"]

    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    # Sort by weight
    top_phishing_idx = np.argsort(coefficients)[-20:]
    top_legit_idx = np.argsort(coefficients)[:20]

    print("\nTop indicators of PHISHING:\n")
    for i in reversed(top_phishing_idx):
        print(f"{feature_names[i]:20s} {coefficients[i]:.4f}")

    print("\nTop indicators of LEGITIMATE email:\n")
    for i in top_legit_idx:
        print(f"{feature_names[i]:20s} {coefficients[i]:.4f}")

    # Save to CSV for dissertation
    df = pd.DataFrame({
        "feature": feature_names,
        "weight": coefficients
    })

    df_sorted = df.sort_values("weight", ascending=False)

    out = BASE / "top_model_features.csv"
    df_sorted.to_csv(out, index=False)

    print(f"\nSaved full feature weights to {out}")

if __name__ == "__main__":
    main()