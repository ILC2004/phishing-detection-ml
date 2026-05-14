from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model_utils import build_feature_frame

BASE = Path(__file__).resolve().parent
DATA = BASE / 'final_phishing_dataset.csv'
MODEL = BASE / 'phishing_pipeline.joblib'


def load_data():
    df = pd.read_csv(DATA)
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    return df


def main():
    df = load_data().dropna(subset=['text', 'label']).copy()
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)

    print("Dataset shape:", df.shape)
    print("Class balance:")
    print(df['label'].value_counts())

    X_raw = df['text'].tolist()
    y = df['label']

    feat_df = build_feature_frame(X_raw)

    numeric_cols = [c for c in feat_df.columns if c != 'text']
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(ngram_range=(1, 2), max_features=3000), 'text'),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler(with_mean=False))
            ]), numeric_cols)
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        feat_df, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds, digits=3))
    joblib.dump(model, MODEL)
    print(f"Saved model to {MODEL}")


if __name__ == '__main__':
    main()