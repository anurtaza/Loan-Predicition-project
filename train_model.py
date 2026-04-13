from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "loan_approval_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def build_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]
    )


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Loan_ID"])
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline(categorical_features, numeric_features)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "test_size": len(X_test),
        "train_size": len(X_train),
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "classification_report": report,
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(json.dumps({"accuracy": metrics["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
