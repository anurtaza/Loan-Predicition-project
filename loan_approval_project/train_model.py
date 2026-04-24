from __future__ import annotations

import importlib
import json
import os
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
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = BASE_DIR / "mlruns"
MLFLOW_EXPERIMENT_NAME = "loan-approval-experiment"
MLFLOW_REGISTERED_MODEL_NAME = "loan-approval-model"


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


def log_to_mlflow(model: Pipeline, metrics: dict, model_params: dict) -> dict:
    mlflow_spec = importlib.util.find_spec("mlflow")
    if mlflow_spec is None:
        return {
            "enabled": False,
            "message": "mlflow is not installed. Install requirements.txt to enable tracking.",
        }

    mlflow = importlib.import_module("mlflow")
    mlflow_sklearn = importlib.import_module("mlflow.sklearn")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
    artifact_location = MLFLOW_ARTIFACTS_DIR.resolve().as_uri()

    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=MLFLOW_EXPERIMENT_NAME,
            artifact_location=artifact_location,
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(experiment_id=experiment_id, run_name="decision-tree-baseline") as run:
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric(
            "f1_score_class_1",
            metrics["classification_report"]["1"]["f1-score"],
        )
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="artifacts")
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="artifacts")

        model_info = mlflow_sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
        )

        latest_versions = client.search_model_versions(
            f"name = '{MLFLOW_REGISTERED_MODEL_NAME}'"
        )
        latest_version = max((int(item.version) for item in latest_versions), default=1)

        return {
            "enabled": True,
            "tracking_uri": tracking_uri,
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
            "registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
            "model_version": latest_version,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
        }


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
    model_params = {
        "model_type": "DecisionTreeClassifier",
        "max_depth": 5,
        "test_size": 0.2,
        "random_state": 42,
    }

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

    mlflow_info = log_to_mlflow(model, metrics, model_params)
    metrics["mlflow"] = mlflow_info
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    if mlflow_info["enabled"]:
        print(
            "MLflow run logged with model "
            f"{mlflow_info['registered_model_name']} v{mlflow_info['model_version']}"
        )
    else:
        print(mlflow_info["message"])
    print(json.dumps({"accuracy": metrics["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
