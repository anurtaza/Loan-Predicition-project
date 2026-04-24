from __future__ import annotations

from pathlib import Path

import json
import pandas as pd
from fastapi import FastAPI

from app.model_loader import load_model
from app.schemas import LoanApplication, PredictionResponse


BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

app = FastAPI(
    title="Loan Approval Prediction API",
    description="API that predicts whether a loan application will be approved or rejected.",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {
        "message": "Loan Approval Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    return {
        "status": "ok",
        "model_loaded": (BASE_DIR / "artifacts" / "loan_approval_model.joblib").exists(),
        "accuracy": metrics.get("accuracy"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_loan_status(application: LoanApplication):
    model = load_model()

    input_frame = pd.DataFrame(
        [
            {
                "Gender": application.gender,
                "Married": application.married,
                "Dependents": application.dependents,
                "Education": application.education,
                "Self_Employed": application.self_employed,
                "ApplicantIncome": application.applicant_income,
                "CoapplicantIncome": application.coapplicant_income,
                "LoanAmount": application.loan_amount,
                "Loan_Amount_Term": application.loan_amount_term,
                "Credit_History": application.credit_history,
                "Property_Area": application.property_area,
            }
        ]
    )

    prediction_code = int(model.predict(input_frame)[0])
    probabilities = model.predict_proba(input_frame)[0]

    approved_probability = float(probabilities[1])
    rejected_probability = float(probabilities[0])

    return PredictionResponse(
        prediction="Approved" if prediction_code == 1 else "Rejected",
        prediction_code="Y" if prediction_code == 1 else "N",
        approval_probability=round(approved_probability, 4),
        rejection_probability=round(rejected_probability, 4),
    )
