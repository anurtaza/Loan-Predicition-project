from __future__ import annotations

import json
import os
from urllib import error, request

import streamlit as st


DEFAULT_API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{DEFAULT_API_BASE_URL}/predict"
HEALTH_URL = f"{DEFAULT_API_BASE_URL}/health"


def fetch_health() -> dict:
    try:
        with request.urlopen(HEALTH_URL, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def predict(payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        PREDICT_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


st.set_page_config(page_title="Loan Approval Frontend", page_icon="💳", layout="centered")
st.title("Loan Approval Prediction")
st.caption("Streamlit frontend for the FastAPI loan approval service")

health = fetch_health()
if health:
    st.success(
        f"API is online. Current model accuracy: {health.get('accuracy', 'unknown')}"
    )
else:
    st.warning("API is not reachable yet. Start FastAPI before making predictions.")

with st.form("loan_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=1.0, value=128.0)
    loan_amount_term = st.number_input("Loan Amount Term", min_value=1.0, value=360.0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "gender": gender,
        "married": married,
        "dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income,
        "loan_amount": loan_amount,
        "loan_amount_term": loan_amount_term,
        "credit_history": credit_history,
        "property_area": property_area,
    }

    try:
        result = predict(payload)
        st.subheader("Prediction Result")
        st.write(f"Decision: **{result['prediction']}**")
        st.write(f"Prediction Code: `{result['prediction_code']}`")
        st.progress(int(result["approval_probability"] * 100))
        st.write(f"Approval probability: `{result['approval_probability']}`")
        st.write(f"Rejection probability: `{result['rejection_probability']}`")
    except error.URLError as exc:
        st.error(f"Could not connect to FastAPI service: {exc}")
