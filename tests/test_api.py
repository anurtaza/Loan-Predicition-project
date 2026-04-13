from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_endpoint():
    payload = {
        "gender": "Male",
        "married": "Yes",
        "dependents": "1",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 4583,
        "coapplicant_income": 1508,
        "loan_amount": 128,
        "loan_amount_term": 360,
        "credit_history": 1,
        "property_area": "Rural",
    }
    response = client.post("/predict", json=payload)
    body = response.json()

    assert response.status_code == 200
    assert body["prediction_code"] in {"Y", "N"}
    assert 0 <= body["approval_probability"] <= 1
