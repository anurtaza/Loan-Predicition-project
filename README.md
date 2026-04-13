# Loan Approval Prediction Project

This project predicts whether a loan application should be approved or rejected using the Loan Prediction Dataset.

## Project structure

```text
loan_approval_project/
├── app/
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
├── artifacts/
├── data/
│   └── train.csv
├── tests/
│   └── test_api.py
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── train_model.py
```

## What the project does

- trains a machine learning model on historical loan data
- predicts loan approval through a FastAPI endpoint
- returns both the decision and prediction probabilities
- can be run locally or inside Docker

## Dataset features

- `Gender`
- `Married`
- `Dependents`
- `Education`
- `Self_Employed`
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`
- `Property_Area`

Target:

- `Loan_Status` where `Y` means approved and `N` means rejected

## Run locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python train_model.py
```

3. Start the API:

```bash
uvicorn app.main:app --reload
```

4. Open Swagger UI:

`http://127.0.0.1:8000/docs`

## API endpoint

### `POST /predict`

Example request:

```json
{
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
  "property_area": "Rural"
}
```

Example response:

```json
{
  "prediction": "Approved",
  "prediction_code": "Y",
  "approval_probability": 0.79,
  "rejection_probability": 0.21
}
```

## Run tests

```bash
pytest
```

## Docker

Build image:

```bash
docker build -t loan-approval-api .
```

Run container:

```bash
docker run -p 8000:8000 loan-approval-api
```

## Notes

- The training pipeline handles missing values automatically.
- Categorical features are encoded with `OneHotEncoder`.
- Numeric features are imputed and scaled before prediction.
