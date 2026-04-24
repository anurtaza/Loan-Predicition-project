# Loan Approval Prediction Project

This project predicts whether a loan application should be approved or rejected using the Loan Prediction Dataset.

It now extends the original Practical Task 6 into a more complete ML system by adding:

- a `Streamlit` frontend for user interaction
- `MLflow` experiment tracking
- `MLflow` model registration with model name and versioning

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
├── frontend/
│   └── streamlit_app.py
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
- provides a Streamlit frontend for manual user input
- tracks experiments in MLflow
- registers trained models in MLflow Model Registry
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

2. Install the new SIS dependencies:

```bash
pip install streamlit mlflow
```

3. Train the model:

```bash
python train_model.py
```

The training script now:

- logs model parameters
- logs evaluation metrics such as accuracy and F1-score
- logs model artifacts
- registers the model in MLflow as `loan-approval-model`

4. Start the API:

```bash
uvicorn app.main:app --reload
```

5. Start the Streamlit frontend:

```bash
streamlit run frontend/streamlit_app.py
```

6. Open the interfaces:

- FastAPI docs: `http://127.0.0.1:8000/docs`
- Streamlit frontend: `http://localhost:8501`

## MLflow

Run the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

`http://127.0.0.1:5000`

You will see:

- experiment runs
- logged parameters
- evaluation metrics
- saved artifacts
- registered model versions

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

The current Dockerfile starts the FastAPI API. For the SIS extension, Streamlit and MLflow are also included in `requirements.txt`, so you can run them locally after dependency installation.

Run the full system with Docker Compose:

```bash
docker compose up --build
```

Services:

- FastAPI API: `http://127.0.0.1:8000`
- Streamlit frontend: `http://127.0.0.1:8501`
- MLflow UI: `http://127.0.0.1:5000`

## Notes

- The training pipeline handles missing values automatically.
- Categorical features are encoded with `OneHotEncoder`.
- Numeric features are imputed before prediction.
- MLflow integration is built into the training script.
- The Streamlit frontend sends prediction requests to FastAPI.
