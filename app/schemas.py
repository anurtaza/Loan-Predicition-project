from __future__ import annotations

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    gender: str = Field(..., examples=["Male"])
    married: str = Field(..., examples=["Yes"])
    dependents: str = Field(..., examples=["0"])
    education: str = Field(..., examples=["Graduate"])
    self_employed: str = Field(..., examples=["No"])
    applicant_income: float = Field(..., ge=0, examples=[5000])
    coapplicant_income: float = Field(..., ge=0, examples=[1500])
    loan_amount: float = Field(..., gt=0, examples=[128])
    loan_amount_term: float = Field(..., gt=0, examples=[360])
    credit_history: float = Field(..., ge=0, le=1, examples=[1])
    property_area: str = Field(..., examples=["Urban"])


class PredictionResponse(BaseModel):
    prediction: str
    prediction_code: str
    approval_probability: float
    rejection_probability: float
