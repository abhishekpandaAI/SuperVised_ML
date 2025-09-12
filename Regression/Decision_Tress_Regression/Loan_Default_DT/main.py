import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
from sklearn.tree import export_text

# Load model
model = joblib.load("models/decision_tree_loan.joblib")

app = FastAPI(title="Loan Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def read_root():
    return FileResponse(os.path.join("frontend", "index.html"))

# -------------------------
# Input schema
# -------------------------
class LoanData(BaseModel):
    Age: int
    AnnualIncome: float
    CreditScore: float
    EmploymentStatus: str
    EducationLevel: str
    LoanAmount: float
    LoanDuration: int
    MaritalStatus: str
    NumberOfDependents: int
    DebtToIncomeRatio: float
    PreviousLoanDefaults: int
    NetWorth: float

# -------------------------
# 1) Predict endpoint
# -------------------------
@app.post("/predict")
def predict(data: LoanData):
    df = pd.DataFrame([data.dict()])
    score = model.predict(df)[0]

    if score < 30:
        status = "Low Risk ✅"
    elif score < 60:
        status = "Medium Risk ⚠️"
    else:
        status = "High Risk ❌"

    return {"predicted_score": round(float(score), 2), "status": status}

# -------------------------
# 2) Batch predictions
# -------------------------
@app.post("/predict_batch")
def predict_batch(data: list[LoanData]):
    df = pd.DataFrame([d.dict() for d in data])
    scores = model.predict(df)

    results = []
    for i, s in enumerate(scores):
        if s < 30:
            status = "Low Risk ✅"
        elif s < 60:
            status = "Medium Risk ⚠️"
        else:
            status = "High Risk ❌"
        results.append({"id": i, "predicted_score": round(float(s), 2), "status": status})

    return {"results": results}

# -------------------------
# 3) Model info
# -------------------------
@app.get("/model_info")
def model_info():
    return {
        "type": str(type(model.named_steps['tree']).__name__),
        "parameters": {k: str(v) for k, v in model.named_steps['tree'].get_params().items()}
    }

# -------------------------
# 4) Feature importance
# -------------------------
@app.get("/feature_importance")
def feature_importance():
    tree_model = model.named_steps["tree"]
    importance = tree_model.feature_importances_

    # Get feature names from preprocessor
    features = model.named_steps["preprocessor"].get_feature_names_out()

    return {
        "importance": [
            {"feature": f, "score": round(float(s), 4)}
            for f, s in sorted(zip(features, importance), key=lambda x: -x[1])
        ]
    }

# -------------------------
# 5) Explain rules
# -------------------------
@app.get("/explain")
def explain_model():
    try:
        rules = export_text(model.named_steps["tree"],
                            feature_names=model.named_steps["preprocessor"].get_feature_names_out(),
                            max_depth=3)
    except Exception as e:
        rules = str(e)
    return {"rules": rules}

# -------------------------
# 6) Health check
# -------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "model": "DecisionTreeRegressor"}
