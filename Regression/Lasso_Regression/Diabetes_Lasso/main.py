# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib, os, json

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "diabetes_logreg_l1.joblib")
OUTPUTS = os.path.join(BASE, "outputs", "results.json")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURE_NAMES = bundle["feature_names"]

app = FastAPI(title="Diabetes Classifier (L1 Logistic)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema matching feature names
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
def predict(data: DiabetesInput):
    df = pd.DataFrame([data.dict()])[FEATURE_NAMES]
    proba = model.predict_proba(df)[0, 1]
    
    if proba < 0.3:
        label = "Not Diabetic"
    elif proba < 0.6:
        label = "Prediabetic"
    else:
        label = "Diabetic"
    
    return {
        "predicted_class": label,
        "predicted_proba": round(float(proba), 3)
    }

@app.get("/feature-importance")
def feature_importance():
    # returns coefficients saved in outputs/results.json
    if os.path.exists(OUTPUTS):
        with open(OUTPUTS, "r") as f:
            results = json.load(f)
        return results.get("coefficients", {})
    # fallback: read from bundle
    coefs = bundle.get("coefficients", {})
    return coefs
