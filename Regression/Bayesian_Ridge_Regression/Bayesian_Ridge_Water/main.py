from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib, os, json

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "bayesian_water.joblib")
OUTPUTS = os.path.join(BASE, "outputs", "results.json")

# Load model
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Water Quality Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Input schema
class WaterData(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

def check_who_limits(data: dict):
    reasons = []
    safe = True
    if not (6.5 <= data['ph'] <= 8.5):
        reasons.append("pH out of range (6.5–8.5)")
        safe = False
    if data['Hardness'] > 300:
        reasons.append("Hardness too high")
        safe = False
    if data['Solids'] > 500:
        reasons.append("Solids exceed 500 mg/L")
        safe = False
    if data['Chloramines'] > 4:
        reasons.append("Chloramines exceed safe level")
        safe = False
    if data['Sulfate'] > 250:
        reasons.append("Sulfate too high")
        safe = False
    if not (50 <= data['Conductivity'] <= 500):
        reasons.append("Conductivity out of safe range")
        safe = False
    if data['Organic_carbon'] > 5:
        reasons.append("Organic carbon too high")
        safe = False
    if data['Trihalomethanes'] > 80:
        reasons.append("Trihalomethanes exceed safe level")
        safe = False
    if data['Turbidity'] > 5:
        reasons.append("Turbidity too high")
        safe = False
    return safe, reasons


@app.post("/predict")
def predict_quality(data: WaterData):
    df = pd.DataFrame([data.dict()])

    # WHO rule check
    who_safe, who_reasons = check_who_limits(data.dict())

    # ML model score
    score = float(model.predict(df)[0])
    status = "Safe 🚰" if score >= 0.5 else "Unsafe ⚠️"

    # Override with WHO if everything safe
    if who_safe:
        return {"predicted_score": score, "status": "Safe 🚰", "reasons": ["All key water quality indicators are within safe WHO limits."]}
    else:
        return {"predicted_score": score, "status": status, "reasons": who_reasons}


@app.get("/metrics")
def metrics():
    with open(OUTPUTS, "r") as f:
        return json.load(f)