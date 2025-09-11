import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model bundle
model_bundle = joblib.load("models/ridge_house.joblib")
model = model_bundle["model"]
numeric_features = model_bundle["numeric_features"]
categorical_features = model_bundle["categorical_features"]

app = FastAPI(title="House Price Prediction API")
# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"] if serving frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str

@app.post("/predict")
def predict_price(data: HouseData):
    df = pd.DataFrame([data.dict()])  # convert to DataFrame
    prediction = model.predict(df)[0]  # pipeline handles preprocessing
    return {"predicted_price": round(prediction, 2)}

@app.get("/feature-importance")
def feature_importance():
    # Load feature coefficients from saved JSON
    results_path = os.path.join("outputs", "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)
    return results["feature_coeffs"]
