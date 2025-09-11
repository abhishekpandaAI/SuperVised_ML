from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Stock & Index Prediction API")

# Load both models
stock_artifact = load("models/stock_model.joblib")
stock_model, stock_features = stock_artifact["model"], stock_artifact["features"]

index_artifact = load("models/index_model.joblib")
index_model, index_features = index_artifact["model"], index_artifact["features"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    Close_lag_1: float
    Close_lag_2: float
    Close_lag_3: float
    MA_5: float
    MA_10: float
    Return_1: float
    DayOfWeek: int
    Month: int

@app.post("/predict_stock")
def predict_stock(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = stock_model.predict(df)[0]
    return {"predicted_close": prediction}

@app.post("/predict_index")
def predict_index(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = index_model.predict(df)[0]
    return {"predicted_close": prediction}
