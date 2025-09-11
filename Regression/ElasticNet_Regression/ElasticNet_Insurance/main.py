from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib, os, json

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "elastic_insurance.joblib")
OUTPUTS = os.path.join(BASE, "outputs", "results.json")

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Medical Insurance Cost Prediction API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Input schema
class InsuranceData(BaseModel):
    age: float
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.post("/predict")
def predict_cost(data: InsuranceData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    return {"predicted_charges": round(float(pred), 2)}

@app.get("/metrics")
def metrics():
    with open(OUTPUTS, "r") as f:
        return json.load(f)

@app.get("/feature-importance")
def feature_importance():
    """Return feature names and coefficients from ElasticNet."""
    elastic = model.named_steps["elastic"]
    preproc = model.named_steps["preproc"]
    feature_names = preproc.get_feature_names_out()
    coefs = elastic.coef_

    result = [{"feature": f, "coefficient": round(float(c), 4)} for f, c in zip(feature_names, coefs)]
    # Sort by absolute coefficient size
    result = sorted(result, key=lambda x: abs(x["coefficient"]), reverse=True)
    return result