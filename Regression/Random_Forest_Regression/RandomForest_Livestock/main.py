from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import os, json

# Load model
model_bundle = joblib.load("models/random_forest_milk.joblib")
model = model_bundle["model"]
features = model_bundle["features"]

# Load dataset for summaries (optional: use cached dataset)
DATA_PATH = os.path.join("data", "farm_milk_production.csv")
df_data = pd.read_csv(DATA_PATH)

app = FastAPI(title="Gocarin Milk Yield Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/assets", StaticFiles(directory="frontend/assets"), name="assets")



@app.get("/")
def read_root():
    return FileResponse(os.path.join("frontend", "index.html"))

# --- Input schema ---
class CowData(BaseModel):
    Cow_ID: str
    Feed_kg: float
    Temp_C: float
    Humidity: float
    Milking_Time_min: float

# --- Farmer prediction ---
@app.post("/predict")
def predict(data: CowData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df[features])[0]
    return {
        "cow": data.Cow_ID,
        "predicted_yield": round(float(pred), 2),
        "unit": "Liters"
    }

# --- Feature importance ---
@app.get("/feature_importance")
def feature_importance():
    # If the model is a pipeline, get the last step (rf)
    tree_model = model
    if hasattr(model, "named_steps"):
        tree_model = model.named_steps.get("rf", model)

    if not hasattr(tree_model, "feature_importances_"):
        return {"error": "Feature importances not available"}

    importance = tree_model.feature_importances_

    # Get feature names from preprocessor
    try:
        preproc = model.named_steps["preprocessor"]
        features = preproc.get_feature_names_out()
    except Exception:
        features = [f"feature_{i}" for i in range(len(importance))]

    return {
        "importance": [
            {"feature": f, "score": round(float(s), 4)}
            for f, s in sorted(zip(features, importance), key=lambda x: -x[1])
        ]
    }


# --- Trends ---
@app.get("/trends")
def trends():
    grouped = df_data.groupby(["Date", "Cow_ID"])["Milk_Liters"].mean().reset_index()
    dates = sorted(df_data["Date"].unique())
    cows = []
    for cow in df_data["Cow_ID"].unique()[:5]:  # limit to first 5 cows for clarity
        subset = grouped[grouped["Cow_ID"] == cow]
        yields = [subset[subset["Date"] == d]["Milk_Liters"].mean() if d in subset["Date"].values else None for d in dates]
        cows.append({"id": cow, "yields": [round(y,2) if y else None for y in yields]})
    return {"dates": dates, "cows": cows}

# --- CEO Summary ---
@app.get("/ceo_summary")
def ceo_summary():
    avg_yield = round(df_data["Milk_Liters"].mean(), 2)

    # Best cow (highest avg yield)
    cow_means = df_data.groupby("Cow_ID")["Milk_Liters"].mean()
    best_cow_id = cow_means.idxmax()
    best_cow_yield = round(cow_means.max(), 2)

    # Underperforming cows (below 15L)
    low_cows = cow_means[cow_means < 15].index.tolist()

    cows_list = [
        {"id": cow, "yield": round(y, 2)} for cow, y in cow_means.items()
    ]

    return {
        "avg_yield": avg_yield,
        "best_cow": {"id": best_cow_id, "yield": best_cow_yield},
        "low_cows": low_cows,
        "cows": cows_list
    }
