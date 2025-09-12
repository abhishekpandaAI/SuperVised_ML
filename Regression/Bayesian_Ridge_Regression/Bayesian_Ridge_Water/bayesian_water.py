import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE,"data","water_potability.csv")
MODELS = os.path.join(BASE,"models")
OUTPUTS = os.path.join(BASE,"outputs")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUTS,exist_ok=True)


# 1. Load Data set

df = pd.read_csv(DATA)
df = df.dropna() # drop missing values for simplicity

y = df["Potability"] # 0 = unsafe, 1 = safe
X = df.drop(columns=["Potability"])

# 2. Split the Data

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
 # 3) Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("bayesridge", BayesianRidge())
])
pipeline.fit(X_train, y_train)

# 4) Evaluate
y_pred = pipeline.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

print(f"Validation RMSE: {rmse:.3f}, R²: {r2:.4f}")

# 5) Save model + results
dump(pipeline, os.path.join(MODELS, "bayesian_water.joblib"))
results = {
    "rmse": float(rmse),
    "r2": float(r2),
    "sample_predictions": [
        {"actual": float(a), "predicted": float(p)}
        for a, p in zip(y_valid[:20], y_pred[:20])
    ]
}
with open(os.path.join(OUTPUTS, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

    # 6) Plot
plt.scatter(y_valid, y_pred, alpha=0.6, c='teal')
plt.xlabel("Actual Potability (0=Unsafe, 1=Safe)")
plt.ylabel("Predicted Potability Score")
plt.title("Bayesian Ridge - Water Quality")
plt.savefig(os.path.join(OUTPUTS, "pred_vs_actual.png"))
plt.close()

print("✅ Training complete. Outputs saved in:", OUTPUTS)