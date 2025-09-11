# elastic_insurance.py
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "insurance.csv")
MODELS = os.path.join(BASE, "models")
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

# 1) Load data
df = pd.read_csv(DATA)
print("Shape:", df.shape)
print(df.head())

y = df["charges"]
X = df.drop(columns=["charges"])

numeric = ["age", "bmi", "children"]
categorical = ["sex", "smoker", "region"]

# 2) Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

# 3) Elastic Net pipeline
pipeline = Pipeline([
    ("preproc", preprocessor),
    ("elastic", ElasticNet(max_iter=5000))
])

# 4) Grid Search
param_grid = {
    "elastic__alpha": [0.01, 0.1, 1, 10],
    "elastic__l1_ratio": [0.2, 0.5, 0.8]  # 0=Ridge, 1=Lasso
}
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
best_params = grid.best_params_
print("Best params:", best_params)

# 5) Evaluate
y_pred = best.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)
print(f"Validation RMSE: {rmse:.2f}, R2: {r2:.4f}")

# 6) Save model
dump(best, os.path.join(MODELS, "elastic_insurance.joblib"))

# 7) Save results
results = {
    "rmse": float(rmse),
    "r2": float(r2),
    "best_params": best_params
}
with open(os.path.join(OUTPUTS, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# 8) Feature importance
elastic = best.named_steps["elastic"]
preproc = best.named_steps["preproc"]
feat_names = preproc.get_feature_names_out()
coef = pd.Series(elastic.coef_, index=feat_names).sort_values(key=lambda x: abs(x), ascending=False)

plt.figure(figsize=(8,6))
coef.head(20).sort_values().plot(kind="barh", color="teal")
plt.title("Elastic Net Feature Importance - Insurance Charges")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "feature_importance.png"))
plt.close()

print("✅ Model + outputs saved")
