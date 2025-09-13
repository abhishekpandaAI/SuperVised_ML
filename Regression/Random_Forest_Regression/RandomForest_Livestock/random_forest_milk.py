# random_forest_milk.py
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data", "farm_milk_production.csv")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA)
print("Data shape:", df.shape)
print(df.head())

# Target
y = df["Milk_Liters"]

# Features
X = df[["Feed_kg", "Temp_C", "Humidity", "Milking_Time_min", "Cow_ID"]]

# 2. Preprocess
numeric_features = ["Feed_kg", "Temp_C", "Humidity", "Milking_Time_min"]
categorical_features = ["Cow_ID"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# 3. Pipeline with RandomForest
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))
])

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Hyperparameter search
param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [8, 12, None],
    "rf__min_samples_leaf": [1, 5, 10]
}
grid = GridSearchCV(pipe, param_grid, cv=4,
                    scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)

# 6. Evaluate
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R²: {r2:.3f}")

# 7. Save model & results
dump({
    "model": best_model,
    "features": X.columns.tolist()
}, os.path.join(MODEL_DIR, "random_forest_milk.joblib"))

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump({"rmse": rmse, "r2": r2, "best_params": grid.best_params_}, f, indent=2)

# 8. Plot predictions
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--", color="gray")
plt.xlabel("Actual Milk Yield (Liters)")
plt.ylabel("Predicted Milk Yield (Liters)")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pred_vs_actual.png"))
plt.close()

# 9. Feature importance
rf = best_model.named_steps["rf"]
feat_names = best_model.named_steps["preprocessor"].get_feature_names_out()
importances = rf.feature_importances_
feat_df = pd.DataFrame({"feature": feat_names, "importance": importances})
feat_df = feat_df.sort_values("importance", ascending=False)
feat_df.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

plt.figure(figsize=(10,6))
plt.bar(feat_df["feature"], feat_df["importance"])
plt.xticks(rotation=45, ha="right")
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"))
plt.close()

print("✅ Training complete. Model & outputs saved.")
