# decision_tree_loan.py
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data/loan_risk.csv")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) Load Data
# -----------------------------
df = pd.read_csv(DATA)
print("Data shape:", df.shape)

# Select only the fields we want for frontend
selected_features = [
    "Age", "AnnualIncome", "CreditScore", "EmploymentStatus", 
    "EducationLevel", "LoanAmount", "LoanDuration", "MaritalStatus", 
    "NumberOfDependents", "DebtToIncomeRatio", 
    "PreviousLoanDefaults", "NetWorth"
]

X = df[selected_features]
y = df["RiskScore"]

# -----------------------------
# 2) Preprocessing
# -----------------------------
numeric = ["Age", "AnnualIncome", "CreditScore", "LoanAmount",
           "LoanDuration", "NumberOfDependents", "DebtToIncomeRatio", "NetWorth"]
categorical = ["EmploymentStatus", "EducationLevel", "MaritalStatus"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

# -----------------------------
# 3) Model Training
# -----------------------------
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("tree", DecisionTreeRegressor(random_state=42))
])

params = {
    "tree__max_depth": [3, 5, 8, None],
    "tree__min_samples_leaf": [1, 5, 10]
}

grid = GridSearchCV(pipe, param_grid=params, cv=5,
                    scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
grid.fit(X, y)

best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)

# -----------------------------
# 4) Evaluation
# -----------------------------
y_pred = best_model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
print(f"RMSE: {rmse:.2f}, R²: {r2:.3f}")

# -----------------------------
# 5) Save Model & Results
# -----------------------------
dump(best_model, os.path.join(MODEL_DIR, "decision_tree_loan.joblib"))

results = {
    "rmse": float(rmse),
    "r2": float(r2),
    "best_params": grid.best_params_,
}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# -----------------------------
# 6) Visualize Tree
# -----------------------------
plt.figure(figsize=(18, 8))
plot_tree(best_model.named_steps["tree"], 
          feature_names=best_model.named_steps["preprocessor"].get_feature_names_out(),
          filled=True, max_depth=3, fontsize=8)
plt.title("Decision Tree - Loan Risk")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tree_visual.png"))
plt.close()

print("✅ Training complete. Model & outputs saved.")
