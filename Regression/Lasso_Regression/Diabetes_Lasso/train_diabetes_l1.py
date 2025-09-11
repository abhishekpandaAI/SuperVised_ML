# train_diabetes_l1.py
import os, json
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "diabetes.csv")
MODELS_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load data
df = pd.read_csv(DATA_PATH)
print("Loaded shape:", df.shape)
print(df.columns.tolist())

# 2) Prepare X, y
target_col = "Outcome"
X = df.drop(columns=[target_col])
y = df[target_col]

feature_names = X.columns.tolist()

# 3) Train/test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4) Pipeline: imputer -> scaler -> logistic (L1)
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l1", solver="saga", max_iter=5000, tol=1e-4))
])

# 5) Hyperparameter search (C = inverse of regularization strength)
param_grid = {"clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
best_C = grid.best_params_["clf__C"]
print("Best C:", best_C)

# 6) Evaluate
y_pred = best.predict(X_valid)
y_proba = best.predict_proba(X_valid)[:,1]
acc = accuracy_score(y_valid, y_pred)
auc = roc_auc_score(y_valid, y_proba)
cm = confusion_matrix(y_valid, y_pred)
print(f"Validation Accuracy: {acc:.4f}, AUC: {auc:.4f}")
print("Confusion matrix:\n", cm)

# 7) Extract coefficients (after scaling -> coefficients correspond to features)
clf = best.named_steps["clf"]
# NOTE: pipeline transforms, but coefficients map to original feature order (imputer + scaler don't change columns)
coefs = clf.coef_.ravel()  # array length = number of features
coef_series = pd.Series(coefs, index=feature_names).sort_values(key=lambda x: x.abs(), ascending=False)
print("Top coefficients:\n", coef_series.head(10))

# 8) Save model bundle (model + feature names + metrics + selected features)
bundle = {
    "model": best,
    "feature_names": feature_names,
    "metrics": {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "best_C": float(best_C)
    },
    "coefficients": coef_series.to_dict()
}
dump(bundle, os.path.join(MODELS_DIR, "diabetes_logreg_l1.joblib"))

# 9) Save outputs JSON for frontend visualization
results = {
    "accuracy": float(acc),
    "roc_auc": float(auc),
    "best_C": float(best_C),
    "coefficients": {k: float(v) for k, v in coef_series.items()}
}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Saved model and outputs.")
