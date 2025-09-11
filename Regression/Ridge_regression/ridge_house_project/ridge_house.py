# ridge_house.py
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load dataset
df = pd.read_csv(os.path.join(DATA_DIR, "Housing.csv"))
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2) Target
y = df['price']
X = df.drop(columns=['price'])

# 3) Identify numeric & categorical
numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_features = [c for c in X.columns if c not in numeric_features]

# 4) Preprocessing
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 5) Ridge Regression pipeline
pipeline = Pipeline([
    ('preproc', preprocessor),
    ('ridge', Ridge())
])

# 6) Split & train
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'ridge__alpha': [0.1, 1, 10, 50, 100]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_alpha = grid.best_params_['ridge__alpha']
print("Best alpha:", best_alpha)

# 7) Evaluate
y_pred = best_model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)
print(f"Validation RMSE: {rmse:.2f}, R2: {r2:.4f}")

# 8) Save model + metadata
model_bundle = {
    "model": best_model,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "best_alpha": best_alpha
}
dump(model_bundle, os.path.join(MODEL_DIR, "ridge_house.joblib"))
print("✅ Model and metadata saved to models/ridge_house.joblib")

# 9) Feature Coefficients
preproc = best_model.named_steps['preproc']
try:
    feature_names = preproc.get_feature_names_out()
except:
    feature_names = numeric_features + categorical_features

coef = best_model.named_steps['ridge'].coef_
coef_df = pd.DataFrame({'feature': feature_names[:len(coef)], 'coefficient': coef})
coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False)

# 10) Save results.json
results = {
    'rmse': float(rmse),
    'r2': float(r2),
    'best_alpha': float(best_alpha),
    'sample_predictions': [
        {'actual': float(a), 'predicted': float(p)}
        for a, p in zip(y_valid[:20], y_pred[:20])
    ],
    'feature_coeffs': coef_df.head(20).to_dict(orient='records')
}
with open(os.path.join(OUT_DIR, "results.json"), 'w') as f:
    json.dump(results, f, indent=2)

# 11) Plots
plt.figure(figsize=(6,6))
plt.scatter(y_valid, y_pred, alpha=0.7)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], '--', color='gray')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pred_vs_actual.png"))
plt.close()

plt.figure(figsize=(6,4))
residuals = y_valid - y_pred
plt.hist(residuals, bins=20, edgecolor='k')
plt.title("Residuals")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residuals.png"))
plt.close()

print("Outputs written to:", OUT_DIR)
