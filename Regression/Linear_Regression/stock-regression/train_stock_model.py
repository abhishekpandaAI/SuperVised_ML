import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

def build_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    for lag in [1,2,3,5]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)

    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['Return_1'] = df['Close'].pct_change(periods=1)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    return df.dropna().reset_index(drop=True)

# Load stock data
df = pd.read_csv("data/stocks_df.csv")

# Train for Reliance (can be parameterized later)
if "Stock" in df.columns:
    df = df[df['Stock'] == 'RELIANCE']
elif "TICKER" in df.columns:
    df = df[df['TICKER'] == 'RELIANCE']
else:
    print("⚠️ No symbol column found, using entire dataset")


df = build_features(df)

FEATURES = ['Close_lag_1','Close_lag_2','Close_lag_3','MA_5','MA_10','Return_1','DayOfWeek','Month']
X, y = df[FEATURES], df['Close']

split_idx = int(len(df) * 0.8)
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])
pipeline.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
dump({"model": pipeline, "features": FEATURES}, "models/stock_model.joblib")
print("✅ Stock model trained and saved")
