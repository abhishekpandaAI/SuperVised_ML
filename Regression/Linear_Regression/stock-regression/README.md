📈 Stock & Index Price Prediction (Linear Regression + FastAPI)

This project trains Linear Regression models to predict both:

📊 Stock prices (from stocks_df.csv)

📈 Market Indexes (from nse_indexes.csv)

It provides a FastAPI backend, a simple frontend (HTML form), and visualization scripts.

🚀 Getting Started
1️⃣ Clone / Download Project
git clone https://github.com/your-username/stock-regression.git
cd stock-regression

2️⃣ Create Virtual Environment
On macOS / Linux
    python3 -m venv venv
    source venv/bin/activate

On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Prepare Data

Ensure your dataset is in the data/ folder:

data/
 ├─ stocks_df.csv       # individual stock OHLC data
 ├─ nse_indexes.csv     # index OHLC data
 └─ indexes_df.csv      # reference list of index names

5️⃣ Train Models

Train stock model:

python train_stock_model.py


Train index model:

python train_index_model.py


This saves trained models into models/:

models/
 ├─ stock_model.joblib
 └─ index_model.joblib

6️⃣ Run API
uvicorn app.main:app --reload


Open API docs (Swagger):
👉 http://127.0.0.1:8000/docs

Endpoints:

POST /predict_stock → Predict stock closing price

POST /predict_index → Predict index closing price

7️⃣ Run Frontend (Optional)

Open the file frontend/index.html in your browser.
python -m http.server 3000
It will allow you to input features and get predictions via the FastAPI backend.

8️⃣ Visualize Results
python visualize_results.py


This plots actual vs predicted closing prices.

🐳 Run with Docker (Optional)

Build the image:

docker build -t stock-regression .


Run container:

docker run -p 8000:8000 stock-regression


API will be available at 👉 http://127.0.0.1:8000/docs

📌 Tech Stack

Python 3.10+

Pandas / NumPy → Data processing

Scikit-learn → Linear Regression models

FastAPI → REST API

Uvicorn → Server

Joblib → Model persistence

Matplotlib / Plotly → Visualization

✅ You’re all set! 🎉
------
features used for predicting stock or index prices:

Close_lag_1 = 1000

The closing price of the stock/index 1 day ago.

Close_lag_2 = 995

The closing price 2 days ago.

Close_lag_3 = 990

The closing price 3 days ago.

MA_5 = 997

The 5-day moving average of closing prices (average of last 5 days).

MA_10 = 992

The 10-day moving average of closing prices.

Return_1 = 0.005

The daily return from the previous day (percentage change in price): here 0.5%.

DayOfWeek = 1

The day of the week for the current date: 0 = Monday, 1 = Tuesday, … 6 = Sunday.

Month = 9

The current month of the year: 1 = January, … 12 = December.

✅ These are all input features for your machine learning model (Linear Regression) to predict the next day’s closing price.