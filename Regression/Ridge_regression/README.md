🏡 House Price Prediction (Ridge Regression)

This project uses the Housing Prices dataset
 to predict house prices based on property features such as area, bedrooms, bathrooms, stories, and furnishing status. The model is trained with Ridge Regression, which applies L2 regularization to reduce overfitting and handle multicollinearity.

📂 Project Structure
Ridge_Regression/
├── data/
│   └── Housing.csv              # Dataset (from Kaggle)
├── models/
│   └── ridge_house.joblib       # Trained Ridge model
├── outputs/
│   ├── results.json             # Metrics (RMSE, R², alpha)
│   └── pred_vs_actual.png       # Predicted vs Actual plot
│   └── residuals.png            # Residuals distribution
├── frontend/
│   └── index.html               # Simple web UI
├── ridge_house.py               # Training script
├── main.py                      # FastAPI backend
├── requirements.txt             # Dependencies
└── README.md

🚀 How to Run
1️⃣ Setup Environment
cd Regression/Ridge_Regression
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt

2️⃣ Train the Model
python ridge_house.py


This will:

Train a Ridge regression model

Save the model in models/ridge_house.joblib

Save evaluation metrics in outputs/results.json

Generate plots:

pred_vs_actual.png → scatter plot comparing predictions vs actual prices

residuals.png → residual distribution plot

3️⃣ Start the API
uvicorn main:app --reload


API available at: http://127.0.0.1:8000

Interactive docs: http://127.0.0.1:8000/docs

4️⃣ Run the Frontend

Either:

Open frontend/index.html directly in a browser, or

Serve it with Python:

python -m http.server 8001


Then open: http://localhost:8001/frontend/index.html

🎯 Features

Prediction Form: Enter house details (area, bedrooms, stories, parking, etc.) and get price prediction.

Prediction History: Line chart of predictions in the session.

Model Metrics: RMSE, R², and best hyperparameter (alpha) displayed.

Visuals: Scatter plot (predicted vs actual) + histogram of residuals.

📊 Example Input
{
  "area": 7500,
  "bedrooms": 3,
  "bathrooms": 2,
  "stories": 2,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "yes",
  "hotwaterheating": "no",
  "airconditioning": "yes",
  "parking": 2,
  "prefarea": "yes",
  "furnishingstatus": "semi-furnished"
}

✅ Example Output
{
  "predicted_price": 12215000.0
}

🔎 Why Ridge?

Handles multicollinearity (when features are correlated).

Prevents overfitting by shrinking large coefficients.

Keeps all features (unlike Lasso, which removes some).

Best suited for housing data, where many factors are correlated (e.g., bedrooms, area, bathrooms).