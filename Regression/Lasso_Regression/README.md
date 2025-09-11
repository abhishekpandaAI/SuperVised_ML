🩺 Diabetes Prediction (Lasso Regression)

This project uses the Diabetes dataset
 to predict disease progression one year after baseline measurements. The model is trained with Lasso Regression, which uses L1 regularization to automatically remove irrelevant features.

📂 Project Structure
Lasso_Regression/
├── data/
│   └── diabetes.csv              # Dataset (from Kaggle)
├── models/
│   └── lasso_diabetes.joblib     # Trained Lasso model
├── outputs/
│   ├── results.json              # Metrics (RMSE, R², alpha)
│   └── feature_importance.png    # Feature importance plot
├── frontend/
│   └── index.html                # Simple web UI
├── lasso_diabetes.py             # Training script
├── main.py                       # FastAPI backend
├── requirements.txt              # Dependencies
└── README.md

🚀 How to Run
1️⃣ Setup Environment
cd Regression/Lasso_Regression
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt

2️⃣ Train the Model
python lasso_diabetes.py


This will:

Train a Lasso regression model

Save the model in models/lasso_diabetes.joblib

Save evaluation metrics in outputs/results.json

Save feature importance plot in outputs/feature_importance.png

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

Prediction Form: Enter patient health indicators (Glucose, BMI, Insulin, Age, etc.) and get predictions.

Prediction History: Line chart tracking predictions in the session.

Model Metrics: RMSE, R², and best hyperparameter (alpha) displayed.

Feature Importance: Bar chart showing most important features chosen by Lasso (irrelevant ones shrink to 0).

📊 Example Input
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 79,
  "BMI": 25.5,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 30
}

✅ Example Output
{
  "predicted_progression": 122.45
}

🔎 Why Lasso?

Performs feature selection automatically by shrinking some coefficients to zero.

Useful when dataset contains many irrelevant features.

Helps build simpler, interpretable models in healthcare and beyond.

💡 In healthcare, this means we can discover which biomarkers truly matter for predicting disease risk, while ignoring the noise.