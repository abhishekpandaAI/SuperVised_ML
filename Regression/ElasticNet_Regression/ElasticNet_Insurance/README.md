💰 Medical Insurance Cost Prediction (Elastic Net Regression)

This project uses the Insurance dataset
 from Kaggle to predict medical charges based on demographic and lifestyle factors. The model is trained with Elastic Net Regression, which combines the strengths of Ridge (L2) and Lasso (L1) regularization.

📂 Project Structure
ElasticNet_Insurance/
├── data/
│   └── insurance.csv             # Dataset (from Kaggle)
├── models/
│   └── elastic_insurance.joblib  # Trained Elastic Net model
├── outputs/
│   ├── results.json              # Metrics (RMSE, R², best params)
│   └── feature_importance.png    # Feature importance plot
├── frontend/
│   └── index.html                # Simple web UI
├── elastic_insurance.py          # Training script
├── main.py                       # FastAPI backend
├── requirements.txt              # Dependencies
└── README.md

🚀 How to Run
1️⃣ Setup Environment
cd Regression/ElasticNet_Insurance
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt

2️⃣ Train the Model
python elastic_insurance.py


This will:

Train an Elastic Net regression model

Save the model in models/elastic_insurance.joblib

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

Prediction Form: Enter patient details (age, BMI, smoker, etc.) and get predicted insurance charges.

Prediction History: Line chart tracking multiple predictions in one session.

Model Metrics: RMSE, R², and best hyperparameters shown in UI.

Feature Importance: Bar chart of most influential features (e.g., smoking status, BMI).

📊 Example Input
{
  "age": 31,
  "sex": "female",
  "bmi": 25.74,
  "children": 0,
  "smoker": "no",
  "region": "southeast"
}

✅ Example Output
{
  "predicted_charges": 3756.62
}

🔎 Why Elastic Net?

Ridge (L2) → Shrinks coefficients, handles multicollinearity.

Lasso (L1) → Performs feature selection (some coefficients = 0).

Elastic Net (L1+L2) → Combines both: keeps groups of correlated variables while selecting the most important.

This makes Elastic Net ideal for datasets like Insurance, where demographic and lifestyle features are correlated.