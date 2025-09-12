🌊 Bayesian Ridge Regression – Water Quality Prediction
📌 Overview

This project uses Bayesian Ridge Regression to predict drinking water quality based on multiple chemical and physical parameters.

The model is trained on a Water Quality dataset and extended with WHO (World Health Organization) guidelines to classify water as:

✅ Safe (potable, within recommended limits)

⚠️ Unsafe (out of safe limits or predicted as unsafe by ML model)

⚡ Why Bayesian Ridge?

Unlike ordinary regression, Bayesian Ridge Regression applies a probabilistic approach.

It balances bias vs variance, avoids overfitting, and works well when there are many correlated features (like pH, Solids, Sulfates).

It gives a distribution of coefficients rather than fixed values → better interpretability.

🧪 Features Used
Parameter	Description	WHO Safe Range
ph	Acidity/alkalinity level	6.5 – 8.5
Hardness	Minerals (calcium, magnesium)	≤ 300 mg/L
Solids (TDS)	Total dissolved solids	≤ 500 mg/L (up to 1000)
Chloramines	Disinfectant level	≤ 4 mg/L
Sulfate	Sulfate salts	≤ 250 mg/L
Conductivity	Ability to conduct electricity	50 – 500 µS/cm
Organic_carbon	Organic contamination	≤ 5 mg/L
Trihalomethanes	Byproducts of disinfection	≤ 80 µg/L
Turbidity	Water clarity	≤ 5 NTU (ideally < 1 NTU)
🛠️ Project Structure
Regression/Bayesian_Ridge_Water/
├── data/
│   └── WaterQuality.csv
├── models/
│   └── bayesian_water.joblib
├── outputs/
│   ├── results.json
│   └── pred_vs_actual.png
├── frontend/
│   └── index.html        # UI for predictions & charts
├── bayesian_water.py     # Training script
├── main.py               # FastAPI backend with WHO override
├── requirements.txt
└── README.md

🚀 How It Works

Training (bayesian_water.py)

Loads dataset

Preprocesses numeric features

Trains Bayesian Ridge model

Evaluates & saves model

Backend (main.py)

FastAPI service with /predict endpoint

Accepts JSON input of water parameters

Predicts potability score

WHO override applied: if all values are safe → always returns Safe 🚰

Frontend (index.html)

User enters water test parameters

Shows prediction score, Safe/Unsafe status, reasons

Live input validation (green/red) based on WHO safe ranges

Charts:

📈 Prediction history (scores over time)

📊 Safe vs Unsafe counts

WHO Guidelines displayed on the right panel

🔎 Example Input
{
  "ph": 7.4,
  "Hardness": 120,
  "Solids": 350,
  "Chloramines": 2.5,
  "Sulfate": 180,
  "Conductivity": 300,
  "Organic_carbon": 3.5,
  "Trihalomethanes": 50,
  "Turbidity": 1.2
}

✅ Example Output
{
  "predicted_score": 0.379,
  "status": "Safe 🚰",
  "reasons": [
    "All key water quality indicators are within safe WHO limits."
  ]
}

📊 Visualizations

Predicted vs Actual chart (from training)

Residuals histogram

Frontend charts for:

Prediction history

Safe vs Unsafe distribution

💡 Use Cases

Public Health – test potability of rural/urban water supply

IoT Water Sensors – integrate with real-time monitoring

Smart Cities – dashboard for water utilities

Household Water Purifiers – quick feedback for consumers

🛠️ Setup & Run
# 1. Navigate to project
cd Regression/Bayesian_Ridge_Water

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train model
python bayesian_water.py

# 5. Run backend
uvicorn main:app --reload

# 6. Open frontend
open frontend/index.html   # or just double click it

🎯 Conclusion

This project shows how Bayesian Ridge Regression can be applied to environmental health monitoring.
By combining Machine Learning with WHO guidelines, the system ensures both scientific accuracy and practical reliability.