# Supervised Machine Learning Repository

A comprehensive collection of **Supervised Machine Learning algorithms** implemented in Python, covering both **Regression** and **Classification** tasks. This repository is designed for learning, experimentation, and building predictive models with real-world datasets.

## 📂 Repository Structure

```
SuperVised_ML/
│
├── data/                    # Sample datasets (CSV files)
├── models/                  # Saved trained models (.joblib)
├── notebooks/               # Jupyter notebooks for experimentation
├── regression/              # Regression algorithms
│   ├── linear_regression.py
│   ├── polynomial_regression.py
│   ├── ridge_regression.py
│   └── lasso_regression.py
├── classification/          # Classification algorithms
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── svm.py
├── frontend/                # HTML/JS UI for predictions
├── train_stock_model.py     # Stock prediction model
├── train_index_model.py     # Index prediction model
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 📈 Regression Algorithms

| Algorithm | File | Description | Use Cases |
|-----------|------|-------------|-----------|
| **Linear Regression** | `linear_regression.py` | Predicts continuous values using a linear relationship | House prices, sales forecasting |
| **Polynomial Regression** | `polynomial_regression.py` | Extends linear regression to model non-linear relationships | Curved data patterns, growth models |
| **Ridge Regression** | `ridge_regression.py` | Linear regression with L2 regularization to prevent overfitting | High-dimensional data, multicollinearity |
| **Lasso Regression** | `lasso_regression.py` | Linear regression with L1 regularization for feature selection | Sparse models, feature importance |
| **Stock Prediction** | `train_stock_model.py` | Financial time series prediction using lag features | Stock price forecasting |
| **Index Prediction** | `train_index_model.py` | Market index prediction with technical indicators | Market trend analysis |

## 🎯 Classification Algorithms

| Algorithm | File | Description | Use Cases |
|-----------|------|-------------|-----------|
| **Logistic Regression** | `logistic_regression.py` | Predicts binary or multi-class outcomes using logistic function | Email spam detection, medical diagnosis |
| **K-Nearest Neighbors** | `knn.py` | Classification based on k closest training examples | Recommendation systems, pattern recognition |
| **Decision Tree** | `decision_tree.py` | Tree-based classifier with interpretable rules | Credit approval, medical decision making |
| **Random Forest** | `random_forest.py` | Ensemble of decision trees for improved accuracy | Feature importance, robust predictions |
| **Support Vector Machine** | `svm.py` | Finds optimal hyperplane to separate classes | Text classification, image recognition |

## 🗄️ Datasets & Features

### Stock/Index Prediction Features
- **Lagged Values**: Previous day prices (lag_1, lag_2, lag_3)
- **Moving Averages**: 5-day and 10-day rolling averages
- **Returns**: Daily percentage changes
- **Temporal Features**: Day of week, month
- **Technical Indicators**: Volatility measures

### Sample Datasets
- Historical stock price data (OHLCV format)
- Market index data
- Training/validation splits included

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/abhishekpandaAI/SuperVised_ML.git
cd SuperVised_ML

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models
```bash
# Train stock prediction model
python train_stock_model.py

# Train index prediction model
python train_index_model.py
```

### 3. Run Web Interface
```bash
# Start FastAPI backend
uvicorn main:app --reload

# Open frontend/index.html in your browser
```

## 📊 Model Performance

All models include:
- **Training/Validation split** for proper evaluation
- **Cross-validation** for robust performance metrics
- **Feature scaling** where appropriate
- **Hyperparameter tuning** for optimal results
- **Model persistence** using joblib

## 🌐 Web Interface

The repository includes a user-friendly web interface built with:
- **HTML/CSS/JavaScript** frontend
- **FastAPI** backend for model serving
- **Interactive forms** for feature input
- **Visualization charts** for predictions
- **Real-time predictions** with trained models

## 📋 Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
fastapi
uvicorn
joblib
jupyter
```

## 🔧 Usage Examples

### Regression Example
```python
from regression.linear_regression import LinearRegressionModel

# Initialize and train model
model = LinearRegressionModel()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Classification Example
```python
from classification.random_forest import RandomForestModel

# Initialize and train model
model = RandomForestModel()
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.evaluate(X_test, y_test)
```

## 📈 Model Evaluation Metrics

### Regression Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

### Classification Metrics
- Accuracy Score
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Commit your changes (`git commit -am 'Add new algorithm'`)
4. Push to the branch (`git push origin feature/new-algorithm`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abhishek Panda**
- GitHub: [@abhishekpandaAI](https://github.com/abhishekpandaAI)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Scikit-learn for machine learning algorithms
- Pandas for data manipulation
- Matplotlib/Seaborn for visualization
- FastAPI for web framework

---

**⭐ Star this repository if you found it helpful!**