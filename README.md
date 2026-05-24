# RetailPulse – AI-Powered Customer Analytics & Demand Forecasting

## 📌 Overview
RetailPulse is a data science project that analyzes retail customer behavior, predicts churn, segments customers, and optimizes inventory using machine learning.

## 🌐 Live Demo
🚀 Streamlit App: https://retailpulse-kohinoor.streamlit.app/

## 🚀 Key Highlights
- Interactive Streamlit dashboard
- Customer segmentation using K-Means
- Churn prediction using Random Forest
- Inventory optimization recommendations
- Dynamic filters and downloadable reports
- GitHub-ready project structure

## 🏗️ Project Structure

```text
RetailPulse/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── dashboard/
│   └── app.py
├── reports/
├── tests/
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Docker
- Git & GitHub

## 📊 Modules
1. Exploratory Data Analysis
2. Feature Engineering
3. Customer Segmentation
4. Demand Forecasting
5. Churn Prediction
6. Inventory Optimization
7. Streamlit Dashboard

## ▶️ How to Run

```bash
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

## 📈 Dashboard Features
- Upload custom CSV dataset
- Filter sales and frequency dynamically
- View business KPIs
- Analyze customer segments
- Predict churn risk
- Generate inventory reorder recommendations
- Download processed results as CSV

## 📌 Expected Dataset Columns

```text
customer_id, sales, quantity, frequency, recency, inventory, churn
```

## 📈 Future Improvements
- Add larger real-world retail dataset
- Add XGBoost model
- Add SHAP explainability
- Add Prophet/LSTM forecasting
- Add MLflow tracking
- Add GitHub Actions CI/CD
- Deploy with Docker

## 👤 Author
**Shahzada Kohinoor**