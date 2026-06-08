import os
import pandas as pd
import numpy as np

os.makedirs("data/processed", exist_ok=True)
np.random.seed(42)

n = 4338

rfm = pd.DataFrame({
    "CustomerID": range(1, n + 1),
    "Recency": np.random.randint(1, 365, n),
    "Frequency": np.random.randint(1, 40, n),
    "Monetary": np.random.randint(100, 20000, n),
})

rfm["Cluster_Name"] = np.random.choice(
    ["Champions", "Loyal Customers", "Recent Low-Value", "At Risk / Lost"],
    n
)

churn = pd.DataFrame({
    "CustomerID": rfm["CustomerID"],
    "Churn_Probability": np.random.random(n),
})
churn["Churned"] = (churn["Churn_Probability"] > 0.65).astype(int)

forecast = pd.DataFrame({
    "Week": pd.date_range("2026-01-01", periods=12, freq="W"),
    "ForecastedDemand": np.random.randint(90000, 180000, 12)
})

inventory = forecast.copy()
inventory["RecommendedOrder"] = inventory["ForecastedDemand"] * 1.15

rfm.to_csv("data/processed/rfm_clustered.csv", index=False)
churn.to_csv("data/processed/churn_scores.csv", index=False)
forecast.to_csv("data/processed/weekly_forecast.csv", index=False)
inventory.to_csv("data/processed/inventory_recommendations.csv", index=False)

print("Required files created successfully!")