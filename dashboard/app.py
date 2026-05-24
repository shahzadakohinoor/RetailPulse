import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="RetailPulse", layout="wide")

st.title("RetailPulse - AI Customer Analytics Dashboard")

# ---------------- LOAD DATA ----------------
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Using Uploaded Dataset")
else:
    data_path = os.path.join("data", "raw", "retail_data.csv")

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.sidebar.success("Using Real Dataset")
    else:
        st.warning("No dataset found. Using sample data.")
        np.random.seed(42)
        df = pd.DataFrame({
            "customer_id": range(1, 501),
            "sales": np.random.randint(100, 5000, 500),
            "quantity": np.random.randint(1, 20, 500),
            "frequency": np.random.randint(1, 30, 500),
            "recency": np.random.randint(1, 365, 500),
            "inventory": np.random.randint(10, 300, 500),
            "churn": np.random.randint(0, 2, 500)
        })

required_cols = ["customer_id", "sales", "quantity", "frequency", "recency", "inventory", "churn"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filters")

min_sales, max_sales = int(df["sales"].min()), int(df["sales"].max())
sales_range = st.sidebar.slider("Sales Range", min_sales, max_sales, (min_sales, max_sales))

df = df[(df["sales"] >= sales_range[0]) & (df["sales"] <= sales_range[1])]

menu = st.sidebar.radio(
    "Select Page",
    ["Overview", "Customer Segmentation", "Churn Prediction", "Inventory Optimization"]
)

st.write("Dataset Shape:", df.shape)

# ---------------- OVERVIEW ----------------
if menu == "Overview":
    st.subheader("Business Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"₹{df['sales'].sum():,.0f}")
    c2.metric("Customers", df["customer_id"].nunique())
    c3.metric("Avg Sales", f"₹{df['sales'].mean():.2f}")
    c4.metric("Churn Rate", f"{df['churn'].mean() * 100:.2f}%")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="sales", title="Sales Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df,
            x="frequency",
            y="sales",
            color=df["churn"].astype(str),
            title="Frequency vs Sales"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------- SEGMENTATION ----------------
elif menu == "Customer Segmentation":
    st.subheader("Customer Segmentation")

    k = st.slider("Select Number of Clusters", 2, 6, 4)

    X = df[["recency", "frequency", "sales"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["segment"] = model.fit_predict(X_scaled)

    fig = px.scatter(
        df,
        x="frequency",
        y="sales",
        color=df["segment"].astype(str),
        hover_data=["customer_id", "recency"],
        title="Customer Segments"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Summary")
    st.dataframe(df.groupby("segment")[["sales", "frequency", "recency"]].mean())

    st.download_button(
        "Download Segmented Customers",
        df.to_csv(index=False),
        "segmented_customers.csv",
        "text/csv"
    )

# ---------------- CHURN ----------------
elif menu == "Churn Prediction":
    st.subheader("Churn Prediction")

    X = df[["sales", "quantity", "frequency", "recency", "inventory"]]
    y = df["churn"]

    if y.nunique() < 2:
        st.error("Churn column must contain both 0 and 1 values.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", round(accuracy_score(y_test, preds), 3))
    c2.metric("AUC Score", round(roc_auc_score(y_test, probs), 3))

    df["churn_risk"] = model.predict_proba(X)[:, 1]

    st.subheader("Top Churn Risk Customers")
    st.dataframe(df.sort_values("churn_risk", ascending=False).head(20))

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = px.bar(importance, x="Feature", y="Importance", title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Churn Predictions",
        df.to_csv(index=False),
        "churn_predictions.csv",
        "text/csv"
    )

# ---------------- INVENTORY ----------------
elif menu == "Inventory Optimization":
    st.subheader("Inventory Optimization")

    demand_factor = st.slider("Demand Forecast Multiplier", 1.0, 3.0, 1.5)

    df["forecast_demand"] = df["quantity"] * demand_factor
    df["reorder_quantity"] = np.maximum(df["forecast_demand"] - df["inventory"], 0).round()

    st.dataframe(df[[
        "customer_id",
        "quantity",
        "inventory",
        "forecast_demand",
        "reorder_quantity"
    ]])

    fig = px.bar(
        df.head(30),
        x="customer_id",
        y="reorder_quantity",
        title="Reorder Quantity Recommendation"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Inventory Recommendations",
        df.to_csv(index=False),
        "inventory_recommendations.csv",
        "text/csv"
    )