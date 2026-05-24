import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="RetailPulse", layout="wide")

st.title("RetailPulse - AI Customer Analytics Dashboard")

# Load dataset
try:
    df = pd.read_csv("data/raw/retail_data.csv")
    st.sidebar.success("Using Real Dataset")
except FileNotFoundError:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Using Uploaded Dataset")
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

st.write("Dataset Shape:", df.shape)

required_cols = ["customer_id", "sales", "quantity", "frequency", "recency", "inventory", "churn"]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

menu = st.sidebar.radio(
    "Select Page",
    ["Overview", "Customer Segmentation", "Churn Prediction", "Inventory Optimization"]
)

if menu == "Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sales", f"₹{df['sales'].sum():,.0f}")
    c2.metric("Customers", df["customer_id"].nunique())
    c3.metric("Avg Sales", f"₹{df['sales'].mean():.2f}")
    c4.metric("Total Quantity", int(df["quantity"].sum()))

    fig = px.histogram(df, x="sales", title="Sales Distribution")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="frequency",
        y="sales",
        color="churn",
        title="Frequency vs Sales"
    )
    st.plotly_chart(fig2, use_container_width=True)

elif menu == "Customer Segmentation":
    st.subheader("Customer Segmentation")

    X = df[["recency", "frequency", "sales"]]

    k = st.slider("Select Number of Clusters", 2, 6, 4)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["segment"] = model.fit_predict(X)

    fig = px.scatter(
        df,
        x="frequency",
        y="sales",
        color=df["segment"].astype(str),
        hover_data=["customer_id", "recency"],
        title="Customer Segments"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df)

elif menu == "Churn Prediction":
    st.subheader("Churn Prediction")

    X = df[["sales", "quantity", "frequency", "recency", "inventory"]]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", round(accuracy_score(y_test, preds), 3))
    c2.metric("AUC Score", round(roc_auc_score(y_test, probs), 3))

    df["churn_risk"] = model.predict_proba(X)[:, 1]

    st.subheader("Top Churn Risk Customers")
    st.dataframe(
        df.sort_values("churn_risk", ascending=False).head(20)
    )

elif menu == "Inventory Optimization":
    st.subheader("Inventory Optimization")

    df["forecast_demand"] = df["quantity"] * 1.5

    df["reorder_quantity"] = np.maximum(
        df["forecast_demand"] - df["inventory"], 0
    ).round()

    st.dataframe(
        df[[
            "customer_id",
            "quantity",
            "inventory",
            "forecast_demand",
            "reorder_quantity"
        ]]
    )

    fig = px.bar(
        df.head(30),
        x="customer_id",
        y="reorder_quantity",
        title="Reorder Quantity Recommendation"
    )

    st.plotly_chart(fig, use_container_width=True)