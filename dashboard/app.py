import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="RetailPulse", layout="wide")

st.title("RetailPulse - AI Customer Analytics Dashboard")
st.caption("Customer Segmentation | Churn Prediction | Inventory Optimization")

# ---------------- DATA LOADING ----------------
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

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

required_cols = [
    "customer_id", "sales", "quantity",
    "frequency", "recency", "inventory", "churn"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Dashboard Filters")

sales_range = st.sidebar.slider(
    "Sales Range",
    int(df["sales"].min()),
    int(df["sales"].max()),
    (int(df["sales"].min()), int(df["sales"].max()))
)

frequency_range = st.sidebar.slider(
    "Frequency Range",
    int(df["frequency"].min()),
    int(df["frequency"].max()),
    (int(df["frequency"].min()), int(df["frequency"].max()))
)

churn_filter = st.sidebar.multiselect(
    "Churn Status",
    options=sorted(df["churn"].unique()),
    default=sorted(df["churn"].unique())
)

df = df[
    (df["sales"] >= sales_range[0]) &
    (df["sales"] <= sales_range[1]) &
    (df["frequency"] >= frequency_range[0]) &
    (df["frequency"] <= frequency_range[1]) &
    (df["churn"].isin(churn_filter))
]

menu = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Customer Segmentation",
        "Churn Prediction",
        "Inventory Optimization",
        "Data Explorer"
    ]
)

st.sidebar.write("Rows after filter:", df.shape[0])

# ---------------- OVERVIEW ----------------
if menu == "Overview":
    st.subheader("Business Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Sales", f"₹{df['sales'].sum():,.0f}")
    c2.metric("Customers", df["customer_id"].nunique())
    c3.metric("Avg Sales", f"₹{df['sales'].mean():.2f}")
    c4.metric("Total Quantity", int(df["quantity"].sum()))
    c5.metric("Churn Rate", f"{df['churn'].mean() * 100:.2f}%")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="sales",
            nbins=30,
            title="Sales Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df,
            x="frequency",
            y="sales",
            color=df["churn"].astype(str),
            size="quantity",
            title="Frequency vs Sales"
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df,
        x="churn",
        y="sales",
        color=df["churn"].astype(str),
        title="Sales by Churn Status"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- CUSTOMER SEGMENTATION ----------------
elif menu == "Customer Segmentation":
    st.subheader("Customer Segmentation")

    k = st.slider("Number of Customer Segments", 2, 8, 4)

    features = st.multiselect(
        "Select Features for Segmentation",
        ["recency", "frequency", "sales", "quantity", "inventory"],
        default=["recency", "frequency", "sales"]
    )

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["segment"] = model.fit_predict(X_scaled)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df,
            x="frequency",
            y="sales",
            color=df["segment"].astype(str),
            hover_data=["customer_id", "recency"],
            title="Customer Segments"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df,
            x="segment",
            y="sales",
            color=df["segment"].astype(str),
            title="Sales by Segment"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Summary")
    st.dataframe(
        df.groupby("segment")[["sales", "frequency", "recency", "quantity", "inventory"]].mean(),
        use_container_width=True
    )

    st.download_button(
        "Download Segmented Customers",
        df.to_csv(index=False),
        "segmented_customers.csv",
        "text/csv"
    )

# ---------------- CHURN PREDICTION ----------------
elif menu == "Churn Prediction":
    st.subheader("Churn Prediction")

    X = df[["sales", "quantity", "frequency", "recency", "inventory"]]
    y = df["churn"]

    if y.nunique() < 2:
        st.error("Churn column must contain both 0 and 1 values.")
        st.stop()

    test_size = st.slider("Test Size", 0.2, 0.4, 0.25)
    trees = st.slider("Number of Trees", 50, 300, 150)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=trees,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", round(accuracy_score(y_test, preds), 3))
    c2.metric("AUC Score", round(roc_auc_score(y_test, probs), 3))

    df["churn_risk"] = model.predict_proba(X)[:, 1]

    st.subheader("Top Churn Risk Customers")
    st.dataframe(
        df.sort_values("churn_risk", ascending=False).head(20),
        use_container_width=True
    )

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            importance,
            x="Feature",
            y="Importance",
            title="Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cm = confusion_matrix(y_test, preds)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )
        fig = px.imshow(
            cm_df,
            text_auto=True,
            title="Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Churn Predictions",
        df.to_csv(index=False),
        "churn_predictions.csv",
        "text/csv"
    )

# ---------------- INVENTORY OPTIMIZATION ----------------
elif menu == "Inventory Optimization":
    st.subheader("Inventory Optimization")

    demand_factor = st.slider("Demand Forecast Multiplier", 1.0, 5.0, 1.5)
    safety_stock = st.slider("Safety Stock %", 0, 100, 20)

    df["forecast_demand"] = df["quantity"] * demand_factor
    df["safety_stock"] = df["forecast_demand"] * (safety_stock / 100)

    df["reorder_quantity"] = np.maximum(
        df["forecast_demand"] + df["safety_stock"] - df["inventory"],
        0
    ).round()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Forecast Demand", round(df["forecast_demand"].sum(), 2))
    c2.metric("Total Safety Stock", round(df["safety_stock"].sum(), 2))
    c3.metric("Total Reorder Qty", round(df["reorder_quantity"].sum(), 2))

    st.dataframe(
        df[[
            "customer_id",
            "quantity",
            "inventory",
            "forecast_demand",
            "safety_stock",
            "reorder_quantity"
        ]],
        use_container_width=True
    )

    fig = px.bar(
        df.sort_values("reorder_quantity", ascending=False).head(30),
        x="customer_id",
        y="reorder_quantity",
        title="Top Reorder Recommendations"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Inventory Recommendations",
        df.to_csv(index=False),
        "inventory_recommendations.csv",
        "text/csv"
    )

# ---------------- DATA EXPLORER ----------------
elif menu == "Data Explorer":
    st.subheader("Interactive Data Explorer")

    selected_columns = st.multiselect(
        "Select Columns",
        df.columns.tolist(),
        default=df.columns.tolist()
    )

    st.dataframe(df[selected_columns], use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    chart_x = st.selectbox("X-axis", df.columns)
    chart_y = st.selectbox("Y-axis", df.select_dtypes(include=np.number).columns)

    fig = px.scatter(
        df,
        x=chart_x,
        y=chart_y,
        title=f"{chart_x} vs {chart_y}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Filtered Dataset",
        df.to_csv(index=False),
        "filtered_dataset.csv",
        "text/csv"
    )