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

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #050816, #0f172a);
    color: #dbeafe;
    font-family: monospace;
}

[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid #1e293b;
}

.hero {
    background: linear-gradient(135deg, #111827, #1e293b);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #334155;
    margin-bottom: 25px;
    box-shadow: 0 0 30px rgba(251,191,36,0.12);
}

.hero h1 {
    color: #fbbf24;
    font-size: 44px;
    margin-bottom: 4px;
}

.hero p {
    color: #93c5fd;
    letter-spacing: 3px;
    font-size: 15px;
}

.stMetric {
    background: #111827;
    border: 1px solid #334155;
    padding: 18px;
    border-radius: 14px;
    transition: 0.3s ease;
}

.stMetric:hover {
    transform: scale(1.03);
    border-color: #fbbf24;
    box-shadow: 0 0 20px rgba(251,191,36,0.25);
}

div[data-testid="stMetricValue"] {
    color: #fbbf24;
    font-size: 28px;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
}

h1, h2, h3 {
    color: #dbeafe;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
    border-bottom: 1px solid #334155;
}

.stTabs [data-baseweb="tab"] {
    color: #94a3b8;
    font-size: 16px;
    letter-spacing: 1px;
}

.stTabs [aria-selected="true"] {
    color: #fbbf24;
    border-bottom: 3px solid #fbbf24;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#DBEAFE"),
        title_font=dict(size=20, color="#FBBF24"),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


st.markdown("""
<div class="hero">
<h1>RetailPulse ▲</h1>
<p>AI RETAIL INTELLIGENCE TERMINAL</p>
</div>
""", unsafe_allow_html=True)

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

required_cols = ["customer_id", "sales", "quantity", "frequency", "recency", "inventory", "churn"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

st.sidebar.header("Filters")

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

if df.empty:
    st.error("No data available after applying filters.")
    st.stop()

st.sidebar.metric("Rows After Filter", df.shape[0])

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Segments",
    "Churn Risk",
    "Inventory",
    "Data Explorer"
])

with tab1:
    st.subheader("Business Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", df["customer_id"].nunique())
    c2.metric("Revenue", f"₹{df['sales'].sum():,.0f}")
    c3.metric("Avg Sales", f"₹{df['sales'].mean():.0f}")
    c4.metric("Quantity", int(df["quantity"].sum()))
    c5.metric("Churn Rate", f"{df['churn'].mean() * 100:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="sales", nbins=30, title="Sales Distribution")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with col2:
        fig = px.scatter(
            df,
            x="frequency",
            y="sales",
            color=df["churn"].astype(str),
            size="quantity",
            title="Frequency vs Sales"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    fig = px.box(
        df,
        x=df["churn"].astype(str),
        y="sales",
        color=df["churn"].astype(str),
        title="Sales by Churn Status"
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.dataframe(df.head(20), use_container_width=True)

with tab2:
    st.subheader("Customer Segmentation")

    k = st.slider("Number of Segments", 2, 8, 4)

    features = st.multiselect(
        "Select Features",
        ["recency", "frequency", "sales", "quantity", "inventory"],
        default=["recency", "frequency", "sales"]
    )

    if len(features) < 2:
        st.error("Select at least 2 features.")
        st.stop()

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
            size="quantity",
            hover_data=["customer_id", "recency"],
            title="Customer Segments"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with col2:
        fig = px.pie(
            df,
            names="segment",
            values="sales",
            hole=0.45,
            title="Revenue Share by Segment"
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

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

with tab3:
    st.subheader("Churn Prediction")

    X = df[["sales", "quantity", "frequency", "recency", "inventory"]]
    y = df["churn"]

    if y.nunique() < 2:
        st.error("Churn column must contain both 0 and 1 values.")
        st.stop()

    test_size = st.slider("Test Size", 0.2, 0.4, 0.25)
    trees = st.slider("Number of Trees", 50, 300, 150)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=trees, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Random Forest")
    c2.metric("Accuracy", round(accuracy_score(y_test, preds), 3))
    c3.metric("AUC Score", round(roc_auc_score(y_test, probs), 3))

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
        fig = px.bar(importance, x="Feature", y="Importance", title="Feature Importance")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with col2:
        cm = confusion_matrix(y_test, preds)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )
        fig = px.imshow(cm_df, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.download_button(
        "Download Churn Predictions",
        df.to_csv(index=False),
        "churn_predictions.csv",
        "text/csv"
    )

with tab4:
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
    c1.metric("Forecast Demand", round(df["forecast_demand"].sum(), 2))
    c2.metric("Safety Stock", round(df["safety_stock"].sum(), 2))
    c3.metric("Reorder Qty", round(df["reorder_quantity"].sum(), 2))

    fig = px.bar(
        df.sort_values("reorder_quantity", ascending=False).head(30),
        x="customer_id",
        y="reorder_quantity",
        title="Top Reorder Recommendations"
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.dataframe(
        df[[
            "customer_id", "quantity", "inventory",
            "forecast_demand", "safety_stock", "reorder_quantity"
        ]],
        use_container_width=True
    )

    st.download_button(
        "Download Inventory Recommendations",
        df.to_csv(index=False),
        "inventory_recommendations.csv",
        "text/csv"
    )

with tab5:
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

    fig = px.scatter(df, x=chart_x, y=chart_y, title=f"{chart_x} vs {chart_y}")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.download_button(
        "Download Filtered Dataset",
        df.to_csv(index=False),
        "filtered_dataset.csv",
        "text/csv"
    )