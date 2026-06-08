import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="RetailPulse (AI-Driven Analytics)",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BG = "#0a0e16"
PANEL = "#10151f"
BORDER = "#1e2733"
TEXT = "#c8d3e0"
MUTED = "#5d6b7d"
ACCENT = "#f0a92b"
POS = "#2dbd6e"
NEG = "#e5484d"

SEGMENT_COLORS = {
    "Champions": "#2dbd6e",
    "Loyal Customers": "#3b9eff",
    "Recent Low-Value": "#f0a92b",
    "At Risk / Lost": "#e5484d",
}

st.markdown(f"""
<style>
:root {{
 --bg:{BG}; --panel:{PANEL}; --border:{BORDER};
 --text:{TEXT}; --muted:{MUTED}; --accent:{ACCENT};
}}
.stApp {{ background-color: var(--bg); }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
* {{ font-family: Inter, sans-serif; }}
h1,h2,h3,h4 {{ color: var(--text); }}

.hdr {{
 display:flex; justify-content:space-between; align-items:flex-end;
 border-bottom:1px solid var(--border); padding-bottom:12px; margin-bottom:14px;
}}
.brand {{ font-size:22px; font-weight:700; color:var(--text); }}
.sq {{ color:var(--accent); }}
.brand small {{
 color:var(--muted); font-size:11px; letter-spacing:.18em;
 text-transform:uppercase; display:block; margin-top:4px;
}}
.meta {{
 font-family:monospace; font-size:11px; color:var(--muted);
 text-align:right; line-height:1.6;
}}
.meta b {{ color:var(--text); }}

.ticker {{
 display:flex; background:var(--panel); border:1px solid var(--border);
 border-radius:6px; overflow:hidden; margin-bottom:16px;
}}
.tk {{ flex:1; padding:12px 16px; border-right:1px solid var(--border); }}
.tk:last-child {{ border-right:none; }}
.tk span {{
 font-size:10px; letter-spacing:.13em; text-transform:uppercase; color:var(--muted);
}}
.tk b {{
 font-family:monospace; font-size:16px; color:var(--text); display:block; margin-top:4px;
}}
.tk b.acc {{ color:var(--accent); }}

.tilerow {{ display:flex; gap:10px; margin:8px 0 12px 0; }}
.tile {{
 flex:1; background:var(--panel); border:1px solid var(--border);
 border-radius:6px; padding:15px 16px;
}}
.tlabel {{
 font-size:10px; letter-spacing:.12em; text-transform:uppercase;
 color:var(--muted); margin-bottom:9px;
}}
.tval {{
 font-family:monospace; font-size:25px; font-weight:700;
 color:var(--text); line-height:1;
}}
.tsub {{ font-family:monospace; font-size:11px; margin-top:7px; }}
.pos {{ color:{POS}; }}
.neg {{ color:{NEG}; }}
.acc {{ color:{ACCENT}; }}
.mut {{ color:{MUTED}; }}

.sect {{
 font-size:11px; letter-spacing:.14em; text-transform:uppercase;
 color:var(--muted); margin:22px 0 9px 0; padding-left:9px;
 border-left:2px solid var(--accent);
}}
.note {{
 background:rgba(240,169,43,.05); border:1px solid var(--border);
 border-left:2px solid var(--accent); border-radius:4px; padding:12px 15px;
 font-size:13px; color:var(--text); margin-top:12px; line-height:1.55;
}}
.stTabs [data-baseweb="tab-list"] {{
 gap:0; border-bottom:1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
 background:transparent; border-radius:0; padding:10px 22px;
 font-size:12px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
}}
.stTabs [aria-selected="true"] {{
 color:var(--text); border-bottom:2px solid var(--accent);
}}
[data-testid="stDataFrame"] {{
 border:1px solid var(--border); border-radius:6px;
}}
</style>
""", unsafe_allow_html=True)


def gbp(x):
    if x >= 1e6:
        return f"£{x/1e6:.1f}M"
    if x >= 1e3:
        return f"£{x/1e3:.0f}K"
    return f"£{x:,.0f}"


def tile(label, value, sub="", tone="mut"):
    return (
        f'<div class="tile"><div class="tlabel">{label}</div>'
        f'<div class="tval">{value}</div>'
        f'<div class="tsub {tone}">{sub}</div></div>'
    )


def tilerow(tiles):
    st.markdown('<div class="tilerow">' + "".join(tiles) + '</div>', unsafe_allow_html=True)


def sect(text):
    st.markdown(f'<div class="sect">{text}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=320, legend=True):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=42, b=10),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family="monospace", color=TEXT, size=11),
        title_font=dict(size=13, color=MUTED),
        colorway=[ACCENT, "#3b9eff", POS, NEG, "#a78bfa"],
        showlegend=legend,
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


@st.cache_data
def load_data():
    rfm = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "rfm_clustered.csv"))
    churn = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "churn_scores.csv"))
    forecast = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "weekly_forecast.csv"))
    inventory = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "inventory_recommendations.csv"))

    forecast.columns = ["Week", "ForecastedDemand"]

    rfm = rfm.merge(
        churn[["CustomerID", "Churn_Probability"]],
        on="CustomerID",
        how="left"
    )

    return rfm, churn, forecast, inventory


try:
    rfm, churn, forecast, inventory = load_data()
except FileNotFoundError as e:
    st.error(f"Data file not found. Run from project root or dashboard folder. Details: {e}")
    st.stop()

TOTAL_REV = rfm["Monetary"].sum()
AVG_VAL = rfm["Monetary"].mean()
N_CHAMP = int((rfm["Cluster_Name"] == "Champions").sum())
N_RISK = int((churn["Churn_Probability"] > 0.7).sum())
CHURN_RATE = churn["Churned"].mean() * 100
FC_AVG = forecast["ForecastedDemand"].mean()

st.markdown(f"""
<div class="hdr">
  <div class="brand">RetailPulse <span class="sq">▲</span>
    <small>Retail Intelligence Terminal</small>
  </div>
  <div class="meta">
    UK ONLINE RETAIL · DEC 2010–DEC 2011<br>
    SOURCE <b>397,884 TXNS</b> · <b>{len(rfm):,} CUSTOMERS</b>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="ticker">
  <div class="tk"><span>Customers</span><b>{len(rfm):,}</b></div>
  <div class="tk"><span>Revenue</span><b class="acc">{gbp(TOTAL_REV)}</b></div>
  <div class="tk"><span>Champions</span><b>{N_CHAMP:,}</b></div>
  <div class="tk"><span>Churn risk >0.7</span><b>{N_RISK:,}</b></div>
  <div class="tk"><span>Fcst demand / wk</span><b>{FC_AVG:,.0f}</b></div>
  <div class="tk"><span>Portfolio churn</span><b>{CHURN_RATE:.0f}%</b></div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.tabs(
    ["Overview", "Segments", "Demand Forecast", "Churn Risk", "Inventory"]
)

with t1:
    champ_rev = rfm.loc[rfm["Cluster_Name"] == "Champions", "Monetary"].sum()
    champ_rev_sh = champ_rev / TOTAL_REV * 100
    champ_cnt_sh = N_CHAMP / len(rfm) * 100

    tilerow([
        tile("Total Revenue", gbp(TOTAL_REV), "lifetime, cleaned txns"),
        tile("Avg Customer Value", gbp(AVG_VAL), "per customer"),
        tile("Champions", f"{N_CHAMP:,}", f"{champ_cnt_sh:.0f}% of base", "pos"),
        tile("High Churn Risk", f"{N_RISK:,}", "prob > 0.70 — act now", "neg"),
    ])

    sect("Customer base vs revenue concentration")
    c1, c2 = st.columns([3, 2])

    with c1:
        seg = rfm["Cluster_Name"].value_counts().reset_index()
        seg.columns = ["Segment", "Customers"]
        fig = px.bar(
            seg,
            x="Customers",
            y="Segment",
            orientation="h",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            title="CUSTOMERS BY SEGMENT"
        )
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        st.plotly_chart(style_fig(fig, 300, False), use_container_width=True)

    with c2:
        rev = rfm.groupby("Cluster_Name")["Monetary"].sum().reset_index()
        fig = px.pie(
            rev,
            names="Cluster_Name",
            values="Monetary",
            hole=0.62,
            color="Cluster_Name",
            color_discrete_map=SEGMENT_COLORS,
            title="REVENUE SHARE"
        )
        fig.update_traces(textinfo="percent", textfont_size=11)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    note(
        f"<b>Concentration:</b> Champions are {champ_cnt_sh:.0f}% of customers "
        f"but ~{champ_rev_sh:.0f}% of revenue. Retention spend is best aimed here "
        f"and at the <b>At Risk / Lost</b> tier."
    )

with t2:
    pick = st.selectbox("Segment filter", ["All"] + sorted(rfm["Cluster_Name"].unique()))
    view = rfm if pick == "All" else rfm[rfm["Cluster_Name"] == pick]

    tilerow([
        tile("Customers", f"{len(view):,}", "in selection"),
        tile("Avg Spend", gbp(view["Monetary"].mean()), "monetary"),
        tile("Avg Orders", f"{view['Frequency'].mean():.1f}", "frequency"),
        tile("Avg Recency", f"{view['Recency'].mean():.0f}d", "since last buy"),
    ])

    sect("RFM space")
    c1, c2 = st.columns([3, 2])

    with c1:
        fig = px.scatter(
            view,
            x="Recency",
            y="Monetary",
            color="Cluster_Name",
            color_discrete_map=SEGMENT_COLORS,
            opacity=0.6,
            title="EACH POINT = ONE CUSTOMER"
        )
        fig.update_yaxes(range=[0, 20000])
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    with c2:
        avg = view.groupby("Cluster_Name")["Monetary"].mean().sort_values().reset_index()
        fig = px.bar(
            avg,
            x="Monetary",
            y="Cluster_Name",
            orientation="h",
            color="Cluster_Name",
            color_discrete_map=SEGMENT_COLORS,
            title="AVG SPEND PER SEGMENT"
        )
        st.plotly_chart(style_fig(fig, 380, False), use_container_width=True)

    sect("Customer detail")
    st.dataframe(
        view[["CustomerID", "Recency", "Frequency", "Monetary", "Cluster_Name", "Churn_Probability"]]
        .sort_values("Monetary", ascending=False)
        .head(100),
        use_container_width=True,
        height=240
    )

with t3:
    tilerow([
        tile("Avg Weekly Demand", f"{FC_AVG:,.0f}", "units"),
        tile("Peak Week", f"{forecast['ForecastedDemand'].max():,.0f}", "units", "acc"),
        tile("Model", "Holt-Winters", "damped trend"),
        tile("Accuracy", "8.7%", "MAPE", "pos"),
    ])

    sect("Forecasted weekly demand")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast["Week"],
        y=forecast["ForecastedDemand"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=7, color=ACCENT),
        fill="tozeroy",
        fillcolor="rgba(240,169,43,0.08)"
    ))
    fig.update_layout(title="UNITS PER WEEK")
    st.plotly_chart(style_fig(fig, 320, False), use_container_width=True)

    sect("Forecast table")
    show = forecast.copy()
    show["ForecastedDemand"] = show["ForecastedDemand"].round(0).astype(int)
    st.dataframe(show, use_container_width=True, height=220)

with t4:
    thr = st.slider("Risk threshold", 0.0, 1.0, 0.7, 0.05)
    at_risk = churn[churn["Churn_Probability"] >= thr].sort_values(
        "Churn_Probability",
        ascending=False
    )
    avg_risk = at_risk["Churn_Probability"].mean() if len(at_risk) else 0

    tilerow([
        tile("Flagged", f"{len(at_risk):,}", f"prob ≥ {thr:.2f}", "neg"),
        tile("Share of Base", f"{len(at_risk)/len(churn)*100:.0f}%", "customers"),
        tile("Avg Risk Score", f"{avg_risk:.2f}", "flagged"),
        tile("Model AUC", "0.72", "leakage-free", "acc"),
    ])

    c1, c2 = st.columns([3, 2])

    with c1:
        sect("Churn probability distribution")
        fig = px.histogram(churn, x="Churn_Probability", nbins=34)
        fig.update_traces(marker_color="#3b9eff")
        fig.add_vline(x=thr, line_dash="dash", line_color=ACCENT)
        st.plotly_chart(style_fig(fig, 320, False), use_container_width=True)

    with c2:
        sect("Risk bands")
        bands = pd.cut(
            churn["Churn_Probability"],
            [0, .3, .7, 1.0],
            labels=["Low", "Medium", "High"]
        )
        bc = bands.value_counts().reindex(["Low", "Medium", "High"]).reset_index()
        bc.columns = ["Band", "Customers"]
        fig = px.bar(
            bc,
            x="Customers",
            y="Band",
            orientation="h",
            color="Band",
            color_discrete_sequence=[POS, ACCENT, NEG]
        )
        st.plotly_chart(style_fig(fig, 320, False), use_container_width=True)

    sect("Highest-risk customers")
    st.dataframe(at_risk.head(100), use_container_width=True, height=230)

with t5:
    avg_d = inventory["ForecastedDemand"].mean()
    avg_o = inventory["RecommendedOrder"].mean()
    buf = avg_o - avg_d

    tilerow([
        tile("Avg Weekly Demand", f"{avg_d:,.0f}", "forecast units"),
        tile("Avg Recommended Order", f"{avg_o:,.0f}", "incl. buffer", "acc"),
        tile("Safety Buffer", f"{buf:,.0f}", "95% service level", "pos"),
        tile("Lead Time", "1 wk", "assumed"),
    ])

    sect("Forecast vs recommended order")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=inventory["Week"],
        y=inventory["ForecastedDemand"],
        name="Forecast demand",
        marker_color="#3b4b5e"
    ))
    fig.add_trace(go.Bar(
        x=inventory["Week"],
        y=inventory["RecommendedOrder"],
        name="Recommended order",
        marker_color=ACCENT
    ))
    fig.update_layout(barmode="group", title="UNITS PER WEEK")
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    sect("Reorder table")
    show = inventory.copy()
    for col in ["ForecastedDemand", "RecommendedOrder"]:
        show[col] = show[col].round(0).astype(int)
    st.dataframe(show, use_container_width=True, height=220)

st.markdown(
    f"""
    <div style="border-top:1px solid {BORDER};margin-top:24px;padding-top:10px;
    font-size:11px;color:{MUTED};font-family:monospace;">
    SHAHZADA KOHINOOR · RETAILPULSE · ZIDIO DEVELOPMENT · DATA SCIENCE & ANALYTICS PORTFOLIO
    </div>
    """,
    unsafe_allow_html=True
)