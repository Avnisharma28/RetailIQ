
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# Project paths
ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports" / "forecast_4week_predictions.csv"
STORE_REPORT = ROOT / "reports" / "store_forecast_performance.csv"

st.set_page_config(
    page_title="RetailIQ | Retail Intelligence",
    page_icon="📊",
    layout="wide"
)

# RetailIQ dashboard styling
st.markdown("""
<style>
    /* Main application background */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Headings */
    h1 {
        font-size: 2.6rem !important;
        font-weight: 750 !important;
        letter-spacing: -1px;
        color: #F8FAFC !important;
    }

    h2, h3 {
        color: #E2E8F0 !important;
        font-weight: 650 !important;
    }

    /* KPI cards */
    [data-testid="stMetric"] {
        background: #172033;
        border: 1px solid #29364B;
        padding: 22px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #94A3B8;
        font-size: 0.9rem;
    }

    [data-testid="stMetricValue"] {
        color: #38BDF8;
        font-weight: 700;
    }

    /* Divider */
    hr {
        border-color: #263244;
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #29364B;
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["Date"])
    stores = pd.read_csv(STORE_REPORT)
    return predictions, stores

# Consistent Plotly styling for RetailIQ
def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        font=dict(
            family="Arial",
            color="#CBD5E1",
            size=13
        ),
        title_font=dict(
            size=18,
            color="#F8FAFC"
        ),
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="#263244",
            zeroline=False
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        ),
        colorway=["#38BDF8", "#818CF8", "#34D399"]
    )
    return fig

df, store_report = load_data()

# Sidebar
st.sidebar.title("RetailIQ")
st.sidebar.caption("AI-Powered Retail Forecasting")


# Store and department filters
st.sidebar.markdown("### Forecast Filters")

store_options = sorted(df["Store"].unique().tolist())

selected_store = st.sidebar.selectbox(
    "Select Store",
    options=["All Stores"] + store_options
)

if selected_store == "All Stores":
    store_filtered = df.copy()
else:
    store_filtered = df[
        df["Store"] == selected_store
    ].copy()

department_options = sorted(
    store_filtered["Dept"].unique().tolist()
)

selected_department = st.sidebar.selectbox(
    "Select Department",
    options=["All Departments"] + department_options
)

if selected_department == "All Departments":
    filtered = store_filtered.copy()
else:
    filtered = store_filtered[
        store_filtered["Dept"] == selected_department
    ].copy()


# Header
st.title("RetailIQ")
st.subheader("Retail Sales Forecasting & Decision Intelligence")
st.caption("Historical Walmart sales | Four-week forecasting")

if filtered.empty:
    st.warning("Select at least one store to view results.")
    st.stop()

# Metrics
mae = (
    filtered["Predicted_4Week_Sales"]
    - filtered["Target_4Week_Sales"]
).abs().mean()

total_absolute_error = (
    filtered["Predicted_4Week_Sales"]
    - filtered["Target_4Week_Sales"]
).abs().sum()

total_actual = filtered["Target_4Week_Sales"].abs().sum()

wape = (
    total_absolute_error / total_actual * 100
    if total_actual > 0 else float("nan")
)

bias = (
    filtered["Predicted_4Week_Sales"]
    - filtered["Target_4Week_Sales"]
).mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Forecast MAE", f"${mae:,.0f}")
col2.metric("Forecast WAPE", f"{wape:.2f}%")
col3.metric(
    "Average Bias",
    f"-${abs(bias):,.0f}" if bias < 0 else f"${bias:,.0f}"
)
col4.metric("Forecast Examples", f"{len(filtered):,}")

st.divider()

# ===== MODEL PERFORMANCE COMPARISON =====

st.subheader("RetailIQ | Overall Model Performance (All Stores)")

comparison = pd.read_csv("reports/model_comparison.csv").iloc[0]

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Baseline MAE",
    f"${comparison['Baseline_MAE']:,.0f}"
)

metric2.metric(
    "ML Model MAE",
    f"${comparison['ML_MAE']:,.0f}"
)

metric3.metric(
    "MAE Improvement",
    f"{comparison['MAE_Improvement_Percent']:.2f}%"
)

model_chart = pd.DataFrame({
    "Model": ["Baseline Forecast", "ML Forecast"],
    "MAE": [
        comparison["Baseline_MAE"],
        comparison["ML_MAE"]
    ]
})

fig_comparison = px.bar(
    model_chart,
    x="Model",
    y="MAE",
    color="Model",
    text="MAE",
    color_discrete_map={
        "Baseline Forecast": "#718096",
        "ML Forecast": "#38BDF8"
    }
)

fig_comparison.update_traces(
    texttemplate="$%{text:,.0f}",
    textposition="outside",
    hovertemplate=(
        "<b>%{x}</b>"
        "<extra></extra>"
        "<br>Mean Absolute Error: $%{y:,.2f}"
    )
)

fig_comparison.update_layout(
    showlegend=False,
    yaxis_title="Mean Absolute Error ($)",
    xaxis_title=""
)

st.plotly_chart(
    style_chart(fig_comparison),
    width="stretch"
)

st.caption(
    f"Compared on {int(comparison['Forecasts_Compared']):,} "
    "historical validation forecasts with available baseline values. "
    "Lower MAE indicates better forecasting accuracy."
)

st.divider()

# Automated business insights
st.subheader("RetailIQ | Business Insights")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("#### Forecast Accuracy")

    if wape < 10:
        st.success(
            f"Forecast WAPE is {wape:.2f}%. "
            "The selected segment has relatively low "
            "forecasting error."
        )
    elif wape < 20:
        st.warning(
            f"Forecast WAPE is {wape:.2f}%. "
            "This segment has moderate forecasting error "
            "and may benefit from additional analysis."
        )
    else:
        st.error(
            f"Forecast WAPE is {wape:.2f}%. "
            "This segment has elevated forecasting error. "
            "Review demand patterns and model performance."
        )

with insight_col2:
    st.markdown("#### Forecast Bias")

    if bias > 500:
        st.warning(
            f"Average overforecast: ${bias:,.0f}. "
            "Predictions tend to exceed actual sales. "
            "Monitor potential excess inventory exposure."
        )
    elif bias < -500:
        st.warning(
            f"Average underforecast: ${abs(bias):,.0f}. "
            "Predictions tend to fall below actual sales. "
            "Monitor potential stockout exposure."
        )
    else:
        st.info(
            f"Average forecast bias: {'-$' + format(abs(bias), ',.0f') if bias < 0 else '$' + format(bias, ',.0f')}. "
            "The average signed error is relatively small."
        )

st.caption(
    "Insights are descriptive and based on historical "
    "validation forecasts. Thresholds are illustrative, "
    "not validated operational risk limits."
)

st.divider()

# Actual vs predicted sales
st.subheader("Actual vs Predicted Sales")

weekly = (
    filtered.groupby("Date", as_index=False)[
        ["Target_4Week_Sales", "Predicted_4Week_Sales"]
    ]
    .sum()
)


fig = px.line(
    weekly,
    x="Date",
    y=["Target_4Week_Sales", "Predicted_4Week_Sales"],
    labels={
        "value": "Four-week Sales ($)",
        "Date": "Forecast Origin",
        "variable": "Series"
    }
)

fig.for_each_trace(
    lambda trace: trace.update(
        name={
            "Target_4Week_Sales": "Actual Sales",
            "Predicted_4Week_Sales": "Forecast Sales"
        }.get(trace.name, trace.name)
    )
)

fig.update_layout(hovermode="x unified")

fig.update_layout(hovermode="x unified")
st.plotly_chart(style_chart(fig), use_container_width=True)

# Store comparison
st.subheader("Store Forecasting Performance")


# Recalculate store performance for the selected department
store_filtered = filtered.copy()

store_filtered["Absolute_Error"] = (
    store_filtered["Predicted_4Week_Sales"]
    - store_filtered["Target_4Week_Sales"]
).abs()

store_filtered = store_filtered.groupby("Store").agg(
    Total_Absolute_Error=("Absolute_Error", "sum"),
    Total_Actual=("Target_4Week_Sales", "sum")
).reset_index()

store_filtered["WAPE"] = (
    store_filtered["Total_Absolute_Error"]
    / store_filtered["Total_Actual"].abs().replace(0, float("nan"))
    * 100
)


# Store Forecasting Performance Visualization

if len(store_filtered) > 1:

    # Compare forecasting accuracy across stores
    fig2 = px.bar(
        store_filtered.sort_values("WAPE", ascending=False),
        x="Store",
        y="WAPE",
        color="WAPE",
        labels={
            "Store": "Store",
            "WAPE": "Forecast WAPE (%)"
        }
    )

    fig2.update_traces(
        hovertemplate=(
            "Store: %{x}<br>"
            "Forecast WAPE: %{y:.2f}%"
            "<extra></extra>"
        )
    )

    st.plotly_chart(
        style_chart(fig2),
        use_container_width=True
    )

elif len(store_filtered) == 1:

    # Show a focused performance summary for one store
    selected_store = store_filtered.iloc[0]

    store_wape = selected_store["WAPE"]
    store_accuracy = 100 - store_wape

    st.markdown(
        f"### Store {int(selected_store['Store'])} Performance"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Forecast WAPE",
        f"{store_wape:.2f}%"
    )

    col2.metric(
        "100% − WAPE",
        f"{store_accuracy:.2f}%"
    )

    st.caption(
        "100% − WAPE is a descriptive complement to "
        "forecast error, not a formal accuracy score."
    )

else:
    st.info("No store performance data available.")

# Forecast risk monitoring
st.markdown("### Forecast Risk Monitor")

risk_stores = store_filtered.sort_values(
    "WAPE",
    ascending=False
).head(5).copy()

risk_stores["Risk_Level"] = risk_stores["WAPE"].apply(
    lambda x: (
        "High" if x >= 20
        else "Moderate" if x >= 10
        else "Low"
    )
)

risk_stores["Recommended_Action"] = risk_stores[
    "Risk_Level"
].map({
    "High": "Prioritize forecast review",
    "Moderate": "Investigate demand variation",
    "Low": "Continue routine monitoring"
})

st.dataframe(
    risk_stores[
        ["Store", "WAPE", "Risk_Level", "Recommended_Action"]
    ].rename(columns={
        "Store": "Store ID",
        "WAPE": "Forecast Error (%)",
        "Risk_Level": "Risk Level",
        "Recommended_Action": "Suggested Action"
    }),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Forecast Error (%)": st.column_config.NumberColumn(
            format="%.2f%%"
        )
    }
)

st.caption(
    "Risk categories are based on illustrative "
    "forecast-error thresholds, not validated "
    "inventory or financial risk estimates."
)

# RetailIQ Scenario Simulator
st.subheader("RetailIQ | Scenario Simulator")

st.caption(
    "Explore hypothetical demand changes using "
    "historical model predictions as a starting point."
)

scenario_change = st.slider(
    "Assumed Demand Change (%)",
    min_value=-30,
    max_value=30,
    value=0,
    step=5
)

# Use the most recent validation forecast origin
latest_date = filtered["Date"].max()

latest_forecasts = filtered[
    filtered["Date"] == latest_date
].copy()

base_forecast = latest_forecasts[
    "Predicted_4Week_Sales"
].sum()

adjusted_forecast = base_forecast * (
    1 + scenario_change / 100
)

forecast_difference = adjusted_forecast - base_forecast

col_a, col_b, col_c = st.columns(3)

col_a.metric(
    "Base Forecast",
    f"${base_forecast:,.0f}"
)

col_b.metric(
    "Scenario Forecast",
    f"${adjusted_forecast:,.0f}"
)

col_c.metric(
    "Change in Forecast",
    f"${forecast_difference:,.0f}"
)

scenario_df = pd.DataFrame({
    "Scenario": ["Base Forecast", "Adjusted Scenario"],
    "Sales": [base_forecast, adjusted_forecast]
})

scenario_fig = px.bar(
    scenario_df,
    x="Scenario",
    y="Sales",
    color="Scenario",
    text="Sales",
    color_discrete_map={
        "Base Forecast": "#38BDF8",
        "Adjusted Scenario": "#34D399"
    }
)


scenario_fig.update_traces(
    texttemplate="$%{y:,.0f}",
    textposition="outside",
    hovertemplate="Forecast Sales: $%{y:,.0f}<extra></extra>"
)


scenario_fig.update_layout(
    showlegend=False,
    yaxis_title="Four-week sales ($)"
)

st.plotly_chart(
    style_chart(scenario_fig),
    use_container_width=True
)

st.caption(
    f"Scenario based on the historical forecast origin "
    f"{latest_date:%B %d, %Y}. "
    "This is a hypothetical proportional adjustment, "
    "not a new ML prediction or causal estimate."
)

st.divider()

# Error distribution
st.subheader("Forecast Error Distribution")

fig3 = px.histogram(
    filtered,
    x="Absolute_Error",
    nbins=50,
    labels={"Absolute_Error": "Absolute Forecast Error ($)"}
)

st.plotly_chart(style_chart(fig3), use_container_width=True)


# Forecast Details Table
st.subheader("Forecast Details")

display_df = filtered.sort_values(
    "Date", ascending=False
).copy()

# Format the date
display_df["Date"] = pd.to_datetime(
    display_df["Date"]
).dt.strftime("%b %d, %Y")

# Select and rename columns
column_names = {
    "Store": "Store",
    "Dept": "Department",
    "Date": "Forecast Date",
    "Target_4Week_Sales": "Actual Sales ($)",
    "Predicted_4Week_Sales": "Predicted Sales ($)",
    "Baseline_4Week_Sales": "Baseline Sales ($)",
    "Absolute_Error": "Absolute Error ($)",
    "Percentage_Error": "Error (%)",
}

display_df = display_df[
    list(column_names.keys())
].rename(columns=column_names)
# Display missing baseline values as blank cells
display_df["Baseline Sales ($)"] = (
    pd.to_numeric(
        display_df["Baseline Sales ($)"],
        errors="coerce"
    ).astype("float64")
)
# Display a professionally formatted table
st.dataframe(
    display_df,
    column_config={
        "Actual Sales ($)": st.column_config.NumberColumn(
            "Actual Sales ($)", format="$%.2f"
        ),
        "Predicted Sales ($)": st.column_config.NumberColumn(
            "Predicted Sales ($)", format="$%.2f"
        ),
        "Baseline Sales ($)": st.column_config.NumberColumn(
    "Baseline Sales ($)",
    format="$%.2f",
    default=None,
),
        "Absolute Error ($)": st.column_config.NumberColumn(
            "Absolute Error ($)", format="$%.2f"
        ),
        "Error (%)": st.column_config.NumberColumn(
            "Error (%)", format="%.2f%%"
        ),
    },
    use_container_width=True,
    hide_index=True,
)

