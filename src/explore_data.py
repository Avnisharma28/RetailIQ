
from pathlib import Path
import pandas as pd
import plotly.express as px

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "processed" / "walmart_sales.parquet"
OUTPUT = ROOT / "reports" / "figures"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Load prepared Walmart data
df = pd.read_parquet(DATA)

print("\n===== RETAILIQ EXPLORATORY ANALYSIS =====")
print(f"Total records: {len(df):,}")
print(f"Stores: {df['Store'].nunique()}")
print(f"Departments: {df['Dept'].nunique()}")

# 1. Total weekly sales trends
weekly = (
    df.groupby("Date", as_index=False)["Weekly_Sales"]
    .sum()
    .sort_values("Date")
)

fig = px.line(
    weekly,
    x="Date",
    y="Weekly_Sales",
    title="Walmart Total Weekly Sales",
    labels={"Weekly_Sales": "Weekly Sales ($)"}
)
fig.write_html(OUTPUT / "weekly_sales.html")

# 2. Top 10 stores by total sales
stores = (
    df.groupby("Store", as_index=False)["Weekly_Sales"]
    .sum()
    .nlargest(10, "Weekly_Sales")
)

fig = px.bar(
    stores,
    x="Store",
    y="Weekly_Sales",
    title="Top 10 Walmart Stores by Historical Sales",
    labels={"Weekly_Sales": "Total Sales ($)"}
)
fig.update_xaxes(type="category")
fig.write_html(OUTPUT / "top_stores.html")

# 3. Holiday vs non-holiday weekly sales
# Compare average sales per store-week, not per department row
store_weeks = (
    df.groupby(["Store", "Date", "IsHoliday"], as_index=False)
    ["Weekly_Sales"].sum()
)

holiday = (
    store_weeks.groupby("IsHoliday", as_index=False)
    ["Weekly_Sales"].mean()
)

holiday["Week Type"] = holiday["IsHoliday"].map({
    True: "Holiday",
    False: "Non-Holiday"
})

fig = px.bar(
    holiday,
    x="Week Type",
    y="Weekly_Sales",
    title="Average Store-Week Sales: Holiday vs Non-Holiday",
    labels={"Weekly_Sales": "Average Store-Week Sales ($)"}
)
fig.write_html(OUTPUT / "holiday_sales.html")

# 4. Sales by store type
store_type = (
    df.groupby(["Type", "Store", "Date"], as_index=False)
    ["Weekly_Sales"].sum()
)

store_type_avg = (
    store_type.groupby("Type", as_index=False)
    ["Weekly_Sales"].mean()
)

fig = px.bar(
    store_type_avg,
    x="Type",
    y="Weekly_Sales",
    title="Average Weekly Sales per Store by Store Type",
    labels={"Weekly_Sales": "Average Weekly Sales ($)"}
)
fig.write_html(OUTPUT / "store_type_sales.html")

# Print business insights
print("\n===== BUSINESS INSIGHTS =====")
print("\nTop 10 stores:")
print(stores.to_string(index=False))

print("\nAverage sales by holiday status:")
print(holiday.to_string(index=False))

print("\nAverage sales by store type:")
print(store_type_avg.to_string(index=False))

print("\nSUCCESS! Four interactive charts created.")
print(f"Charts saved to: {OUTPUT}")
