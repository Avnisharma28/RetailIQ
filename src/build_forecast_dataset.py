
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "Data" / "processed" / "walmart_sales.parquet"
OUTPUT = ROOT / "Data" / "processed" / "forecast_4week.parquet"

print("\n===== RETAILIQ: FOUR-WEEK FORECAST DATASET =====")

df = pd.read_parquet(INPUT)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Store", "Dept", "Date"]
).reset_index(drop=True)

# Build a calendar-aligned dataset
# Missing store-department weeks remain missing, not zero.
all_dates = pd.date_range(
    df["Date"].min(),
    df["Date"].max(),
    freq="7D"
)

pairs = df[["Store", "Dept"]].drop_duplicates()

calendar = pairs.merge(
    pd.DataFrame({"Date": all_dates}),
    how="cross"
)

df = calendar.merge(
    df,
    on=["Store", "Dept", "Date"],
    how="left",
    validate="one_to_one"
)

# Only keep rows where the store-department
# actually has an observed sales value.
df["Observed"] = df["Weekly_Sales"].notna()

df = df.sort_values(
    ["Store", "Dept", "Date"]
).reset_index(drop=True)

groups = df.groupby(["Store", "Dept"])["Weekly_Sales"]

# Historical features available at forecast origin
df["Sales_Current"] = df["Weekly_Sales"]
df["Sales_Lag_1"] = groups.shift(1)
df["Sales_Lag_4"] = groups.shift(4)
df["Sales_Lag_8"] = groups.shift(8)
df["Sales_Lag_52"] = groups.shift(52)

for window in [4, 8, 12]:
    df[f"Rolling_Mean_{window}"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
        .transform(
            lambda s: s.rolling(
                window, min_periods=window
            ).mean()
        )
    )

# Calendar features
df["Month"] = df["Date"].dt.month
df["WeekOfYear"] = (
    df["Date"].dt.isocalendar().week.astype(int)
)
df["Quarter"] = df["Date"].dt.quarter

# Predict the total of the next 4 calendar weeks
future_weeks = [
    groups.shift(-i) for i in range(1, 5)
]

future_sales = pd.concat(future_weeks, axis=1)

# Only label examples with all 4 future weeks observed
df["Target_4Week_Sales"] = future_sales.sum(
    axis=1, min_count=4
)

# Exclude incomplete targets
df = df.dropna(
    subset=["Target_4Week_Sales"]
).copy()

# Keep only forecast origins with observed current sales
df = df[df["Observed"]].copy()

# Do not use future actual sales as model inputs.
# Future sales are used only as training labels.

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT, index=False)

print(f"Forecast examples: {len(df):,}")
print(f"Stores: {df['Store'].nunique()}")
print(f"Departments: {df['Dept'].nunique()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

print("\nMissing historical features:")
print(
    df[
        [
            "Sales_Lag_1",
            "Sales_Lag_4",
            "Sales_Lag_8",
            "Sales_Lag_52",
            "Rolling_Mean_4",
            "Rolling_Mean_8",
            "Rolling_Mean_12"
        ]
    ].isna().sum()
)

print("\nSUCCESS! Four-week forecasting dataset created.")
print(f"Saved to: {OUTPUT}")
