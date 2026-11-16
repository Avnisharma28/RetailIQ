
from pathlib import Path
import pandas as pd
import numpy as np

# Project directories
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "Data" / "processed" / "walmart_sales.parquet"
OUTPUT_PATH = ROOT / "Data" / "processed" / "walmart_features.parquet"

print("\n===== RETAILIQ FEATURE ENGINEERING =====")

# Load prepared Walmart data
df = pd.read_parquet(DATA_PATH)

# Ensure dates are correctly formatted
df["Date"] = pd.to_datetime(df["Date"])

# Sort by store, department, and date
df = df.sort_values(
    ["Store", "Dept", "Date"]
).reset_index(drop=True)

print(f"Original records: {len(df):,}")

# Calendar features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["Quarter"] = df["Date"].dt.quarter

# Historical sales features
groups = df.groupby(["Store", "Dept"])["Weekly_Sales"]

# Previous week's sales
df["Sales_Lag_1"] = groups.shift(1)

# Sales four weeks earlier
df["Sales_Lag_4"] = groups.shift(4)

# Sales eight weeks earlier
df["Sales_Lag_8"] = groups.shift(8)

# Sales 52 weeks earlier
df["Sales_Lag_52"] = groups.shift(52)

# Historical rolling averages
# Shift first so the current week's sales are excluded
df["Rolling_Mean_4"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .transform(lambda s: s.shift(1).rolling(4).mean())
)

df["Rolling_Mean_8"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .transform(lambda s: s.shift(1).rolling(8).mean())
)

# Historical sales volatility
df["Rolling_Std_8"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .transform(lambda s: s.shift(1).rolling(8).std())
)

# Identify missing historical features
lag_columns = [
    "Sales_Lag_1",
    "Sales_Lag_4",
    "Sales_Lag_8",
    "Sales_Lag_52",
    "Rolling_Mean_4",
    "Rolling_Mean_8",
    "Rolling_Std_8"
]

print("\n===== MISSING HISTORICAL FEATURES =====")
print(df[lag_columns].isna().sum())

# Preserve missing values rather than inventing history
# Tree-based models can handle these with suitable preprocessing

# Save engineered dataset
df.to_parquet(OUTPUT_PATH, index=False)

print("\n===== FEATURE ENGINEERING COMPLETE =====")
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Saved to: {OUTPUT_PATH}")
