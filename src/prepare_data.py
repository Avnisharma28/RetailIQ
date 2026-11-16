
from pathlib import Path
import pandas as pd

# Locate project folders
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "Raw" / "walmart-recruiting-store-sales-forecasting"
PROCESSED = ROOT / "Data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

# Load Walmart datasets
train = pd.read_csv(RAW / "train.csv", parse_dates=["Date"])
test = pd.read_csv(RAW / "test.csv", parse_dates=["Date"])
features = pd.read_csv(RAW / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(RAW / "stores.csv")

print("\n===== RETAILIQ DATA OVERVIEW =====")
print(f"Training records: {len(train):,}")
print(f"Test records: {len(test):,}")
print(f"Stores: {train['Store'].nunique()}")
print(f"Departments: {train['Dept'].nunique()}")
print(f"Date range: {train['Date'].min().date()} to {train['Date'].max().date()}")

# Merge training data with store details
sales = train.merge(
    stores, on="Store", how="left", validate="many_to_one"
)

# Add weather, economic, and promotional features
sales = sales.merge(
    features,
    on=["Store", "Date", "IsHoliday"],
    how="left",
    validate="many_to_one"
)

# Check data quality
assert len(sales) == len(train), "Unexpected row count after merging"
assert sales["Type"].notna().all(), "Missing store information"
assert sales["Temperature"].notna().all(), "Missing feature information"

print("\n===== DATA QUALITY REPORT =====")
print(sales.isnull().sum())
print(f"\nNegative sales records: {(sales['Weekly_Sales'] < 0).sum():,}")

# Save prepared data
output = PROCESSED / "walmart_sales.parquet"
sales.to_parquet(output, index=False)

print("\nSUCCESS! RetailIQ data preparation completed.")
print(f"Saved to: {output}")
