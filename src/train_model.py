
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "processed" / "walmart_sales.parquet"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# Load prepared data
df = pd.read_parquet(DATA).sort_values("Date").copy()

# Calendar features
df["Month"] = df["Date"].dt.month
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["Year"] = df["Date"].dt.year

# Encode store type
df["Type"] = df["Type"].map({"A": 0, "B": 1, "C": 2})

features = [
    "Store", "Dept", "Size", "Type",
    "Month", "WeekOfYear", "IsHoliday"
]

# Chronological split
# Training: through June 2012
# Validation: July through October 2012
cutoff = pd.Timestamp("2012-07-01")

train = df[df["Date"] < cutoff].copy()
valid = df[df["Date"] >= cutoff].copy()

X_train = train[features]
y_train = train["Weekly_Sales"]

X_valid = valid[features]
y_valid = valid["Weekly_Sales"]

print("\n===== RETAILIQ FORECASTING =====")
print(f"Training records: {len(train):,}")
print(f"Validation records: {len(valid):,}")

# Baseline: average sales for each store and department
baseline_means = train.groupby(
    ["Store", "Dept"]
)["Weekly_Sales"].mean()

global_mean = y_train.mean()

baseline_predictions = [
    baseline_means.get((s, d), global_mean)
    for s, d in zip(valid["Store"], valid["Dept"])
]

baseline_mae = mean_absolute_error(
    y_valid, baseline_predictions
)

print(f"\nBaseline MAE: ${baseline_mae:,.2f}")

# Train Random Forest
print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=18,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_valid)

mae = mean_absolute_error(y_valid, predictions)
rmse = np.sqrt(mean_squared_error(y_valid, predictions))

print("\n===== MODEL PERFORMANCE =====")
print(f"Random Forest MAE: ${mae:,.2f}")
print(f"Random Forest RMSE: ${rmse:,.2f}")

improvement = (baseline_mae - mae) / baseline_mae * 100
print(f"MAE improvement vs baseline: {improvement:.2f}%")

# Save model and feature definitions
joblib.dump(
    {"model": model, "features": features},
    MODELS / "retailiq_random_forest.joblib"
)

print("\nSUCCESS! RetailIQ forecasting model saved.")
