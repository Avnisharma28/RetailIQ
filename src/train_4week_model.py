
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "processed" / "forecast_4week.parquet"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA)
df["Date"] = pd.to_datetime(df["Date"])

# Encode store type
df["Type_Code"] = df["Type"].map({
    "A": 0, "B": 1, "C": 2
})

features = [
    "Store",
    "Dept",
    "Size",
    "Type_Code",
    "Month",
    "WeekOfYear",
    "Quarter",
    "IsHoliday",
    "Sales_Current",
    "Sales_Lag_1",
    "Sales_Lag_4",
    "Sales_Lag_8",
    "Sales_Lag_52",
    "Rolling_Mean_4",
    "Rolling_Mean_8",
    "Rolling_Mean_12"
]

# Forecast origins:
# Train through June 1, 2012
# Validation starts July 6, 2012
# This creates a 5-week gap between origin dates.
train = df[
    df["Date"] <= "2012-06-01"
].copy()

valid = df[
    df["Date"] >= "2012-07-06"
].copy()

X_train = train[features]
y_train = train["Target_4Week_Sales"]

X_valid = valid[features]
y_valid = valid["Target_4Week_Sales"]

print("\n===== RETAILIQ FOUR-WEEK FORECASTING =====")
print(f"Training records: {len(train):,}")
print(f"Validation records: {len(valid):,}")

# Baseline: assume next 4 weeks resemble
# the most recent 4 observed weeks.
baseline = valid["Rolling_Mean_4"] * 4

# Evaluate baseline only where its history exists.
baseline_mask = baseline.notna()

baseline_mae = mean_absolute_error(
    y_valid[baseline_mask],
    baseline[baseline_mask]
)

print(f"\nFour-week baseline MAE: ${baseline_mae:,.2f}")

# Train gradient boosting model
print("\nTraining advanced forecasting model...")

model = HistGradientBoostingRegressor(
    max_iter=250,
    learning_rate=0.06,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_valid)

# Compare models on the same rows
model_mae = mean_absolute_error(
    y_valid[baseline_mask],
    predictions[baseline_mask]
)

model_rmse = np.sqrt(mean_squared_error(
    y_valid[baseline_mask],
    predictions[baseline_mask]
))

improvement = (
    (baseline_mae - model_mae) / baseline_mae
) * 100

print("\n===== MODEL PERFORMANCE =====")
print(f"Baseline MAE: ${baseline_mae:,.2f}")
print(f"Advanced Model MAE: ${model_mae:,.2f}")
print(f"Advanced Model RMSE: ${model_rmse:,.2f}")
print(f"Improvement vs baseline: {improvement:.2f}%")

# Save validation predictions for error analysis
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

results = valid[
    ["Store", "Dept", "Date", "Target_4Week_Sales"]
].copy()

results["Predicted_4Week_Sales"] = predictions
results["Baseline_4Week_Sales"] = baseline.values

results["Absolute_Error"] = (
    results["Target_4Week_Sales"]
    - results["Predicted_4Week_Sales"]
).abs()

results["Percentage_Error"] = np.where(
    results["Target_4Week_Sales"].abs() > 100,
    results["Absolute_Error"]
    / results["Target_4Week_Sales"].abs() * 100,
    np.nan
)

results.to_csv(
    REPORTS / "forecast_4week_predictions.csv",
    index=False
)

print("\nValidation predictions saved.")
print("Average absolute error by store:")

print(
    results.groupby("Store")["Absolute_Error"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

joblib.dump(
    {"model": model, "features": features},
    MODELS / "retailiq_4week_model.joblib"
)

print("\nSUCCESS! Four-week forecasting model saved.")
