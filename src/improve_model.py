
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "processed" / "walmart_sales.parquet"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# Load historical Walmart sales
df = pd.read_parquet(DATA).sort_values("Date").copy()

# Calendar features
df["Month"] = df["Date"].dt.month
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

# Encode store type
df["Type"] = df["Type"].map({
    "A": 0,
    "B": 1,
    "C": 2
})

# Use the same features as our first model
features = [
    "Store",
    "Dept",
    "Size",
    "Type",
    "Month",
    "WeekOfYear",
    "IsHoliday"
]

# Same chronological validation split
cutoff = pd.Timestamp("2012-07-01")

train = df[df["Date"] < cutoff].copy()
valid = df[df["Date"] >= cutoff].copy()

X_train = train[features]
y_train = train["Weekly_Sales"]

X_valid = valid[features]
y_valid = valid["Weekly_Sales"]

print("\n===== RETAILIQ XGBOOST EXPERIMENT =====")
print(f"Training records: {len(train):,}")
print(f"Validation records: {len(valid):,}")

# Train XGBoost
print("\nTraining XGBoost model...")

model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_valid)

mae = mean_absolute_error(y_valid, predictions)
rmse = np.sqrt(mean_squared_error(y_valid, predictions))

# Previous model results
baseline_mae = 2495.09
random_forest_mae = 1867.11

improvement = (
    (random_forest_mae - mae) / random_forest_mae
) * 100

print("\n===== MODEL PERFORMANCE =====")
print(f"Historical Baseline MAE: ${baseline_mae:,.2f}")
print(f"Random Forest MAE: ${random_forest_mae:,.2f}")
print(f"XGBoost MAE: ${mae:,.2f}")
print(f"XGBoost RMSE: ${rmse:,.2f}")
print(f"Improvement vs Random Forest: {improvement:.2f}%")

# Save trained model
joblib.dump(
    {"model": model, "features": features},
    MODELS / "retailiq_xgboost.joblib"
)

# Save predictions for later analysis
results = valid[
    ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]
].copy()

results["Predicted_Sales"] = predictions
results["Absolute_Error"] = abs(
    results["Weekly_Sales"] - results["Predicted_Sales"]
)

results.to_csv(
    REPORTS / "xgboost_predictions.csv",
    index=False
)

# Display most important features
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(importance.to_string(index=False))

print("\nSUCCESS! XGBoost experiment completed.")
print("Model and predictions saved.")
