
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "reports" / "forecast_4week_predictions.csv"
OUTPUT = ROOT / "reports" / "store_forecast_performance.csv"

print("\n===== RETAILIQ STORE PERFORMANCE ANALYSIS =====")

df = pd.read_csv(INPUT)

df["Signed_Error"] = (
    df["Predicted_4Week_Sales"]
    - df["Target_4Week_Sales"]
)

df["Absolute_Error"] = df["Signed_Error"].abs()

# Summarize performance by store
store_report = df.groupby("Store").agg(
    Forecasts=("Absolute_Error", "count"),
    MAE=("Absolute_Error", "mean"),
    Total_Actual=("Target_4Week_Sales", "sum"),
    Total_Predicted=("Predicted_4Week_Sales", "sum"),
    Mean_Bias=("Signed_Error", "mean")
).reset_index()

# Weighted absolute percentage error
# Use sum of absolute errors, not absolute net error
absolute_error_by_store = (
    df.groupby("Store")["Absolute_Error"]
    .sum()
    .reset_index(name="Total_Absolute_Error")
)

store_report = store_report.merge(
    absolute_error_by_store,
    on="Store"
)

store_report["WAPE"] = np.where(
    store_report["Total_Actual"].abs() > 0,
    store_report["Total_Absolute_Error"]
    / store_report["Total_Actual"].abs() * 100,
    np.nan
)

# Positive bias = overforecasting
# Negative bias = underforecasting
store_report["Bias_Direction"] = np.where(
    store_report["Mean_Bias"] > 0,
    "Overforecasting",
    "Underforecasting"
)

store_report = store_report.sort_values(
    "WAPE",
    ascending=False
)

store_report.to_csv(OUTPUT, index=False)

print("\n===== TOP 10 STORES BY RELATIVE ERROR =====")

print(
    store_report[
        ["Store", "Forecasts", "MAE", "WAPE",
         "Mean_Bias", "Bias_Direction"]
    ].head(10).round(2).to_string(index=False)
)

print("\n===== OVERALL MODEL PERFORMANCE =====")

overall_wape = (
    df["Absolute_Error"].sum()
    / df["Target_4Week_Sales"].abs().sum()
) * 100

print(f"Overall WAPE: {overall_wape:.2f}%")

# ===== BASELINE VS ML MODEL PERFORMANCE =====

comparison_df = df.dropna(
    subset=[
        "Target_4Week_Sales",
        "Predicted_4Week_Sales",
        "Baseline_4Week_Sales"
    ]
).copy()

actual = comparison_df["Target_4Week_Sales"]
predicted = comparison_df["Predicted_4Week_Sales"]
baseline = comparison_df["Baseline_4Week_Sales"]

ml_mae = (actual - predicted).abs().mean()
baseline_mae = (actual - baseline).abs().mean()

total_actual = actual.abs().sum()

ml_wape = (
    (actual - predicted).abs().sum()
    / total_actual * 100
)

baseline_wape = (
    (actual - baseline).abs().sum()
    / total_actual * 100
)

improvement = (
    (baseline_mae - ml_mae)
    / baseline_mae * 100
)

print("\n===== BASELINE VS ML MODEL =====")
print(f"Compared forecasts: {len(comparison_df):,}")
print(f"Baseline MAE: ${baseline_mae:,.2f}")
print(f"ML Model MAE: ${ml_mae:,.2f}")
print(f"Baseline WAPE: {baseline_wape:.2f}%")
print(f"ML Model WAPE: {ml_wape:.2f}%")
print(f"MAE Improvement: {improvement:.2f}%")


# Save model comparison results for the dashboard
comparison_metrics = pd.DataFrame([{
    "Forecasts_Compared": len(comparison_df),
    "Baseline_MAE": baseline_mae,
    "ML_MAE": ml_mae,
    "Baseline_WAPE": baseline_wape,
    "ML_WAPE": ml_wape,
    "MAE_Improvement_Percent": improvement
}])

comparison_metrics.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("Model comparison saved to reports/model_comparison.csv")

print("\nSUCCESS! Store performance report saved.")
print(f"Saved to: {OUTPUT}")
