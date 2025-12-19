"""
ARIMA Model Selection and Performance Summary
=============================================

This script:
- Loads weekly cash flow data
- Runs multiple ARIMA (p,d,q) combinations per country
- Backtests each model on 4-week validation
- Selects the best ARIMA order based on MAPE
- Outputs per-country metrics and overall averages
"""

import itertools
import warnings
import pandas as pd
import numpy as np
from arima_forecasting import load_weekly_data, backtest_arima

warnings.filterwarnings("ignore")

# -------------------------------
# Candidate ARIMA Orders
# -------------------------------
p = [0, 1, 2]
d = [0, 1]
q = [0, 1, 2]
orders = list(itertools.product(p, d, q))

# -------------------------------
# Load Data
# -------------------------------
df = load_weekly_data()
countries = df["Country_Name"].unique()

# -------------------------------
# Run ARIMA Grid Search
# -------------------------------
results_summary = []

for country in countries:
    print(f"\nProcessing {country}...")
    country_df = df[df["Country_Name"] == country].sort_values("Week_Ending_Date").reset_index(drop=True)
    
    best_mape = float("inf")
    best_order = None
    best_result = None
    
    for order in orders:
        # Temporarily override ARIMA_ORDER in backtest
        from arima_forecasting import ARIMA_ORDER
        ARIMA_ORDER = order
        
        res = backtest_arima(country_df, country, target_col="Net_Cash_Flow", validation_weeks=4)
        if res is None:
            continue
        mape = res["metrics"]["MAPE_percent"]
        if mape is not None and mape < best_mape:
            best_mape = mape
            best_order = order
            best_result = res
    
    if best_order is not None:
        metrics = best_result["metrics"]
        results_summary.append({
            "Country": country,
            "Best_ARIMA_Order": best_order,
            "RMSE_USD": round(metrics["RMSE"], 2),
            "MAE_USD": round(metrics["MAE"], 2),
            "MAPE_percent": round(metrics["MAPE_percent"], 2),
            "Train_Weeks": len(best_result["train_df"]),
            "Validation_Weeks": len(best_result["val_df"])
        })
        print(f"Best order for {country}: {best_order}, MAPE: {round(best_mape,2)}%")
    else:
        print(f"No valid ARIMA model found for {country}")

# -------------------------------
# Summary Table
# -------------------------------
summary_df = pd.DataFrame(results_summary)
summary_df = summary_df.sort_values("Country").reset_index(drop=True)
print("\n=== Overall Model Performance ===")
print(summary_df.to_string(index=False))

# -------------------------------
# Average Metrics
# -------------------------------
avg_rmse = round(summary_df["RMSE_USD"].mean(), 2)
avg_mape = round(summary_df["MAPE_percent"].mean(), 2)

print(f"\nAverage RMSE across all countries: ${avg_rmse}")
print(f"Average MAPE across all countries: {avg_mape}%")

# -------------------------------
# Save to CSV
# -------------------------------
summary_df.to_csv("arima_best_model_summary.csv", index=False)
print("\nSaved summary to 'arima_best_model_summary.csv'")
