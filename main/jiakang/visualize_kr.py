"""
Visualize KR Weekly Cash Flows and ARIMA Forecast
=================================================
This script plots historical net cash flow and ARIMA forecasts for KR to inspect anomalies.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------
# Configuration
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "arima_results"
PLOTS_DIR = BASE_DIR / "arima_plots"
PLOTS_DIR.mkdir(exist_ok=True)

# File paths
comparison_file = RESULTS_DIR / "arima_actual_vs_predicted.csv"
forecast_file = RESULTS_DIR / "arima_future_forecasts.csv"

# ------------------------------
# Load ARIMA results
# ------------------------------
comparison_df = pd.read_csv(comparison_file, parse_dates=["Week_Ending_Date"])
forecast_df = pd.read_csv(forecast_file, parse_dates=["Week_Ending_Date"])

# Filter for KR
comparison_kr = comparison_df[comparison_df["Country"] == "KR"].sort_values("Week_Ending_Date")
forecast_kr = forecast_df[forecast_df["Country"] == "KR"].sort_values("Week_Ending_Date")

# ------------------------------
# Plot
# ------------------------------
plt.figure(figsize=(14, 6))

# Historical actual
plt.plot(
    comparison_kr["Week_Ending_Date"],
    comparison_kr["Actual_Cash_Flow"],
    marker="o",
    color="#2E86AB",
    label="Historical (Actual)"
)

# Validation predictions
plt.plot(
    comparison_kr["Week_Ending_Date"],
    comparison_kr["Predicted_Cash_Flow"],
    marker="x",
    linestyle="--",
    color="#A23B72",
    label="Validation (Predicted)"
)

# Future forecast
if not forecast_kr.empty:
    plt.plot(
        forecast_kr["Week_Ending_Date"],
        forecast_kr["Predicted_Cash_Flow"],
        marker="s",
        linestyle="-",
        color="#F18F01",
        label="Forecast"
    )

    if "Lower_CI" in forecast_kr.columns and "Upper_CI" in forecast_kr.columns:
        plt.fill_between(
            forecast_kr["Week_Ending_Date"],
            forecast_kr["Lower_CI"],
            forecast_kr["Upper_CI"],
            color="#F18F01",
            alpha=0.2,
            label="95% CI"
        )

plt.title("KR Weekly Net Cash Flow and ARIMA Forecast", fontsize=14, fontweight="bold")
plt.xlabel("Week Ending Date", fontsize=12)
plt.ylabel("Net Cash Flow", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Save plot
plot_path = PLOTS_DIR / "KR_weekly_cashflow_forecast.png"
plt.savefig(plot_path, dpi=150)
plt.show()

print(f"[OK] Plot saved to {plot_path}")
