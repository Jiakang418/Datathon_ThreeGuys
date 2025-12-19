"""
ARIMA Visualization Module
===========================
Generates visualizations for ARIMA forecasting results.

This script creates:
- Validation comparison plots (Actual vs Predicted)
- Forecast plots with confidence intervals
- Metrics comparison charts
- Per-country detailed plots
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "arima_results"
PLOTS_DIR = BASE_DIR / "arima_plots"
PLOTS_DIR.mkdir(exist_ok=True)

MAPE_THRESHOLD = 15.0  # Threshold for acceptable MAPE

# --------------------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------------------
def load_results_data():
    """Load ARIMA results CSVs."""
    comparison_file = RESULTS_DIR / "arima_actual_vs_predicted.csv"
    forecast_file = RESULTS_DIR / "arima_future_forecasts.csv"
    metrics_file = RESULTS_DIR / "arima_backtest_metrics.csv"

    for f in [comparison_file, forecast_file, metrics_file]:
        if not f.exists():
            raise FileNotFoundError(f"{f} not found. Run 'run_arima.py' first.")

    comparison_df = pd.read_csv(comparison_file, parse_dates=["Week_Ending_Date"])
    forecast_df = pd.read_csv(forecast_file, parse_dates=["Week_Ending_Date"])
    metrics_df = pd.read_csv(metrics_file)

    return comparison_df, forecast_df, metrics_df

# --------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# --------------------------------------------------------------------------------------
def plot_validation_comparison(comparison_df):
    """Plot Actual vs Predicted for each country."""
    for country in comparison_df["Country"].unique():
        df = comparison_df[comparison_df["Country"] == country].sort_values("Week_Ending_Date")
        plt.figure(figsize=(12, 6))
        plt.plot(df["Week_Ending_Date"], df["Actual_Cash_Flow"], label="Actual", marker="o", color="#2E86AB")
        plt.plot(df["Week_Ending_Date"], df["Predicted_Cash_Flow"], label="Predicted", marker="x", linestyle="--", color="#A23B72")
        plt.title(f"ARIMA Validation: {country}", fontsize=14)
        plt.xlabel("Week Ending Date")
        plt.ylabel("Net Cash Flow")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"validation_{country.replace(' ', '_')}.png", dpi=150)
        plt.close()

def plot_forecast_with_history(comparison_df, forecast_df):
    """Plot historical + validation + future forecasts with CI."""
    for country in forecast_df["Country"].unique():
        historical = comparison_df[comparison_df["Country"] == country].sort_values("Week_Ending_Date")
        forecast_1m = forecast_df[(forecast_df["Country"] == country) & (forecast_df["Horizon"] == "1_month")].sort_values("Week_Ending_Date")
        if forecast_1m.empty:
            continue

        plt.figure(figsize=(14, 7))
        plt.plot(historical["Week_Ending_Date"], historical["Actual_Cash_Flow"], label="Historical", marker="o", color="#2E86AB")
        plt.plot(historical["Week_Ending_Date"], historical["Predicted_Cash_Flow"], label="Validation", marker="x", linestyle="--", color="#A23B72")
        plt.plot(forecast_1m["Week_Ending_Date"], forecast_1m["Predicted_Cash_Flow"], label="1-Month Forecast", marker="s", color="#F18F01")
        if "Lower_CI" in forecast_1m.columns and "Upper_CI" in forecast_1m.columns:
            plt.fill_between(forecast_1m["Week_Ending_Date"], forecast_1m["Lower_CI"], forecast_1m["Upper_CI"], color="#F18F01", alpha=0.2, label="95% CI")
        plt.title(f"Cash Flow Forecast: {country}", fontsize=14)
        plt.xlabel("Week Ending Date")
        plt.ylabel("Net Cash Flow")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"forecast_{country.replace(' ', '_')}.png", dpi=150)
        plt.close()

def plot_metrics_comparison(metrics_df):
    """Plot metrics comparison charts (RMSE, MAE, MAPE)."""
    # MAPE bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_df["Country"], metrics_df["MAPE_percent"], alpha=0.7)
    plt.axhline(MAPE_THRESHOLD, linestyle="--", color="red", label=f"Target ({MAPE_THRESHOLD}%)")
    for bar, mape in zip(bars, metrics_df["MAPE_percent"]):
        if mape < MAPE_THRESHOLD:
            bar.set_color("#06A77D")
        else:
            bar.set_color("#D72638")
    plt.title("ARIMA Model Accuracy (MAPE)")
    plt.ylabel("MAPE (%)")
    plt.xlabel("Country")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metrics_mape_comparison.png", dpi=150)
    plt.close()

# --------------------------------------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------------------------------------
def generate_all_visualizations():
    comparison_df, forecast_df, metrics_df = load_results_data()
    plot_validation_comparison(comparison_df)
    plot_forecast_with_history(comparison_df, forecast_df)
    plot_metrics_comparison(metrics_df)
