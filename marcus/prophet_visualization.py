"""
Prophet Visualization Module
============================
Generates charts for Phase 4 (Evaluation) and Phase 5 (Dashboard)

Charts created:
1. Actual vs Predicted comparison plots
2. Future forecast with confidence intervals
3. Model comparison bar chart (to compare with Naive/SMA/EMA)

Author: Marcus
Date: 2025-12-19
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "prophet_results"
PLOTS_DIR = BASE_DIR / "prophet_plots"

# AstraZeneca color palette (professional look)
COLORS = {
    "primary": "#830051",      # AstraZeneca purple
    "secondary": "#00A0DF",    # Blue
    "accent": "#68D2DF",       # Light blue
    "positive": "#00843D",     # Green
    "negative": "#E40046",     # Red
    "neutral": "#6E6E6E",      # Gray
}


# --------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# --------------------------------------------------------------------------------------

def plot_actual_vs_predicted(comparison_df: pd.DataFrame, country: str, save_path: Path = None):
    """
    Plot actual vs predicted values for validation period.
    """
    country_data = comparison_df[comparison_df["Country"] == country].copy()
    country_data = country_data.sort_values("Week_Ending_Date")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Actual values
    ax.plot(
        country_data["Week_Ending_Date"], 
        country_data["Actual_Cash_Flow"],
        marker="o", markersize=10, linewidth=2,
        color=COLORS["primary"], label="Actual"
    )
    
    # Predicted values
    ax.plot(
        country_data["Week_Ending_Date"], 
        country_data["Predicted_Cash_Flow"],
        marker="s", markersize=8, linewidth=2, linestyle="--",
        color=COLORS["secondary"], label="Prophet Forecast"
    )
    
    # Confidence interval
    ax.fill_between(
        country_data["Week_Ending_Date"],
        country_data["Prediction_Lower_95CI"],
        country_data["Prediction_Upper_95CI"],
        alpha=0.2, color=COLORS["secondary"], label="95% CI"
    )
    
    # Formatting
    ax.set_title(f"Prophet Forecast Validation: {country}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Week Ending Date", fontsize=12)
    ax.set_ylabel("Net Cash Flow (USD)", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color=COLORS["neutral"], linestyle="-", linewidth=0.8)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Saved plot: {save_path.name}")
    
    plt.close()
    return fig


def plot_future_forecast(forecast_df: pd.DataFrame, country: str, horizon: str, save_path: Path = None):
    """
    Plot future forecast with confidence intervals.
    """
    data = forecast_df[
        (forecast_df["Country"] == country) & 
        (forecast_df["Horizon"] == horizon)
    ].copy()
    data = data.sort_values("Week_Ending_Date")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Forecast line
    ax.plot(
        data["Week_Ending_Date"],
        data["Predicted_Cash_Flow"],
        marker="o", markersize=6, linewidth=2,
        color=COLORS["primary"], label="Forecast"
    )
    
    # Confidence interval
    ax.fill_between(
        data["Week_Ending_Date"],
        data["Prediction_Lower_95CI"],
        data["Prediction_Upper_95CI"],
        alpha=0.3, color=COLORS["accent"], label="95% Confidence Interval"
    )
    
    # Formatting
    horizon_label = "1 Month" if horizon == "1_month" else "6 Months"
    ax.set_title(f"Prophet {horizon_label} Forecast: {country}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Week Ending Date", fontsize=12)
    ax.set_ylabel("Predicted Net Cash Flow (USD)", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color=COLORS["neutral"], linestyle="-", linewidth=0.8)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Saved plot: {save_path.name}")
    
    plt.close()
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame, save_path: Path = None):
    """
    Bar chart comparing MAPE across all countries.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    countries = metrics_df["Country"].values
    mape_values = metrics_df["MAPE_percent"].values
    
    # Color bars based on performance
    colors = []
    for mape in mape_values:
        if mape is None or pd.isna(mape):
            colors.append(COLORS["neutral"])
        elif mape < 15:
            colors.append(COLORS["positive"])  # Good
        elif mape < 30:
            colors.append(COLORS["secondary"])  # OK
        else:
            colors.append(COLORS["negative"])  # Needs improvement
    
    bars = ax.bar(countries, mape_values, color=colors, edgecolor="black", linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, mape_values):
        if val is not None and not pd.isna(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10
            )
    
    # Reference line for "good" performance
    ax.axhline(y=15, color=COLORS["positive"], linestyle="--", linewidth=1.5, label="Target: 15% MAPE")
    
    ax.set_title("Prophet Model Accuracy by Country (MAPE)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Saved plot: {save_path.name}")
    
    plt.close()
    return fig


def plot_all_countries_forecast(forecast_df: pd.DataFrame, horizon: str, save_path: Path = None):
    """
    Combined plot showing all countries' forecasts.
    """
    data = forecast_df[forecast_df["Horizon"] == horizon].copy()
    countries = data["Country"].unique()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    color_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], 
                   COLORS["positive"], COLORS["negative"], "#FFA500", "#800080", "#008080"]
    
    for i, country in enumerate(countries):
        country_data = data[data["Country"] == country].sort_values("Week_Ending_Date")
        ax.plot(
            country_data["Week_Ending_Date"],
            country_data["Predicted_Cash_Flow"],
            marker="o", markersize=4, linewidth=1.5,
            color=color_cycle[i % len(color_cycle)],
            label=country
        )
    
    horizon_label = "1 Month" if horizon == "1_month" else "6 Months"
    ax.set_title(f"Prophet {horizon_label} Forecast - All Countries", fontsize=14, fontweight="bold")
    ax.set_xlabel("Week Ending Date", fontsize=12)
    ax.set_ylabel("Predicted Net Cash Flow (USD)", fontsize=12)
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color=COLORS["neutral"], linestyle="-", linewidth=0.8)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Saved plot: {save_path.name}")
    
    plt.close()
    return fig


# --------------------------------------------------------------------------------------
# MAIN VISUALIZATION PIPELINE
# --------------------------------------------------------------------------------------

def generate_all_visualizations():
    """Generate all Prophet visualization charts."""
    
    print("=" * 70)
    print("  GENERATING PROPHET VISUALIZATIONS")
    print("=" * 70)
    
    # Create output directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("\n[Step 1] Loading Prophet results...")
    
    comparison_file = RESULTS_DIR / "prophet_actual_vs_predicted.csv"
    forecast_file = RESULTS_DIR / "prophet_future_forecasts.csv"
    metrics_file = RESULTS_DIR / "prophet_backtest_metrics.csv"
    
    if not all(f.exists() for f in [comparison_file, forecast_file, metrics_file]):
        print("  [X] Results not found. Run prophet_forecasting.py first!")
        return
    
    comparison_df = pd.read_csv(comparison_file, parse_dates=["Week_Ending_Date"])
    forecast_df = pd.read_csv(forecast_file, parse_dates=["Week_Ending_Date"])
    metrics_df = pd.read_csv(metrics_file)
    
    print(f"  [OK] Loaded {len(comparison_df)} validation records")
    print(f"  [OK] Loaded {len(forecast_df)} forecast records")
    
    # Generate plots
    print("\n[Step 2] Generating actual vs predicted plots...")
    countries = comparison_df["Country"].unique()
    for country in countries:
        save_path = PLOTS_DIR / f"validation_{country}.png"
        plot_actual_vs_predicted(comparison_df, country, save_path)
    
    print("\n[Step 3] Generating future forecast plots...")
    for country in forecast_df["Country"].unique():
        # 1-month forecast
        save_path = PLOTS_DIR / f"forecast_1month_{country}.png"
        plot_future_forecast(forecast_df, country, "1_month", save_path)
        
        # 6-month forecast
        save_path = PLOTS_DIR / f"forecast_6month_{country}.png"
        plot_future_forecast(forecast_df, country, "6_month", save_path)
    
    print("\n[Step 4] Generating summary plots...")
    
    # Metrics comparison bar chart
    save_path = PLOTS_DIR / "metrics_comparison.png"
    plot_metrics_comparison(metrics_df, save_path)
    
    # All countries combined forecast
    save_path = PLOTS_DIR / "all_countries_forecast_1month.png"
    plot_all_countries_forecast(forecast_df, "1_month", save_path)
    
    save_path = PLOTS_DIR / "all_countries_forecast_6month.png"
    plot_all_countries_forecast(forecast_df, "6_month", save_path)
    
    print("\n" + "=" * 70)
    print("  VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    generate_all_visualizations()

