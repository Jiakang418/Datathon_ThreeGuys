"""
Naive Visualization Module
==========================
Generates charts for Naive forecasting model evaluation

Charts created:
1. Actual vs Predicted comparison plots
2. Future forecast visualizations
3. Method comparison bar chart
4. All countries combined forecast

Author: Marcus
Date: 2025-12-19
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "naive_results"
PLOTS_DIR = BASE_DIR / "naive_plots"

# AstraZeneca color palette (professional look)
COLORS = {
    "primary": "#830051",      # AstraZeneca purple
    "secondary": "#00A0DF",    # Blue
    "accent": "#68D2DF",       # Light blue
    "positive": "#00843D",     # Green
    "negative": "#E40046",     # Red
    "neutral": "#6E6E6E",      # Gray
    "orange": "#FF6B35",       # Orange for variety
}

METHOD_COLORS = {
    "simple": "#830051",     # Purple
    "drift": "#00A0DF",      # Blue
    "seasonal": "#00843D",   # Green
    "mean": "#FF6B35",       # Orange
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
    
    # Predicted values (Naive)
    ax.plot(
        country_data["Week_Ending_Date"], 
        country_data["Predicted_Cash_Flow"],
        marker="s", markersize=8, linewidth=2, linestyle="--",
        color=COLORS["secondary"], label="Naive Forecast"
    )
    
    # Formatting
    ax.set_title(f"Naive Forecast Validation: {country}", fontsize=14, fontweight="bold")
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
    Plot future forecast for Naive model.
    """
    data = forecast_df[
        (forecast_df["Country"] == country) & 
        (forecast_df["Horizon"] == horizon)
    ].copy()
    data = data.sort_values("Week_Ending_Date")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Forecast line (flat for simple naive)
    ax.plot(
        data["Week_Ending_Date"],
        data["Predicted_Cash_Flow"],
        marker="o", markersize=6, linewidth=2,
        color=COLORS["primary"], label="Naive Forecast"
    )
    
    # Fill area to show the constant prediction
    ax.fill_between(
        data["Week_Ending_Date"],
        data["Predicted_Cash_Flow"] * 0.9,  # Simple uncertainty band
        data["Predicted_Cash_Flow"] * 1.1,
        alpha=0.2, color=COLORS["accent"], label="±10% Band"
    )
    
    # Formatting
    horizon_labels = {
        "1_week": "1 Week",
        "1_month": "1 Month",
        "6_month": "6 Months"
    }
    horizon_label = horizon_labels.get(horizon, horizon)
    
    ax.set_title(f"Naive {horizon_label} Forecast: {country}", fontsize=14, fontweight="bold")
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


def plot_method_comparison(metrics_df: pd.DataFrame, save_path: Path = None):
    """
    Bar chart comparing different Naive methods (simple, drift, seasonal, mean).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = metrics_df["Method"].unique()
    countries = metrics_df["Country"].unique()
    n_methods = len(methods)
    n_countries = len(countries)
    
    x = np.arange(n_countries)
    width = 0.2
    
    for i, method in enumerate(methods):
        method_data = metrics_df[metrics_df["Method"] == method]
        mape_values = []
        for country in countries:
            val = method_data[method_data["Country"] == country]["MAPE_percent"].values
            mape_values.append(val[0] if len(val) > 0 else 0)
        
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, mape_values, width, 
                     label=method.capitalize(),
                     color=METHOD_COLORS.get(method, COLORS["neutral"]))
    
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.set_title("Naive Methods Comparison by Country (MAPE)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Saved plot: {save_path.name}")
    
    plt.close()
    return fig


def plot_metrics_by_country(metrics_df: pd.DataFrame, method: str = "simple", save_path: Path = None):
    """
    Bar chart showing MAPE for each country using specified method.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = metrics_df[metrics_df["Method"] == method].copy()
    countries = data["Country"].values
    mape_values = data["MAPE_percent"].values
    
    # Color bars based on performance
    colors = []
    for mape in mape_values:
        if mape is None or pd.isna(mape):
            colors.append(COLORS["neutral"])
        elif mape < 20:
            colors.append(COLORS["positive"])  # Good
        elif mape < 50:
            colors.append(COLORS["secondary"])  # OK
        else:
            colors.append(COLORS["negative"])  # Needs improvement
    
    bars = ax.bar(countries, mape_values, color=colors, edgecolor="black", linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, mape_values):
        if val is not None and not pd.isna(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10
            )
    
    # Reference lines
    ax.axhline(y=20, color=COLORS["positive"], linestyle="--", linewidth=1.5, label="Good: 20% MAPE")
    ax.axhline(y=50, color=COLORS["secondary"], linestyle="--", linewidth=1.5, label="OK: 50% MAPE")
    
    ax.set_title(f"Naive ({method.capitalize()}) Model Accuracy by Country", fontsize=14, fontweight="bold")
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Saved plot: {save_path.name}")
    
    plt.close()
    return fig


def plot_all_countries_forecast(forecast_df: pd.DataFrame, horizon: str, save_path: Path = None):
    """
    Combined plot showing all countries' Naive forecasts.
    """
    data = forecast_df[forecast_df["Horizon"] == horizon].copy()
    countries = data["Country"].unique()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    color_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], 
                   COLORS["positive"], COLORS["negative"], COLORS["orange"], "#800080", "#008080"]
    
    for i, country in enumerate(countries):
        country_data = data[data["Country"] == country].sort_values("Week_Ending_Date")
        ax.plot(
            country_data["Week_Ending_Date"],
            country_data["Predicted_Cash_Flow"],
            marker="o", markersize=4, linewidth=1.5,
            color=color_cycle[i % len(color_cycle)],
            label=country
        )
    
    horizon_labels = {
        "1_week": "1 Week",
        "1_month": "1 Month",
        "6_month": "6 Months"
    }
    horizon_label = horizon_labels.get(horizon, horizon)
    
    ax.set_title(f"Naive {horizon_label} Forecast - All Countries", fontsize=14, fontweight="bold")
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


def plot_avg_method_comparison(metrics_df: pd.DataFrame, save_path: Path = None):
    """
    Bar chart comparing average MAPE across all countries for each method.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_mape = metrics_df.groupby("Method")["MAPE_percent"].mean().sort_values()
    
    methods = avg_mape.index.tolist()
    values = avg_mape.values
    
    colors = [METHOD_COLORS.get(m, COLORS["neutral"]) for m in methods]
    
    bars = ax.barh(methods, values, color=colors, edgecolor="black", linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            ha="left", va="center", fontsize=11
        )
    
    ax.set_title("Average MAPE by Naive Method (All Countries)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average MAPE (%)", fontsize=12)
    ax.set_ylabel("Method", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    
    # Capitalize method names for display
    ax.set_yticklabels([m.capitalize() for m in methods])
    
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
    """Generate all Naive visualization charts."""
    
    print("=" * 70)
    print("  GENERATING NAIVE MODEL VISUALIZATIONS")
    print("=" * 70)
    
    # Create output directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("\n[Step 1] Loading Naive results...")
    
    comparison_file = RESULTS_DIR / "naive_actual_vs_predicted.csv"
    forecast_file = RESULTS_DIR / "naive_future_forecasts.csv"
    metrics_file = RESULTS_DIR / "naive_backtest_metrics.csv"
    
    if not all(f.exists() for f in [comparison_file, forecast_file, metrics_file]):
        print("  [X] Results not found. Run naive_forecasting.py first!")
        return
    
    comparison_df = pd.read_csv(comparison_file, parse_dates=["Week_Ending_Date"])
    forecast_df = pd.read_csv(forecast_file, parse_dates=["Week_Ending_Date"])
    metrics_df = pd.read_csv(metrics_file)
    
    print(f"  [OK] Loaded {len(comparison_df)} validation records")
    print(f"  [OK] Loaded {len(forecast_df)} forecast records")
    print(f"  [OK] Loaded {len(metrics_df)} metric records")
    
    # Generate plots
    print("\n[Step 2] Generating actual vs predicted plots...")
    countries = comparison_df["Country"].unique()
    for country in countries:
        save_path = PLOTS_DIR / f"validation_{country}.png"
        plot_actual_vs_predicted(comparison_df, country, save_path)
    
    print("\n[Step 3] Generating future forecast plots...")
    for country in forecast_df["Country"].unique():
        # 1-week forecast
        save_path = PLOTS_DIR / f"forecast_1week_{country}.png"
        plot_future_forecast(forecast_df, country, "1_week", save_path)
        
        # 1-month forecast
        save_path = PLOTS_DIR / f"forecast_1month_{country}.png"
        plot_future_forecast(forecast_df, country, "1_month", save_path)
        
        # 6-month forecast
        save_path = PLOTS_DIR / f"forecast_6month_{country}.png"
        plot_future_forecast(forecast_df, country, "6_month", save_path)
    
    print("\n[Step 4] Generating summary plots...")
    
    # Method comparison (by country)
    save_path = PLOTS_DIR / "method_comparison_by_country.png"
    plot_method_comparison(metrics_df, save_path)
    
    # Average method comparison
    save_path = PLOTS_DIR / "avg_method_comparison.png"
    plot_avg_method_comparison(metrics_df, save_path)
    
    # Metrics by country (simple naive)
    save_path = PLOTS_DIR / "metrics_by_country_simple.png"
    plot_metrics_by_country(metrics_df, method="simple", save_path=save_path)
    
    # All countries combined forecast
    for horizon in ["1_week", "1_month", "6_month"]:
        save_path = PLOTS_DIR / f"all_countries_forecast_{horizon}.png"
        plot_all_countries_forecast(forecast_df, horizon, save_path)
    
    print("\n" + "=" * 70)
    print("  VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    generate_all_visualizations()

