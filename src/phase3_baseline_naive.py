"""
Phase 3: Baseline Naive Forecasting Model

This script implements a simple Naive forecasting baseline:
- Forecast for all 4 test weeks = Last observed value from training set
- This provides a baseline to compare against more complex hybrid models
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def main():
    """Main execution function."""
    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "data" / "model_dataset" / "weekly_features.csv"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, parse_dates=["Week_Ending_Date"])
    df = df.sort_values(["Country_Name", "Week_Ending_Date"])

    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Countries: {df['Country_Name'].unique()}")
    print(f"Date range: {df['Week_Ending_Date'].min()} to {df['Week_Ending_Date'].max()}")

    # -------------------------------------------------------------------------
    # 2. Process Each Country
    # -------------------------------------------------------------------------
    test_size = 4
    results = {}
    plot_data = {}

    for country in df["Country_Name"].unique():
        print(f"\n{'='*60}")
        print(f"Processing Country: {country}")
        print(f"{'='*60}")

        # Filter data for this country
        country_data = df[df["Country_Name"] == country].copy()
        country_data = country_data.sort_values("Week_Ending_Date")

        # Check if we have enough data
        if len(country_data) < test_size + 1:
            print(f"  Skipping {country}: insufficient data ({len(country_data)} rows)")
            continue

        # Split into Train and Test
        split_idx = len(country_data) - test_size
        train_data = country_data.iloc[:split_idx].copy()
        test_data = country_data.iloc[split_idx:].copy()

        print(f"  Train size: {len(train_data)} weeks")
        print(f"  Test size: {len(test_data)} weeks")

        # ---------------------------------------------------------------------
        # 3. Generate Naive Forecast
        # ---------------------------------------------------------------------
        # Get the last observed value from training set
        last_observed_value = train_data["Net_Cash_Flow"].iloc[-1]
        print(f"  Last observed value (Week 0): ${last_observed_value:,.2f}")

        # Create forecast: all 4 test weeks = last observed value
        naive_forecast = np.full(test_size, last_observed_value)

        # Get actual test values
        actuals = test_data["Net_Cash_Flow"].values

        # ---------------------------------------------------------------------
        # 4. Evaluation
        # ---------------------------------------------------------------------
        rmse = np.sqrt(mean_squared_error(actuals, naive_forecast))
        results[country] = rmse

        print(f"  Naive Forecast (all 4 weeks): ${last_observed_value:,.2f}")
        print(f"  Actual Test Values: {[f'${v:,.2f}' for v in actuals]}")
        print(f"  RMSE: ${rmse:,.2f}")

        # Store data for plotting
        plot_data[country] = {
            "train_data": train_data,
            "test_data": test_data,
            "forecast": naive_forecast,
            "rmse": rmse,
        }

    # -------------------------------------------------------------------------
    # 5. Summary Statistics
    # -------------------------------------------------------------------------
    if not results:
        print("\nNo results to report. Check data availability.")
        return

    print("\n" + "=" * 60)
    print("BASELINE NAIVE FORECAST - RMSE RESULTS")
    print("=" * 60)
    for country, rmse in results.items():
        print(f"{country}: ${rmse:,.2f}")

    avg_rmse = np.mean(list(results.values()))
    print(f"\nAverage RMSE across all countries: ${avg_rmse:,.2f}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 6. Visualization
    # -------------------------------------------------------------------------
    n_countries = len(plot_data)
    fig, axes = plt.subplots(n_countries, 1, figsize=(14, 5 * n_countries))
    if n_countries == 1:
        axes = [axes]

    for idx, (country, data) in enumerate(plot_data.items()):
        ax = axes[idx]

        train_data = data["train_data"]
        test_data = data["test_data"]
        forecast = data["forecast"]
        rmse = data["rmse"]

        # Plot historical data (last 3 months = ~12 weeks, or all if less)
        historical_weeks = min(12, len(train_data))
        train_plot = train_data.iloc[-historical_weeks:]

        ax.plot(
            train_plot["Week_Ending_Date"],
            train_plot["Net_Cash_Flow"],
            "b-",
            label="Historical (Training)",
            linewidth=2,
            marker="o",
            markersize=4,
        )

        # Plot actual test values
        ax.plot(
            test_data["Week_Ending_Date"],
            test_data["Net_Cash_Flow"],
            "ko-",
            label="Actual (Test)",
            linewidth=2,
            marker="s",
            markersize=8,
        )

        # Plot naive forecast (flat line)
        ax.plot(
            test_data["Week_Ending_Date"],
            forecast,
            "r--",
            label=f"Naive Forecast (RMSE=${rmse:,.2f})",
            linewidth=2,
            marker="^",
            markersize=8,
        )

        # Add vertical line separating train and test
        split_date = test_data["Week_Ending_Date"].iloc[0]
        ax.axvline(x=split_date, color="gray", linestyle=":", linewidth=1, alpha=0.7)

        ax.set_title(
            f"Baseline Naive Forecast - {country} (RMSE: ${rmse:,.2f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Week Ending Date", fontsize=12)
        ax.set_ylabel("Net Cash Flow (USD)", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plot_path = output_dir / "baseline_naive_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()

