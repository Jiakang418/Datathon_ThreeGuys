"""
Phase 5: Final Dashboard & Evaluation

This script generates a consolidated executive dashboard showing:
1. Cash Flow Trend (Forecast vs Actual) with anomaly overlays
2. Anomaly Severity by Country
3. Top 5 Transaction Anomalies (Audit Hit-List)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def load_dashboard_data(base_dir: Path):
    """Load all required data files for the dashboard."""
    outputs_dir = base_dir / "outputs"
    
    # Load validation data (actual vs predicted)
    validation_path = outputs_dir / "hybrid_arima_actual_vs_predicted.csv"
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation file not found: {validation_path}")
    validation_df = pd.read_csv(validation_path, parse_dates=["Week_Ending_Date"])
    
    # Load future forecasts
    forecast_path = outputs_dir / "hybrid_arima_future_forecasts.csv"
    if not forecast_path.exists():
        raise FileNotFoundError(f"Forecast file not found: {forecast_path}")
    forecast_df = pd.read_csv(forecast_path, parse_dates=["Week_Ending_Date"])
    
    # Load structural anomalies
    structural_path = outputs_dir / "anomalies_structural_level.csv"
    if not structural_path.exists():
        raise FileNotFoundError(f"Structural anomalies file not found: {structural_path}")
    structural_df = pd.read_csv(structural_path, parse_dates=["Week_Ending_Date"])
    
    # Load transaction anomalies
    transaction_path = outputs_dir / "anomalies_transaction_level.csv"
    if not transaction_path.exists():
        raise FileNotFoundError(f"Transaction anomalies file not found: {transaction_path}")
    transaction_df = pd.read_csv(transaction_path, parse_dates=["Date"])
    
    return validation_df, forecast_df, structural_df, transaction_df


def create_dashboard(validation_df, forecast_df, structural_df, transaction_df, country="ID"):
    """
    Create the consolidated executive dashboard.
    
    Args:
        validation_df: Historical validation data
        forecast_df: Future forecast data
        structural_df: Structural anomaly data
        transaction_df: Transaction anomaly data
        country: Country to focus on for Panel A (default: 'ID')
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle(
        "Cash Flow Forecasting & Anomaly Detection System",
        fontsize=20,
        fontweight="bold",
        y=0.98
    )
    
    # -------------------------------------------------------------------------
    # Panel A: Cash Flow Trend (Forecast vs Actual) - Top, spans 2 columns
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, :])
    
    # Filter data for selected country
    val_country = validation_df[validation_df["Country"] == country].copy()
    forecast_country = forecast_df[forecast_df["Country"] == country].copy()
    structural_country = structural_df[
        (structural_df["Country"] == country) & (structural_df["Anomaly_Flag"] == -1)
    ].copy()
    
    # Sort by date
    val_country = val_country.sort_values("Week_Ending_Date")
    forecast_country = forecast_country.sort_values("Week_Ending_Date")
    
    # Plot historical actuals
    ax_a.plot(
        val_country["Week_Ending_Date"],
        val_country["Actual_Cash_Flow"],
        "o-",
        color="#2E86AB",
        label="Historical Actuals",
        linewidth=2,
        markersize=6,
        alpha=0.8
    )
    
    # Plot historical predictions
    ax_a.plot(
        val_country["Week_Ending_Date"],
        val_country["Predicted_Cash_Flow"],
        "--",
        color="#A23B72",
        label="Model Predictions (Validation)",
        linewidth=2,
        alpha=0.7
    )
    
    # Plot confidence intervals for validation
    ax_a.fill_between(
        val_country["Week_Ending_Date"],
        val_country["Prediction_Lower_95CI"],
        val_country["Prediction_Upper_95CI"],
        color="#A23B72",
        alpha=0.2,
        label="95% Confidence Interval"
    )
    
    # Plot future forecasts (1-month and 6-month)
    forecast_1m = forecast_country[forecast_country["Horizon"] == "1_month"]
    forecast_6m = forecast_country[forecast_country["Horizon"] == "6_month"]
    
    if len(forecast_1m) > 0:
        ax_a.plot(
            forecast_1m["Week_Ending_Date"],
            forecast_1m["Predicted_Cash_Flow"],
            "s-",
            color="#F18F01",
            label="Future Forecast (1-month)",
            linewidth=2,
            markersize=5,
            alpha=0.8
        )
    
    if len(forecast_6m) > 0:
        ax_a.plot(
            forecast_6m["Week_Ending_Date"],
            forecast_6m["Predicted_Cash_Flow"],
            "^--",
            color="#C73E1D",
            label="Future Forecast (6-month)",
            linewidth=2,
            markersize=4,
            alpha=0.7
        )
    
    # Overlay structural anomalies as red dots
    if len(structural_country) > 0:
        anomaly_dates = structural_country["Week_Ending_Date"]
        anomaly_values = structural_country["Net_Cash_Flow"]
        
        ax_a.scatter(
            anomaly_dates,
            anomaly_values,
            color="red",
            s=150,
            marker="X",
            label="Volatility Alert (Structural Anomaly)",
            zorder=10,
            edgecolors="darkred",
            linewidths=1.5
        )
    
    # Add vertical line separating historical and future
    if len(val_country) > 0 and len(forecast_country) > 0:
        last_historical_date = val_country["Week_Ending_Date"].max()
        ax_a.axvline(
            x=last_historical_date,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Forecast Start"
        )
    
    ax_a.set_title(f"Cash Flow Trend - {country} (Indonesia)", fontsize=14, fontweight="bold")
    ax_a.set_xlabel("Week Ending Date", fontsize=11)
    ax_a.set_ylabel("Net Cash Flow (USD)", fontsize=11)
    ax_a.legend(loc="best", fontsize=9, framealpha=0.9)
    ax_a.grid(True, alpha=0.3)
    ax_a.tick_params(axis="x", rotation=45)
    
    # Format y-axis as currency
    ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # -------------------------------------------------------------------------
    # Panel B: Anomaly Severity by Country (Bar Chart)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1, 0])
    
    # Count structural anomalies per country
    anomaly_counts = structural_df[structural_df["Anomaly_Flag"] == -1].groupby("Country").size()
    anomaly_counts = anomaly_counts.sort_values(ascending=False)
    
    # Color mapping: Red for high, Yellow for low
    max_count = anomaly_counts.max() if len(anomaly_counts) > 0 else 1
    colors = [
        plt.cm.Reds(0.3 + 0.7 * (count / max_count)) if max_count > 0 else "yellow"
        for count in anomaly_counts.values
    ]
    
    bars = ax_b.bar(
        anomaly_counts.index,
        anomaly_counts.values,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.8
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax_b.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )
    
    ax_b.set_title("Anomaly Severity by Country", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Country", fontsize=10)
    ax_b.set_ylabel("Number of Structural Anomalies", fontsize=10)
    ax_b.grid(True, alpha=0.3, axis="y")
    ax_b.tick_params(axis="x", rotation=45)
    
    # -------------------------------------------------------------------------
    # Panel C: Top 5 Transaction Anomalies (Table)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.axis("off")
    
    # Get top 5 transaction anomalies by absolute Z-score
    transaction_df_copy = transaction_df.copy()
    transaction_df_copy["Abs_Z_Score"] = transaction_df_copy["Z_Score"].abs()
    top_5 = transaction_df_copy.nlargest(5, "Abs_Z_Score", keep="all")
    top_5 = top_5.sort_values("Abs_Z_Score", ascending=False)
    
    # Prepare table data
    table_data = []
    for idx, row in top_5.iterrows():
        table_data.append([
            row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "N/A",
            row["Entity"],
            row["Category"],
            f"${row['Amount_USD']:,.2f}",
            f"{row['Z_Score']:.2f}"
        ])
    
    # Create table
    table = ax_c.table(
        cellText=table_data,
        colLabels=["Date", "Entity", "Category", "Amount", "Z-Score"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header row
    for i in range(5):
        table[(0, i)].set_facecolor("#2E86AB")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # Color code rows by Z-score magnitude
    for i in range(1, len(table_data) + 1):
        z_score = abs(float(table_data[i - 1][4]))
        if z_score > 15:
            color = "#FF6B6B"  # Red
        elif z_score > 10:
            color = "#FFA07A"  # Light salmon
        else:
            color = "#FFE4B5"  # Moccasin
        
        for j in range(5):
            table[(i, j)].set_facecolor(color)
    
    ax_c.set_title("Audit Hit-List: Top 5 Transaction Anomalies", fontsize=12, fontweight="bold", pad=20)
    
    # -------------------------------------------------------------------------
    # Panel D: Summary Statistics (Optional - can add metrics here)
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[2, :])
    ax_d.axis("off")
    
    # Calculate summary statistics
    total_anomalies_structural = (structural_df["Anomaly_Flag"] == -1).sum()
    total_anomalies_transaction = len(transaction_df)
    total_countries = structural_df["Country"].nunique()
    
    # Get model performance for selected country
    if len(val_country) > 0:
        rmse = np.sqrt(np.mean(val_country["Error"] ** 2))
        mae = np.mean(np.abs(val_country["Error"]))
    else:
        rmse = 0
        mae = 0
    
    summary_text = f"""
    SUMMARY STATISTICS
    {'='*70}
    Total Structural Anomalies Detected: {total_anomalies_structural:,} weeks
    Total Transaction Anomalies Detected: {total_anomalies_transaction:,} transactions
    Countries Monitored: {total_countries}
    Model Performance ({country}): RMSE = ${rmse:,.2f}, MAE = ${mae:,.2f}
    """
    
    ax_d.text(
        0.5, 0.5,
        summary_text,
        ha="center",
        va="center",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    )
    
    return fig


def main():
    """Main execution function."""
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  PHASE 5: FINAL DASHBOARD & EVALUATION")
    print("=" * 70)
    
    # Load data
    print("\n[Loading] Dashboard data files...")
    try:
        validation_df, forecast_df, structural_df, transaction_df = load_dashboard_data(base_dir)
        print(f"  [OK] Validation data: {len(validation_df)} rows")
        print(f"  [OK] Forecast data: {len(forecast_df)} rows")
        print(f"  [OK] Structural anomalies: {len(structural_df)} rows")
        print(f"  [OK] Transaction anomalies: {len(transaction_df)} rows")
    except FileNotFoundError as e:
        print(f"  [!] Error: {e}")
        return
    
    # Create dashboard for Indonesia (ID)
    print("\n[Generating] Executive Dashboard for Indonesia (ID)...")
    fig = create_dashboard(
        validation_df,
        forecast_df,
        structural_df,
        transaction_df,
        country="ID"
    )
    
    # Save dashboard
    output_path = output_dir / "final_dashboard_ID.png"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"\n  [OK] Dashboard saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("  DASHBOARD GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nDashboard Components:")
    print(f"  1. Cash Flow Trend - Indonesia (ID) with anomaly overlays")
    print(f"  2. Anomaly Severity by Country (Bar Chart)")
    print(f"  3. Top 5 Transaction Anomalies (Audit Hit-List)")
    print(f"  4. Summary Statistics")
    
    plt.close(fig)


if __name__ == "__main__":
    main()

