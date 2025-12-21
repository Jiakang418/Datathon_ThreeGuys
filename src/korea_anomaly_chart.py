"""
Korea Cash Flow Timeline with Anomaly Highlights

This script creates a timeline chart showing Korea's cash flow with
anomalies highlighted, especially the high-severity ones.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def main():
    """Create Korea anomaly timeline chart."""
    base_dir = Path(__file__).resolve().parents[1]
    
    # Load data
    weekly_path = base_dir / "data" / "model_dataset" / "weekly_features.csv"
    anomalies_path = base_dir / "outputs" / "anomalies_structural_level.csv"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load weekly data
    weekly_df = pd.read_csv(weekly_path, parse_dates=["Week_Ending_Date"])
    korea_weekly = weekly_df[weekly_df["Country_Name"] == "KR"].copy()
    korea_weekly = korea_weekly.sort_values("Week_Ending_Date")
    
    # Load anomalies
    anomalies_df = pd.read_csv(anomalies_path, parse_dates=["Week_Ending_Date"])
    korea_anomalies = anomalies_df[anomalies_df["Country"] == "KR"].copy()
    korea_anomalies = korea_anomalies.sort_values("Week_Ending_Date")
    
    # Filter for flagged anomalies only
    korea_flagged = korea_anomalies[korea_anomalies["Anomaly_Flag"] == -1].copy()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # -------------------------------------------------------------------------
    # Panel 1: Cash Flow Timeline with Anomalies
    # -------------------------------------------------------------------------
    # Plot all cash flow
    ax1.plot(
        korea_weekly["Week_Ending_Date"],
        korea_weekly["Net_Cash_Flow"],
        "o-",
        color="#2E86AB",
        label="Net Cash Flow",
        linewidth=2,
        markersize=5,
        alpha=0.7
    )
    
    # Highlight anomalies
    if len(korea_flagged) > 0:
        # Color code by severity (more negative = more severe)
        for idx, row in korea_flagged.iterrows():
            score = row["Anomaly_Score"]
            # More negative scores = more severe = darker red
            severity = abs(score) / 0.67  # Normalize to -0.67 (the worst one)
            color_intensity = min(severity, 1.0)
            
            ax1.scatter(
                row["Week_Ending_Date"],
                row["Net_Cash_Flow"],
                s=300,
                marker="X",
                color=plt.cm.Reds(0.5 + 0.5 * color_intensity),
                edgecolors="darkred",
                linewidths=2,
                zorder=10,
                label="Anomaly" if idx == korea_flagged.index[0] else ""
            )
            
            # Add annotation for the most severe one
            if abs(score) >= 0.66:
                ax1.annotate(
                    f"Severity: {score:.3f}\n${row['Net_Cash_Flow']:,.0f}",
                    xy=(row["Week_Ending_Date"], row["Net_Cash_Flow"]),
                    xytext=(10, 30),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="red", lw=2),
                    fontsize=10,
                    fontweight="bold"
                )
    
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.3)
    ax1.set_ylabel("Net Cash Flow (USD)", fontsize=12, fontweight="bold")
    ax1.set_title("Korea Cash Flow Timeline - Anomaly Detection", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # -------------------------------------------------------------------------
    # Panel 2: Anomaly Scores Over Time
    # -------------------------------------------------------------------------
    # Plot anomaly scores
    ax2.plot(
        korea_anomalies["Week_Ending_Date"],
        korea_anomalies["Anomaly_Score"],
        "o-",
        color="#6C757D",
        label="Anomaly Score",
        linewidth=2,
        markersize=5,
        alpha=0.7
    )
    
    # Highlight flagged anomalies
    if len(korea_flagged) > 0:
        ax2.scatter(
            korea_flagged["Week_Ending_Date"],
            korea_flagged["Anomaly_Score"],
            s=200,
            marker="X",
            color="red",
            edgecolors="darkred",
            linewidths=2,
            zorder=10,
            label="Flagged Anomaly"
        )
    
    # Add threshold line (anomalies are typically < -0.5)
    ax2.axhline(y=-0.5, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="Severity Threshold")
    
    # Highlight the -0.66 score
    severe_anomaly = korea_flagged[korea_flagged["Anomaly_Score"] <= -0.66]
    if len(severe_anomaly) > 0:
        for idx, row in severe_anomaly.iterrows():
            ax2.annotate(
                f"{row['Anomaly_Score']:.3f}",
                xy=(row["Week_Ending_Date"], row["Anomaly_Score"]),
                xytext=(0, -40),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.8),
                fontsize=11,
                fontweight="bold",
                color="white"
            )
    
    ax2.set_xlabel("Week Ending Date", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Anomaly Score (Isolation Forest)", fontsize=12, fontweight="bold")
    ax2.set_title("Anomaly Severity Scores Over Time", fontsize=14, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "korea_anomaly_timeline.png"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Chart saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("KOREA ANOMALY SUMMARY")
    print("=" * 70)
    print(f"\nTotal weeks analyzed: {len(korea_weekly)}")
    print(f"Total anomalies detected: {len(korea_flagged)}")
    print(f"Anomaly rate: {len(korea_flagged)/len(korea_weekly)*100:.1f}%")
    
    if len(korea_flagged) > 0:
        print("\nTop 5 Most Severe Anomalies:")
        print("-" * 70)
        top_5 = korea_flagged.nsmallest(5, "Anomaly_Score")
        for idx, row in top_5.iterrows():
            print(f"  {row['Week_Ending_Date'].strftime('%Y-%m-%d')}: "
                  f"Score={row['Anomaly_Score']:.3f}, "
                  f"Cash Flow=${row['Net_Cash_Flow']:,.2f}")
    
    plt.close(fig)


if __name__ == "__main__":
    main()

