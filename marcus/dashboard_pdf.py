"""
Professional PDF Dashboard Generator
=====================================
AstraZeneca Cash Flow Challenge - Visual Storyboard

Creates a comprehensive 5-page professional PDF dashboard with:
- Page 1: Executive Summary & KPIs
- Page 2: Cash Flow Trends & Patterns
- Page 3: Model Comparison & Performance
- Page 4: ARIMA Forecasts with Summary Tables
- Page 5: Anomaly Detection & Recommendations

Author: Three Guys Team
Date: 2025-12-20
"""

import warnings
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MARCUS_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "model_dataset"
RAW_DIR = BASE_DIR / "Raw_Dataset"
OUTPUT_DIR = MARCUS_DIR / "dashboard_output"
OUTPUTS_DIR = BASE_DIR / "outputs"
ARIMA_DIR = BASE_DIR / "main" / "jiakang"

# Layout - more generous margins
ML, MR = 0.07, 0.93
CW = MR - ML  # 0.86

# Colors
C = {
    "purple": "#830051", "blue": "#00A0DF", "green": "#00843D",
    "red": "#E40046", "orange": "#FF6B35", "gold": "#F0B323",
    "gray": "#6E6E6E", "light": "#F5F5F5", "white": "#FFFFFF", "black": "#1A1A1A"
}

CC = {
    "ID": "#830051", "KR": "#00A0DF", "MY": "#68D2DF", "PH": "#00843D",
    "SS": "#E40046", "TH": "#FF6B35", "TW": "#6E6E6E", "VN": "#5C0039"
}


# --------------------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------------------

def load_all_data():
    """Load all trained model outputs."""
    data = {}
    
    files = {
        "weekly": (DATA_DIR / "processed_weekly_cashflow.csv", ["Week_Ending_Date"]),
        "arima_metrics": (OUTPUTS_DIR / "hybrid_arima_backtest_metrics.csv", None),
        "arima_forecast": (OUTPUTS_DIR / "hybrid_arima_future_forecasts.csv", ["Week_Ending_Date"]),
        "arima_comparison": (OUTPUTS_DIR / "hybrid_arima_actual_vs_predicted.csv", ["Week_Ending_Date"]),
        "arima_summary": (ARIMA_DIR / "arima_best_model_summary.csv", None),
        "prophet_metrics": (OUTPUTS_DIR / "hybrid_prophet_backtest_metrics.csv", None),
        "naive_metrics": (OUTPUTS_DIR / "hybrid_naive_backtest_metrics.csv", None),
        "structural_anomalies": (OUTPUTS_DIR / "anomalies_structural_level.csv", ["Week_Ending_Date"]),
        "transaction_anomalies": (OUTPUTS_DIR / "anomalies_transaction_level.csv", None),
    }
    
    for name, (path, dates) in files.items():
        if path.exists():
            data[name] = pd.read_csv(path, parse_dates=dates) if dates else pd.read_csv(path)
            print(f"  [OK] {name}: {len(data[name]):,} records")
    
    return data


def fmt(v, d=1):
    """Format currency."""
    if abs(v) >= 1e6:
        return f"${v/1e6:.{d}f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.{d}f}K"
    return f"${v:.0f}"


def header(fig, title, sub, pg):
    """Add page header/footer."""
    fig.text(0.5, 0.96, title, fontsize=16, fontweight="bold", color=C["purple"], ha="center")
    fig.text(0.5, 0.92, sub, fontsize=10, color=C["gray"], ha="center")
    fig.add_artist(plt.Line2D([ML, MR], [0.89, 0.89], transform=fig.transFigure, color=C["purple"], lw=2))
    fig.text(ML, 0.02, "Three Guys Team | AstraZeneca Datathon 2025", fontsize=8, color=C["gray"])
    fig.text(MR, 0.02, f"Page {pg}/5", fontsize=8, color=C["gray"], ha="right")


# --------------------------------------------------------------------------------------
# PAGE 1: EXECUTIVE SUMMARY
# --------------------------------------------------------------------------------------

def page1(pdf, data):
    """Executive Summary."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C["white"])
    
    # Header
    fig.text(0.5, 0.96, "ASTRAZENECA CASH FLOW FORECASTING", fontsize=20, fontweight="bold", 
             color=C["purple"], ha="center")
    fig.text(0.5, 0.92, f"Intelligent Cash Flow Prediction | ARIMA Model | {datetime.now().strftime('%B %d, %Y')}", 
             fontsize=10, color=C["gray"], ha="center")
    fig.add_artist(plt.Line2D([ML, MR], [0.89, 0.89], transform=fig.transFigure, color=C["purple"], lw=3))
    
    # Get metrics
    w = data.get("weekly", pd.DataFrame())
    a = data.get("arima_metrics", pd.DataFrame())
    sa = data.get("structural_anomalies", pd.DataFrame())
    ta = data.get("transaction_anomalies", pd.DataFrame())
    
    n_countries = w["Country_Name"].nunique() if not w.empty else 0
    n_weeks = w["Week_Ending_Date"].nunique() if not w.empty else 0
    total_cf = w["Net_Cash_Flow"].sum() if not w.empty else 0
    avg_mape = a["MAPE_percent"].mean() if not a.empty else 0
    best = a.loc[a["MAPE_percent"].idxmin()] if not a.empty else None
    n_struct = (sa["Anomaly_Flag"] == -1).sum() if not sa.empty else 0
    n_txn = len(ta) if not ta.empty else 0
    
    # KPI Cards Row (3 cards)
    kpis = [
        ("COUNTRIES", str(n_countries), "Entities Analyzed", C["blue"]),
        ("DATA PERIOD", f"{n_weeks} Weeks", "Jan - Nov 2025", C["purple"]),
        ("NET CASH FLOW", fmt(total_cf), "Total (All Countries)", C["red"]),
    ]
    
    card_w = 0.26
    gap = (CW - 3 * card_w) / 2
    
    for i, (title, val, sub, color) in enumerate(kpis):
        x = ML + i * (card_w + gap)
        ax = fig.add_axes([x, 0.72, card_w, 0.14])
        ax.set_facecolor(C["light"])
        ax.add_patch(mpatches.Rectangle((0, 0), 0.02, 1, transform=ax.transAxes, color=color))
        ax.text(0.5, 0.75, title, transform=ax.transAxes, fontsize=10, color=C["gray"], 
                ha="center", fontweight="bold")
        ax.text(0.5, 0.42, val, transform=ax.transAxes, fontsize=22, color=color, 
                ha="center", fontweight="bold")
        ax.text(0.5, 0.12, sub, transform=ax.transAxes, fontsize=9, color=C["gray"], ha="center")
        ax.axis("off")
    
    # Best Model Card
    ax_best = fig.add_axes([ML, 0.56, CW, 0.12])
    ax_best.set_facecolor("#E8F5E9")
    ax_best.add_patch(mpatches.Rectangle((0, 0), 0.008, 1, transform=ax_best.transAxes, color=C["green"]))
    ax_best.text(0.02, 0.5, "BEST MODEL: ARIMA", transform=ax_best.transAxes, fontsize=16, 
                 color=C["green"], va="center", fontweight="bold")
    if best is not None:
        ax_best.text(0.35, 0.65, f"Avg MAPE: {avg_mape:.1f}%", transform=ax_best.transAxes, 
                     fontsize=11, color=C["black"])
        ax_best.text(0.35, 0.30, f"Best: {best['Country']} ({best['MAPE_percent']:.1f}%)", 
                     transform=ax_best.transAxes, fontsize=11, color=C["black"])
        ax_best.text(0.65, 0.65, f"Structural Anomalies: {n_struct}", 
                     transform=ax_best.transAxes, fontsize=11, color=C["black"])
        ax_best.text(0.65, 0.30, f"Transaction Anomalies: {n_txn:,}", 
                     transform=ax_best.transAxes, fontsize=11, color=C["black"])
    ax_best.axis("off")
    for s in ax_best.spines.values():
        s.set_visible(True)
        s.set_color(C["green"])
        s.set_linewidth(2)
    
    # Main Chart (left)
    ax_main = fig.add_axes([ML, 0.10, 0.50, 0.40])
    if not w.empty:
        totals = w.groupby("Week_Ending_Date")["Net_Cash_Flow"].sum().sort_index()
        colors = [C["green"] if v >= 0 else C["red"] for v in totals.values]
        ax_main.bar(range(len(totals)), totals.values, color=colors, alpha=0.8)
        ax_main.axhline(y=0, color=C["gray"], lw=1)
        z = np.polyfit(range(len(totals)), totals.values, 1)
        ax_main.plot(range(len(totals)), np.poly1d(z)(range(len(totals))), 
                    color=C["purple"], lw=2, ls="--", label="Trend")
        ax_main.set_title("Weekly Net Cash Flow Trend", fontsize=11, fontweight="bold", pad=8)
        ax_main.set_xlabel("Week", fontsize=9)
        ax_main.set_ylabel("USD", fontsize=9)
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(x, 0)))
        ax_main.grid(True, alpha=0.3, axis="y")
        ax_main.legend(fontsize=9)
        ax_main.tick_params(labelsize=8)
    
    # Summary Panel (right)
    ax_sum = fig.add_axes([0.60, 0.10, 0.33, 0.40])
    ax_sum.set_facecolor(C["light"])
    ax_sum.axis("off")
    
    summary = f"""PROJECT SUMMARY
========================

DATA SCOPE
  Weeks: {n_weeks}
  Countries: {n_countries}
  Transactions: 84,528

MODELS EVALUATED
  - ARIMA (Winner)
  - Prophet
  - Naive Baseline

ANALYSIS
  - Time Series Forecasting
  - Anomaly Detection
  - Model Backtesting

OUTPUTS
  - 1-Month Forecasts
  - 6-Month Projections
  - Risk Alerts"""
    
    ax_sum.text(0.08, 0.94, summary, transform=ax_sum.transAxes, fontsize=9, 
                va="top", fontfamily="monospace", color=C["black"], linespacing=1.3)
    for s in ax_sum.spines.values():
        s.set_visible(True)
        s.set_color(C["purple"])
        s.set_linewidth(2)
    
    fig.text(ML, 0.02, "Three Guys Team | AstraZeneca Datathon 2025", fontsize=8, color=C["gray"])
    fig.text(MR, 0.02, "Page 1/5", fontsize=8, color=C["gray"], ha="right")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 1: Executive Summary")


# --------------------------------------------------------------------------------------
# PAGE 2: CASH FLOW TRENDS
# --------------------------------------------------------------------------------------

def page2(pdf, data):
    """Cash Flow Trends."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C["white"])
    header(fig, "CASH FLOW TRENDS & PATTERNS", "Historical analysis across 8 countries and 44 weeks", 2)
    
    w = data.get("weekly", pd.DataFrame())
    hw = (CW - 0.06) / 2  # half width with gap
    
    # Chart 1: By Country (top left)
    ax1 = fig.add_axes([ML, 0.52, hw, 0.32])
    if not w.empty:
        for c in w["Country_Name"].unique():
            d = w[w["Country_Name"] == c].sort_values("Week_Ending_Date")
            ax1.plot(d["Week_Ending_Date"], d["Net_Cash_Flow"], label=c, 
                    color=CC.get(c, C["gray"]), lw=1.5, alpha=0.85)
        ax1.axhline(y=0, color=C["gray"], lw=1, ls="--")
        ax1.set_title("Weekly Cash Flow by Country", fontsize=11, fontweight="bold", pad=6)
        ax1.set_xlabel("Date", fontsize=9)
        ax1.set_ylabel("Net Cash Flow (USD)", fontsize=9)
        ax1.legend(loc="upper right", fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(x, 0)))
        ax1.tick_params(labelsize=8)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=25, ha="right")
    
    # Chart 2: Operating vs Financing (top right)
    ax2 = fig.add_axes([ML + hw + 0.06, 0.52, hw, 0.32])
    if not w.empty:
        t = w.groupby("Week_Ending_Date").agg({
            "Operating_Cash_Flow": "sum", 
            "Financing_Cash_Flow": "sum"
        }).sort_index()
        ax2.fill_between(range(len(t)), t["Operating_Cash_Flow"], alpha=0.7, 
                        color=C["blue"], label="Operating")
        ax2.fill_between(range(len(t)), t["Financing_Cash_Flow"], alpha=0.7, 
                        color=C["orange"], label="Financing")
        ax2.axhline(y=0, color=C["gray"], lw=1)
        ax2.set_title("Operating vs Financing Cash Flow", fontsize=11, fontweight="bold", pad=6)
        ax2.set_xlabel("Week", fontsize=9)
        ax2.set_ylabel("USD", fontsize=9)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(x, 0)))
        ax2.tick_params(labelsize=8)
    
    # Chart 3: Country Ranking (bottom left)
    ax3 = fig.add_axes([ML, 0.10, hw, 0.34])
    if not w.empty:
        totals = w.groupby("Country_Name")["Net_Cash_Flow"].sum().sort_values()
        colors = [CC.get(c, C["gray"]) for c in totals.index]
        bars = ax3.barh(totals.index, totals.values, color=colors, alpha=0.85)
        ax3.axvline(x=0, color=C["gray"], lw=1)
        for bar, v in zip(bars, totals.values):
            offset = abs(totals.values).max() * 0.02
            x_pos = v + offset if v >= 0 else v - offset
            ax3.text(x_pos, bar.get_y() + bar.get_height()/2, fmt(v), 
                    ha="left" if v >= 0 else "right", va="center", fontsize=8, fontweight="bold")
        ax3.set_title("Total Net Cash Flow by Country", fontsize=11, fontweight="bold", pad=6)
        ax3.set_xlabel("USD", fontsize=9)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(x, 0)))
        ax3.grid(True, alpha=0.3, axis="x")
        ax3.tick_params(labelsize=9)
    
    # Chart 4: Monthly Pattern (bottom right)
    ax4 = fig.add_axes([ML + hw + 0.06, 0.10, hw, 0.34])
    if not w.empty:
        wc = w.copy()
        wc["Month"] = wc["Week_Ending_Date"].dt.month_name().str[:3]
        monthly = wc.groupby("Month")["Net_Cash_Flow"].sum()
        order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
        monthly = monthly.reindex([m for m in order if m in monthly.index])
        colors = [C["green"] if v >= 0 else C["red"] for v in monthly.values]
        ax4.bar(monthly.index, monthly.values, color=colors, alpha=0.85)
        ax4.axhline(y=0, color=C["gray"], lw=1)
        ax4.set_title("Monthly Cash Flow Pattern", fontsize=11, fontweight="bold", pad=6)
        ax4.set_xlabel("Month", fontsize=9)
        ax4.set_ylabel("USD", fontsize=9)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(x, 0)))
        ax4.grid(True, alpha=0.3, axis="y")
        ax4.tick_params(labelsize=9)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=25, ha="right")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 2: Cash Flow Trends")


# --------------------------------------------------------------------------------------
# PAGE 3: MODEL COMPARISON
# --------------------------------------------------------------------------------------

def page3(pdf, data):
    """Model Comparison."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C["white"])
    header(fig, "MODEL COMPARISON & PERFORMANCE", "ARIMA vs Prophet vs Naive | 4-Week Backtesting Validation", 3)
    
    am = data.get("arima_metrics", pd.DataFrame())
    pm = data.get("prophet_metrics", pd.DataFrame())
    nm = data.get("naive_metrics", pd.DataFrame())
    
    # Chart: MAPE Comparison (top half)
    ax1 = fig.add_axes([ML, 0.52, CW, 0.32])
    if not am.empty and not pm.empty and not nm.empty:
        countries = am["Country"].values
        x = np.arange(len(countries))
        w = 0.25
        
        a_mape = am.set_index("Country")["MAPE_percent"]
        p_mape = pm.set_index("Country")["MAPE_percent"]
        n_mape = nm.set_index("Country")["MAPE_percent"]
        
        cap = 100
        ax1.bar(x - w, [min(a_mape.get(c, 0), cap) for c in countries], w, 
               label="ARIMA (Best)", color=C["green"], alpha=0.85)
        ax1.bar(x, [min(p_mape.get(c, 0), cap) for c in countries], w, 
               label="Prophet", color=C["purple"], alpha=0.85)
        ax1.bar(x + w, [min(n_mape.get(c, 0), cap) for c in countries], w, 
               label="Naive", color=C["blue"], alpha=0.85)
        
        ax1.axhline(y=25, color=C["green"], lw=1.5, ls="--", alpha=0.8)
        ax1.axhline(y=50, color=C["orange"], lw=1.5, ls="--", alpha=0.8)
        ax1.text(len(countries) - 0.5, 27, "Good (<25%)", fontsize=9, color=C["green"])
        ax1.text(len(countries) - 0.5, 52, "Fair (<50%)", fontsize=9, color=C["orange"])
        
        ax1.set_title("MAPE Comparison by Country (Lower is Better)", fontsize=12, fontweight="bold", pad=8)
        ax1.set_ylabel("MAPE (%)", fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(countries, fontsize=10)
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.tick_params(labelsize=9)
        ax1.set_ylim(0, cap + 10)
    
    # Table Title (positioned clearly above table)
    fig.text(0.5, 0.46, "DETAILED MODEL PERFORMANCE METRICS", fontsize=12, fontweight="bold",
            ha="center", color=C["purple"])
    
    # Performance Table (bottom half)
    ax2 = fig.add_axes([ML, 0.08, CW, 0.34])
    ax2.axis("off")
    
    if not am.empty:
        table_data = []
        for _, r in am.iterrows():
            c = r["Country"]
            a, p, n = r["MAPE_percent"], p_mape.get(c, 0), n_mape.get(c, 0)
            best_model = "ARIMA" if a <= min(p, n) else "Prophet" if p <= n else "Naive"
            status = "Good" if a < 30 else "Fair" if a < 60 else "Review"
            table_data.append([
                c, 
                f"${r['RMSE_USD']:,.0f}", 
                f"${r['MAE_USD']:,.0f}", 
                f"{a:.1f}%", 
                f"{p:.1f}%", 
                f"{n:.1f}%", 
                best_model, 
                status
            ])
        
        cols = ["Country", "RMSE", "MAE", "ARIMA", "Prophet", "Naive", "Winner", "Status"]
        table = ax2.table(
            cellText=table_data, 
            colLabels=cols, 
            loc="upper center", 
            cellLoc="center",
            colWidths=[0.10, 0.13, 0.13, 0.10, 0.10, 0.10, 0.10, 0.10]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)
        
        # Style header
        for i in range(8):
            table[(0, i)].set_facecolor(C["purple"])
            table[(0, i)].set_text_props(fontweight="bold", color="white")
        
        # Style cells
        for ri in range(1, len(table_data) + 1):
            # Winner column
            if "ARIMA" in table_data[ri-1][6]:
                table[(ri, 6)].set_text_props(color=C["green"], fontweight="bold")
            # Status column
            s = table_data[ri-1][7]
            color = C["green"] if s == "Good" else C["orange"] if s == "Fair" else C["red"]
            table[(ri, 7)].set_text_props(color=color, fontweight="bold")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 3: Model Comparison")


# --------------------------------------------------------------------------------------
# PAGE 4: ARIMA FORECASTS
# --------------------------------------------------------------------------------------

def page4(pdf, data):
    """ARIMA Forecasts with Summary Tables."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C["white"])
    header(fig, "ARIMA FORECAST RESULTS", "1-Month and 6-Month Predictions with Confidence Intervals", 4)
    
    fc = data.get("arima_forecast", pd.DataFrame())
    cp = data.get("arima_comparison", pd.DataFrame())
    
    # Chart: Forecast Visualization (top)
    ax1 = fig.add_axes([ML, 0.58, CW, 0.26])
    if not fc.empty:
        f1m = fc[fc["Horizon"] == "1_month"]
        for c in f1m["Country"].unique():
            d = f1m[f1m["Country"] == c].sort_values("Week_Ending_Date")
            ax1.plot(d["Week_Ending_Date"], d["Predicted_Cash_Flow"], marker="o", ms=4, lw=1.5,
                    label=c, color=CC.get(c, C["gray"]))
            ax1.fill_between(d["Week_Ending_Date"], 
                           d["Prediction_Lower_95CI"], 
                           d["Prediction_Upper_95CI"],
                           alpha=0.12, color=CC.get(c, C["gray"]))
        ax1.axhline(y=0, color=C["gray"], lw=1, ls="--")
        ax1.set_title("1-Month ARIMA Forecast with 95% CI", fontsize=11, fontweight="bold", pad=6)
        ax1.set_xlabel("Week", fontsize=9)
        ax1.set_ylabel("Predicted Cash Flow (USD)", fontsize=9)
        ax1.legend(loc="upper right", fontsize=7, ncol=4)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(x, 0)))
        ax1.tick_params(labelsize=8)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha="right")
    
    hw = (CW - 0.06) / 2
    
    # Left Table Title
    fig.text(ML + hw/2, 0.52, "FORECAST SUMMARY BY COUNTRY", fontsize=11, fontweight="bold",
            ha="center", color=C["purple"])
    
    # Left Table: Forecast Summary
    ax2 = fig.add_axes([ML, 0.10, hw, 0.38])
    ax2.axis("off")
    
    if not fc.empty:
        table_data = []
        for c in fc["Country"].unique():
            f1 = fc[(fc["Country"] == c) & (fc["Horizon"] == "1_month")]
            f6 = fc[(fc["Country"] == c) & (fc["Horizon"] == "6_month")]
            cf1 = f1["Predicted_Cash_Flow"].sum()
            cf6 = f6["Predicted_Cash_Flow"].sum()
            avg = cf6 / len(f6) if len(f6) > 0 else 0
            table_data.append([c, fmt(cf1), fmt(cf6), fmt(avg)])
        
        cols = ["Country", "1-Month", "6-Month", "Avg/Week"]
        table = ax2.table(
            cellText=table_data, 
            colLabels=cols, 
            loc="upper center", 
            cellLoc="center",
            colWidths=[0.20, 0.28, 0.28, 0.24]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.7)
        
        for i in range(4):
            table[(0, i)].set_facecolor(C["green"])
            table[(0, i)].set_text_props(fontweight="bold", color="white")
    
    # Right Table Title
    fig.text(ML + hw + 0.06 + hw/2, 0.52, "VALIDATION: ACTUAL vs PREDICTED", fontsize=11, fontweight="bold",
            ha="center", color=C["purple"])
    
    # Right Table: Validation
    ax3 = fig.add_axes([ML + hw + 0.06, 0.10, hw, 0.38])
    ax3.axis("off")
    
    if not cp.empty:
        val_data = []
        for c in cp["Country"].unique():
            d = cp[cp["Country"] == c]
            actual = d["Actual_Cash_Flow"].sum()
            pred = d["Predicted_Cash_Flow"].sum()
            err = abs(actual - pred)
            acc = max(0, 100 - (err / abs(actual) * 100)) if actual != 0 else 0
            val_data.append([c, fmt(actual), fmt(pred), f"{acc:.0f}%"])
        
        cols = ["Country", "Actual", "Predicted", "Accuracy"]
        table = ax3.table(
            cellText=val_data, 
            colLabels=cols, 
            loc="upper center", 
            cellLoc="center",
            colWidths=[0.20, 0.28, 0.28, 0.20]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.7)
        
        for i in range(4):
            table[(0, i)].set_facecolor(C["blue"])
            table[(0, i)].set_text_props(fontweight="bold", color="white")
        
        # Color accuracy
        for ri in range(1, len(val_data) + 1):
            acc_val = float(val_data[ri-1][3].replace("%", ""))
            color = C["green"] if acc_val >= 70 else C["orange"] if acc_val >= 50 else C["red"]
            table[(ri, 3)].set_text_props(color=color, fontweight="bold")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 4: ARIMA Forecasts")


# --------------------------------------------------------------------------------------
# PAGE 5: ANOMALIES & RECOMMENDATIONS
# --------------------------------------------------------------------------------------

def page5(pdf, data):
    """Anomaly Detection & Recommendations."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C["white"])
    header(fig, "ANOMALY DETECTION & RECOMMENDATIONS", "Dual-layer: Structural (Isolation Forest) + Transaction (Z-Score)", 5)
    
    sa = data.get("structural_anomalies", pd.DataFrame())
    ta = data.get("transaction_anomalies", pd.DataFrame())
    am = data.get("arima_metrics", pd.DataFrame())
    
    hw = (CW - 0.06) / 2
    n_struct = (sa["Anomaly_Flag"] == -1).sum() if not sa.empty else 0
    n_txn = len(ta) if not ta.empty else 0
    
    # Chart: Anomaly Timeline (top left)
    ax1 = fig.add_axes([ML, 0.54, hw, 0.30])
    if not sa.empty:
        for c in sa["Country"].unique():
            d = sa[sa["Country"] == c].sort_values("Week_Ending_Date")
            ax1.plot(d["Week_Ending_Date"], d["Net_Cash_Flow"], 
                    color=CC.get(c, C["gray"]), alpha=0.3, lw=1)
            a = d[d["Anomaly_Flag"] == -1]
            if not a.empty:
                ax1.scatter(a["Week_Ending_Date"], a["Net_Cash_Flow"], 
                           color=C["red"], s=35, zorder=5, alpha=0.8)
        ax1.axhline(y=0, color=C["gray"], lw=1, ls="--")
        ax1.scatter([], [], color=C["red"], s=35, label=f"Anomaly ({n_struct})")
        ax1.set_title("Structural Anomalies Timeline", fontsize=11, fontweight="bold", pad=6)
        ax1.set_xlabel("Date", fontsize=9)
        ax1.set_ylabel("Cash Flow (USD)", fontsize=9)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(x, 0)))
        ax1.tick_params(labelsize=8)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=25, ha="right")
    
    # Anomaly Summary Box (top right)
    ax2 = fig.add_axes([ML + hw + 0.06, 0.54, hw, 0.30])
    ax2.set_facecolor("#FFF8E1")
    ax2.axis("off")
    
    anomaly_text = f"""ANOMALY DETECTION SUMMARY
==============================

STRUCTURAL ANOMALIES
  Detected: {n_struct}
  Method: Isolation Forest (5%)
  Features: Cash Flow, Rolling Stats

TRANSACTION ANOMALIES
  Detected: {n_txn:,}
  Method: Z-Score (|Z| > 3.5)

TOP RISK COUNTRIES"""
    
    if not sa.empty:
        top = sa[sa["Anomaly_Flag"] == -1].groupby("Country").size()
        top = top.sort_values(ascending=False).head(3)
        for c, n in top.items():
            anomaly_text += f"\n  {c}: {n} weeks"
    
    ax2.text(0.06, 0.94, anomaly_text, transform=ax2.transAxes, fontsize=9, 
            va="top", fontfamily="monospace", linespacing=1.25)
    for s in ax2.spines.values():
        s.set_visible(True)
        s.set_color(C["orange"])
        s.set_linewidth(2)
    
    # Key Insights (bottom left)
    ax3 = fig.add_axes([ML, 0.08, hw, 0.40])
    ax3.set_facecolor(C["light"])
    ax3.axis("off")
    
    best_c = am.loc[am["MAPE_percent"].idxmin(), "Country"] if not am.empty else "N/A"
    worst_c = am.loc[am["MAPE_percent"].idxmax(), "Country"] if not am.empty else "N/A"
    
    insights = f"""KEY INSIGHTS
==============================

CASH FLOW PATTERNS
  - All countries negative net flow
  - Operating flows dominate
  - Strong weekly seasonality
  - Month-end payment cycles

MODEL PERFORMANCE
  - ARIMA = Best model
  - Best: {best_c}
  - Review: {worst_c}
  - 4-week holdout validation

DATA QUALITY
  - 44 weeks historical data
  - 84,528 transactions
  - Minimal data gaps"""
    
    ax3.text(0.06, 0.94, insights, transform=ax3.transAxes, fontsize=9, 
            va="top", fontfamily="monospace", linespacing=1.25)
    for s in ax3.spines.values():
        s.set_visible(True)
        s.set_color(C["purple"])
        s.set_linewidth(2)
    
    # Recommendations (bottom right)
    ax4 = fig.add_axes([ML + hw + 0.06, 0.08, hw, 0.40])
    ax4.set_facecolor("#E3F2FD")
    ax4.axis("off")
    
    recs = """RECOMMENDATIONS
==============================

SHORT-TERM (1-4 Weeks)
  - Deploy ARIMA in production
  - Set anomaly alerts (>2 std)
  - Monitor KR, TW closely

MEDIUM-TERM (1-6 Months)
  - Weekly rolling updates
  - Add FX rate regressors
  - Country-specific tuning

MODEL IMPROVEMENTS
  - Expand data (2+ years)
  - Test ensemble models
  - Add holiday adjustments
  - Real-time integration"""
    
    ax4.text(0.06, 0.94, recs, transform=ax4.transAxes, fontsize=9, 
            va="top", fontfamily="monospace", linespacing=1.25)
    for s in ax4.spines.values():
        s.set_visible(True)
        s.set_color(C["blue"])
        s.set_linewidth(2)
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 5: Anomalies & Recommendations")


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------

def generate_dashboard_pdf():
    """Generate the 5-page PDF dashboard."""
    print("\n" + "=" * 60)
    print("  GENERATING PROFESSIONAL DASHBOARD")
    print("  AstraZeneca Cash Flow Challenge")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[Step 1] Loading trained model outputs...")
    data = load_all_data()
    
    output = OUTPUT_DIR / "AstraZeneca_CashFlow_Dashboard.pdf"
    print(f"\n[Step 2] Generating PDF...")
    
    with PdfPages(output) as pdf:
        page1(pdf, data)
        page2(pdf, data)
        page3(pdf, data)
        page4(pdf, data)
        page5(pdf, data)
    
    print("\n" + "=" * 60)
    print("  DASHBOARD COMPLETE!")
    print("=" * 60)
    print(f"\n  Output: {output}")
    print(f"  Size: {output.stat().st_size / 1024:.1f} KB")
    print(f"  Pages: 5")
    
    return output


if __name__ == "__main__":
    generate_dashboard_pdf()
