"""
Professional PDF Dashboard Generator
=====================================
AstraZeneca Cash Flow Challenge - Visual Storyboard

Creates a comprehensive 5-page professional PDF dashboard with:
- Page 1: Executive Summary & Business Context
- Page 2: Methodology & Data Pipeline (Phase 1-5)
- Page 3: Model Selection Rationale & Comparison (WHY ARIMA)
- Page 4: Forecast Results & Spotlight: ID (Indonesia)
- Page 5: Anomaly Detection Methodology & Action Plan

Key Additions:
✓ WHY ARIMA was chosen (criteria, trade-offs)
✓ HOW anomalies were detected (methodology)
✓ WHAT features were engineered and why
✓ HOW to interpret confidence intervals
✓ WHICH actions are highest priority
✓ HOW data was preprocessed (pipeline)
✓ Hybrid ARIMA + XGBoost explanation

Author: Three Guys Team
Date: 2025-12-21
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
        "arima_metrics": (ARIMA_DIR / "arima_results" / "arima_backtest_metrics.csv", None),
        "arima_forecast": (ARIMA_DIR / "arima_results" / "arima_future_forecasts.csv", ["Week_Ending_Date"]),
        "arima_comparison": (ARIMA_DIR / "arima_results" / "arima_actual_vs_predicted.csv", ["Week_Ending_Date"]),
        "arima_summary": (ARIMA_DIR / "arima_best_model_summary.csv", None),
        "prophet_metrics": (MARCUS_DIR / "prophet_results" / "prophet_backtest_metrics.csv", None),
        "naive_metrics": (MARCUS_DIR / "naive_results" / "naive_backtest_metrics.csv", None),
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
# PAGE 2: METHODOLOGY & DATA PIPELINE
# --------------------------------------------------------------------------------------

def page2(pdf, data):
    """Methodology & Data Pipeline - Visual Flow Diagram."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C["white"])
    header(fig, "METHODOLOGY & DATA PIPELINE", "5-Phase Approach: Data → Features → Models → Forecasts → Anomalies", 2)
    
    # Phase Flow Diagram (top - visual pipeline)
    ax_flow = fig.add_axes([ML, 0.70, CW, 0.16])
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 10)
    ax_flow.axis("off")
    
    # Draw phase boxes
    phases = [
        (1, "PHASE 1\nPreprocessing", C["blue"]),
        (3, "PHASE 2\nFeatures", C["purple"]),
        (5, "PHASE 3\nModeling", C["green"]),
        (7, "PHASE 4\nAnomalies", C["red"]),
        (9, "PHASE 5\nForecasting", C["gold"])
    ]
    
    for x, label, color in phases:
        rect = mpatches.FancyBboxPatch((x-0.8, 4), 1.6, 2, boxstyle="round,pad=0.1", 
                                       facecolor=color, edgecolor=color, alpha=0.2, linewidth=2)
        ax_flow.add_patch(rect)
        ax_flow.text(x, 5, label, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
        
        # Arrow
        if x < 9:
            ax_flow.annotate("", xy=(x+1.2, 5), xytext=(x+0.8, 5),
                           arrowprops=dict(arrowstyle="->", lw=2, color=C["gray"]))
    
    # Phase Details Table (middle)
    ax_table = fig.add_axes([ML, 0.44, CW, 0.22])
    ax_table.axis("off")
    
    phase_data = [
        ["Phase 1", "Data Preprocessing", "84,528 txns → 352 weekly", "44 weeks | 8 countries"],
        ["Phase 2", "Feature Engineering", "6 features created", "Lag 1,2,4 | Roll Mean/Std"],
        ["Phase 3", "Hybrid Training", "ARIMA(1,1,1) + XGBoost", "4-week validation holdout"],
        ["Phase 4", "Anomaly Detection", "Z-score + Isolation Forest", "16 struct | 1,869 txn flags"],
        ["Phase 5", "Production Forecast", "1m & 6m predictions", "95% CI | Weekly refresh"]
    ]
    
    table = ax_table.table(cellText=phase_data,
                          colLabels=["Phase", "Task", "Output", "Details"],
                          loc="center", cellLoc="left",
                          colWidths=[0.10, 0.25, 0.32, 0.33])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.8)
    
    for i in range(4):
        table[(0, i)].set_facecolor(C["purple"])
        table[(0, i)].set_text_props(fontweight="bold", color="white", ha="center")
    
    colors = [C["blue"], C["purple"], C["green"], C["red"], C["gold"]]
    for i in range(1, 6):
        table[(i, 0)].set_facecolor(colors[i-1])
        table[(i, 0)].set_text_props(fontweight="bold", color="white", ha="center")
    
    # Feature Engineering Table (bottom left)
    hw = (CW - 0.06) / 2
    fig.text(ML + hw/2, 0.40, "FEATURE ENGINEERING", fontsize=10, fontweight="bold",
            ha="center", color=C["orange"])
    
    ax_feat = fig.add_axes([ML, 0.07, hw, 0.28])
    ax_feat.axis("off")
    
    feat_data = [
        ["Lag_1", "1-week ago", "Recent trend", "★★★"],
        ["Lag_2", "2-weeks ago", "Bi-weekly cycle", "★★☆"],
        ["Lag_4", "4-weeks ago", "Monthly pattern", "★★☆"],
        ["Roll_Mean_4w", "4-week avg", "Baseline level", "★★☆"],
        ["Roll_Std_4w", "4-week volatility", "Risk indicator", "★☆☆"],
        ["Implied_Invest", "Derived CF", "Balance check", "★☆☆"]
    ]
    
    feat_table = ax_feat.table(cellText=feat_data,
                               colLabels=["Feature", "Definition", "Purpose", "Importance"],
                               loc="center", cellLoc="left",
                               colWidths=[0.20, 0.25, 0.35, 0.20])
    feat_table.auto_set_font_size(False)
    feat_table.set_fontsize(8)
    feat_table.scale(1.0, 1.6)
    
    for i in range(4):
        feat_table[(0, i)].set_facecolor(C["orange"])
        feat_table[(0, i)].set_text_props(fontweight="bold", color="white", ha="center")
    
    # Hybrid Architecture Diagram (bottom right)
    fig.text(ML + hw + 0.06 + hw/2, 0.40, "HYBRID MODEL ARCHITECTURE", fontsize=10, fontweight="bold",
            ha="center", color=C["green"])
    
    ax_arch = fig.add_axes([ML + hw + 0.06, 0.07, hw, 0.28])
    ax_arch.set_xlim(0, 10)
    ax_arch.set_ylim(0, 15) # Increased height for better spacing
    ax_arch.axis("off")
    
    # Model components with improved spacing and labels
    # Format: (x_center, y_center, label, color)
    components = [
        (5, 13.5, "INPUT DATA\nWeekly Cash Flow", C["gray"]),
        (5, 10.5, "ARIMA COMPONENT\nModeling Trend & Seasonality", C["blue"]),
        (5, 7.5, "RESIDUAL ERROR\nActual - ARIMA Prediction", C["orange"]),
        (5, 4.5, "XGBOOST COMPONENT\nFeature-Based Correction", C["purple"]),
        (5, 1.5, "FINAL ENSEMBLE\nHybrid ARI-XGB Forecast", C["green"])
    ]
    
    for x, y, label, bgcolor in components:
        # Box (Slightly narrower to ensure no bleed)
        rect = mpatches.FancyBboxPatch((x-2.8, y-1.1), 5.6, 2.2, boxstyle="round,pad=0.1",
                                       facecolor=bgcolor, edgecolor=bgcolor, alpha=0.9, linewidth=1)
        ax_arch.add_patch(rect)
        
        # Label
        ax_arch.text(x, y, label, ha="center", va="center", fontsize=8.5, 
                    fontweight="bold", color="white", linespacing=1.3)
        
        # Arrow down (Precise gap connection)
        if y > 1.5:
            ax_arch.annotate("", xy=(x, y-1.9), xytext=(x, y-1.1),
                           arrowprops=dict(arrowstyle="-|>", lw=2, color=C["gray"], mutation_scale=12))
    
    # Feature Input Arrow (Moved inside the axes to avoid overlap with left table)
    ax_arch.annotate("External\nFeatures", xy=(2.2, 4.5), xytext=(0.8, 4.5),
                   arrowprops=dict(arrowstyle="-|>", lw=2, color=C["purple"], mutation_scale=12),
                   fontsize=8, fontweight="bold", color=C["purple"], ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["purple"], alpha=1.0, lw=1))
    
    ax_arch.set_xlim(0, 10)
    ax_arch.set_ylim(0, 15)
    ax_arch.axis("off")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 2: Methodology & Pipeline (Visual)")



# --------------------------------------------------------------------------------------
# PAGE 3: MODEL COMPARISON
# --------------------------------------------------------------------------------------

def page3(pdf, data):
    """Model Selection Rationale & Comparison."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C["white"])
    header(fig, "MODEL SELECTION RATIONALE & COMPARISON", "Why Hybrid ARIMA Was Chosen | 4-Week Backtesting Validation", 3)
    
    am = data.get("arima_metrics", pd.DataFrame())
    pm = data.get("prophet_metrics", pd.DataFrame())
    nm = data.get("naive_metrics", pd.DataFrame())
    
    # Model Recommendation Matrix (top)
    fig.text(0.5, 0.86, "MODEL RECOMMENDATION BY COUNTRY", fontsize=10, 
            fontweight="bold", ha="center", color=C["orange"])
    
    ax_rec = fig.add_axes([ML, 0.69, CW, 0.09])
    ax_rec.axis("off")
    
    if not am.empty and not pm.empty:
        a_mape = am.set_index("Country")["MAPE_percent"]
        p_mape = pm.set_index("Country")["MAPE_percent"]
        
        rec_data = []
        for c in am["Country"].values:
            a, p = a_mape.get(c, 999), p_mape.get(c, 999)
            best = "ARIMA" if a <= p else "Prophet"
            volatility = "Low" if a < 30 else "Medium" if a < 60 else "High"
            rec_data.append([c, f"{a:.1f}%", f"{p:.1f}%", best, volatility])
        
        rec_table = ax_rec.table(cellText=rec_data,
                                colLabels=["Country", "ARIMA", "Prophet", "Recommend", "Risk"],
                                loc="center", cellLoc="center",
                                colWidths=[0.15, 0.20, 0.20, 0.22, 0.18])
        rec_table.auto_set_font_size(False)
        rec_table.set_fontsize(8)
        rec_table.scale(1.0, 1.2)
        
        for i in range(5):
            rec_table[(0, i)].set_facecolor(C["orange"])
            rec_table[(0, i)].set_text_props(fontweight="bold", color="white", ha="center")
        
        # Color code recommendations
        for ri in range(1, len(rec_data) + 1):
            if "ARIMA" in rec_data[ri-1][3]:
                rec_table[(ri, 3)].set_text_props(color=C["green"], fontweight="bold")
            risk = rec_data[ri-1][4]
            color = C["green"] if risk == "Low" else C["orange"] if risk == "Medium" else C["red"]
            rec_table[(ri, 4)].set_text_props(color=color, fontweight="bold")
    
    # Chart: MAPE Comparison (middle)
    fig.text(0.5, 0.58, "MAPE COMPARISON BY COUNTRY", fontsize=10, fontweight="bold",
            ha="center", color=C["purple"])
    
    ax1 = fig.add_axes([ML, 0.41, CW, 0.15])
    if not am.empty and not pm.empty and not nm.empty:
        countries = am["Country"].values
        x = np.arange(len(countries))
        w = 0.25
        
        a_mape = am.set_index("Country")["MAPE_percent"]
        p_mape = pm.set_index("Country")["MAPE_percent"]
        # For naive, get best (lowest MAPE) method per country
        n_mape = nm.groupby("Country")["MAPE_percent"].min()
        
        cap = 100
        ax1.bar(x - w, [min(a_mape.get(c, 0), cap) for c in countries], w, 
               label="ARIMA", color=C["green"], alpha=0.85)
        ax1.bar(x, [min(p_mape.get(c, 0), cap) for c in countries], w, 
               label="Prophet", color=C["purple"], alpha=0.85)
        ax1.bar(x + w, [min(n_mape.get(c, 0), cap) for c in countries], w, 
               label="Naive", color=C["blue"], alpha=0.85)
        
        ax1.axhline(y=25, color=C["green"], lw=1.5, ls="--", alpha=0.7)
        ax1.axhline(y=50, color=C["orange"], lw=1.5, ls="--", alpha=0.7)
        ax1.text(len(countries) - 0.3, 27, "Good", fontsize=7, color=C["green"])
        ax1.text(len(countries) - 0.3, 52, "Fair", fontsize=7, color=C["orange"])
        
        ax1.set_ylabel("MAPE (%)", fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(countries, fontsize=8)
        ax1.legend(loc="upper left", fontsize=8, ncol=3)
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.tick_params(labelsize=7)
        ax1.set_ylim(0, cap + 10)
    
    # Table Title (positioned clearly above table)
    fig.text(0.5, 0.35, "DETAILED MODEL PERFORMANCE METRICS", fontsize=10, fontweight="bold",
            ha="center", color=C["purple"])
    
    # Performance Table (bottom)
    ax2 = fig.add_axes([ML, 0.08, CW, 0.28])
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
                f"${r['RMSE']:,.0f}", 
                f"${r['MAE']:,.0f}", 
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
            loc="center", 
            cellLoc="center",
            colWidths=[0.10, 0.13, 0.13, 0.10, 0.10, 0.10, 0.10, 0.10]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
        
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
    print("  [OK] Page 3: Model Selection Rationale")


# --------------------------------------------------------------------------------------
# PAGE 4: ARIMA FORECASTS
# --------------------------------------------------------------------------------------

def page4(pdf, data):
    """Forecast Results & Spotlight: ID (Indonesia)."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C["white"])
    header(fig, "FORECAST RESULTS & SPOTLIGHT: ID (INDONESIA)", 
           "Best Country Highlight | 6-Month Robust Forecast | Global Summary", 4)
    
    w = data.get("weekly", pd.DataFrame())
    fc = data.get("arima_forecast", pd.DataFrame())
    cp = data.get("arima_comparison", pd.DataFrame())
    am = data.get("arima_metrics", pd.DataFrame())
    sa = data.get("structural_anomalies", pd.DataFrame())
    
    # ---------------------------------------------------------
    # TOP SECTION: ID SPOTLIGHT (The requested ID diagram)
    # ---------------------------------------------------------
    id_hist = w[w["Country_Name"] == "ID"].sort_values("Week_Ending_Date")
    id_valid = cp[cp["Country"] == "ID"].sort_values("Week_Ending_Date")
    id_fc6 = fc[(fc["Country"] == "ID") & (fc["Horizon"] == "6_month")].sort_values("Week_Ending_Date")
    id_anom = sa[sa["Country"] == "ID"].sort_values("Week_Ending_Date")
    
    ax_id = fig.add_axes([ML, 0.63, CW, 0.22])
    
    # Historical Actuals (Tail for visibility)
    hist_show = id_hist.tail(25)
    ax_id.plot(hist_show["Week_Ending_Date"], hist_show["Net_Cash_Flow"], 
               label="Historical Actuals", marker="o", ms=4, lw=1.5, color=C["blue"])
    
    # Validation forecast
    if not id_valid.empty:
        ax_id.plot(id_valid["Week_Ending_Date"], id_valid["Predicted_Cash_Flow"], 
                   label="Validation (ARIMA)", ls="--", color=C["purple"], lw=1.5, marker="s", ms=3)
        
    # Future Forecast (6-month)
    if not id_fc6.empty:
        ax_id.plot(id_fc6["Week_Ending_Date"], id_fc6["Predicted_Cash_Flow"], 
                   label="Future Forecast", marker="*", ms=5, color=C["green"], ls="-", lw=1.2)
        ax_id.fill_between(id_fc6["Week_Ending_Date"], 
                          id_fc6["Lower_CI"], 
                          id_fc6["Upper_CI"], 
                          color=C["green"], alpha=0.1, label="95% CI")

    # Volatility Alerts
    struct_id = id_anom[id_anom["Anomaly_Flag"] == -1]
    visible_dates = hist_show["Week_Ending_Date"].values
    visible_anoms = struct_id[struct_id["Week_Ending_Date"].isin(visible_dates)]
    if not visible_anoms.empty:
        ax_id.scatter(visible_anoms["Week_Ending_Date"], visible_anoms["Net_Cash_Flow"], 
                     color=C["red"], s=80, marker="X", zorder=10, 
                     label=f"Volatility Alert ({len(visible_anoms)})")
        
    ax_id.axhline(0, color=C["black"], lw=1, alpha=0.5)
    ax_id.set_title("ID (INDONESIA) - TREASURY FLOW PERFORMANCE SPOTLIGHT", fontsize=11, fontweight="bold", pad=10)
    ax_id.set_ylabel("Net Cash Flow (USD)", fontsize=9)
    ax_id.legend(loc="lower left", fontsize=8, ncol=2, frameon=True, facecolor="white", framealpha=0.9)
    ax_id.grid(True, alpha=0.15)
    ax_id.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(x, d=0)))
    plt.setp(ax_id.xaxis.get_majorticklabels(), rotation=15, ha="right", fontsize=8)

    # ---------------------------------------------------------
    # MIDDLE SECTION: RISK MATRIX (COMPACT)
    # ---------------------------------------------------------
    fig.text(0.5, 0.565, "FORECAST RELIABILITY & BUSINESS ACTIONS (ALL ENTITIES)", fontsize=9, 
            fontweight="bold", ha="center", color=C["gold"])
    
    ax_risk = fig.add_axes([ML, 0.41, CW, 0.08])
    ax_risk.axis("off")
    
    if not am.empty:
        risk_data = []
        for _, r in am.head(8).iterrows():  # Top 8 countries
            c, mape = r["Country"], r["MAPE_percent"]
            conf = "HIGH" if mape < 30 else "MEDIUM" if mape < 60 else "LOW"
            action = "Full Planning" if mape < 30 else "Buffer Required" if mape < 60 else "Ad-hoc Review"
            risk_data.append([c, f"{mape:.1f}%", conf, action, "Forecast OK" if mape < 60 else "Use Carefully"])
        
        risk_table = ax_risk.table(cellText=risk_data,
                                   colLabels=["Country", "MAPE", "Confidence", "Action Protocol", "Status"],
                                   loc="center", cellLoc="left",
                                   colWidths=[0.12, 0.12, 0.18, 0.30, 0.28])
        risk_table.auto_set_font_size(False); risk_table.set_fontsize(7.5); risk_table.scale(1.0, 1.1)
        for i in range(5):
            risk_table[(0, i)].set_facecolor(C["gold"]); risk_table[(0, i)].set_text_props(fontweight="bold", color="white", ha="center")

    # ---------------------------------------------------------
    # BOTTOM SECTION: GLOBAL SUMMARY TABLES
    # ---------------------------------------------------------
    hw = (CW - 0.04) / 2
    
    # Left Table: Forecast Summary
    fig.text(ML + hw/2, 0.315, "GLOBAL FORECAST PROJECTIONS", fontsize=9, fontweight="bold", ha="center", color=C["green"])
    ax_sum = fig.add_axes([ML, 0.08, hw, 0.22])
    ax_sum.axis("off")
    
    if not fc.empty:
        sum_data = []
        for c in fc["Country"].unique():
            f1 = fc[(fc["Country"] == c) & (fc["Horizon"] == "1_month")]["Predicted_Cash_Flow"].sum()
            f6 = fc[(fc["Country"] == c) & (fc["Horizon"] == "6_month")]["Predicted_Cash_Flow"].sum()
            avg = f6 / 26 if f6 != 0 else 0 # Avg weekly over 6m
            sum_data.append([c, fmt(f1), fmt(f6), fmt(avg)])
        
        st = ax_sum.table(cellText=sum_data, colLabels=["Entity", "1-Month", "6-Month", "Avg/Wk"], loc="center", cellLoc="center")
        st.auto_set_font_size(False); st.set_fontsize(7.5); st.scale(1.0, 1.1)
        for i in range(4):
            st[(0, i)].set_facecolor(C["green"]); st[(0, i)].set_text_props(fontweight="bold", color="white")

    # Right Table: Validation Accuracy
    fig.text(ML + hw + 0.04 + hw/2, 0.315, "VALIDATION TRACK RECORD", fontsize=9, fontweight="bold", ha="center", color=C["blue"])
    ax_val = fig.add_axes([ML + hw + 0.04, 0.08, hw, 0.22])
    ax_val.axis("off")
    
    if not cp.empty:
        val_data = []
        for c in cp["Country"].unique():
            actual = cp[cp["Country"] == c]["Actual_Cash_Flow"].sum()
            pred = cp[cp["Country"] == c]["Predicted_Cash_Flow"].sum()
            m_row = am[am["Country"] == c]
            mape = m_row["MAPE_percent"].values[0] if not m_row.empty else 100
            acc = max(0, 100 - mape)
            val_data.append([c, fmt(actual), fmt(pred), f"{acc:.1f}%"])
        
        vt = ax_val.table(cellText=val_data, colLabels=["Entity", "Actual", "Pred", "Accuracy"], loc="center", cellLoc="center")
        vt.auto_set_font_size(False); vt.set_fontsize(7.5); vt.scale(1.0, 1.1)
        for i in range(4):
            vt[(0, i)].set_facecolor(C["blue"]); vt[(0, i)].set_text_props(fontweight="bold", color="white")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 4: Forecast Results & Business Implications")


# --------------------------------------------------------------------------------------
# PAGE 5: ANOMALIES & RECOMMENDATIONS
# --------------------------------------------------------------------------------------

def page5(pdf, data):
    """Anomaly Detection & Korea Deep Dive."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C["white"])
    header(fig, "ANOMALY DETECTION & KOREA DEEP DIVE", "High-Volatility Focus | Manual Review Required | Anomaly-Based Analysis", 5)
    
    sa = data.get("structural_anomalies", pd.DataFrame())
    ta = data.get("transaction_anomalies", pd.DataFrame())
    am = data.get("arima_metrics", pd.DataFrame())
    w = data.get("weekly", pd.DataFrame())
    
    hw = (CW - 0.04) / 2
    n_struct = (sa["Anomaly_Flag"] == -1).sum() if not sa.empty else 0
    n_txn = len(ta) if not ta.empty else 0
    
    # ---------------------------------------------------------
    # TOP SECTION: Condensed Anomaly Methodology
    # ---------------------------------------------------------
    fig.text(0.5, 0.86, "ANOMALY DETECTION METHODOLOGY", fontsize=10, fontweight="bold", ha="center", color=C["red"])
    
    ax_method = fig.add_axes([ML, 0.75, CW, 0.08])
    ax_method.axis("off")
    
    method_data = [
        ["Transaction-Level", "Z-Score (|Z| > 3.5)", f"{n_txn:,} flagged", "Micro anomalies"],
        ["Structural-Level", "Isolation Forest (5%)", f"{n_struct} flagged", "Macro patterns"]
    ]
    
    method_table = ax_method.table(cellText=method_data,
                                   colLabels=["Scope", "Method", "Detected", "Purpose"],
                                   loc="center", cellLoc="center",
                                   colWidths=[0.22, 0.28, 0.20, 0.30])
    method_table.auto_set_font_size(False)
    method_table.set_fontsize(8)
    method_table.scale(1.0, 1.4)
    
    for i in range(4):
        method_table[(0, i)].set_facecolor(C["red"])
        method_table[(0, i)].set_text_props(fontweight="bold", color="white", ha="center")
    
    # ---------------------------------------------------------
    # MIDDLE SECTION: KOREA DEEP DIVE
    # ---------------------------------------------------------
    fig.text(0.5, 0.70, "KOREA (KR) - HIGH VOLATILITY ENTITY REQUIRING MANUAL REVIEW", 
            fontsize=11, fontweight="bold", ha="center", color=C["orange"])
    
    # Korea Cash Flow Timeline (left)
    ax_kr = fig.add_axes([ML, 0.40, hw, 0.26])
    
    if not w.empty and not sa.empty:
        kr_weekly = w[w["Country_Name"] == "KR"].sort_values("Week_Ending_Date")
        kr_anom = sa[(sa["Country"] == "KR") & (sa["Anomaly_Flag"] == -1)].sort_values("Week_Ending_Date")
        
        ax_kr.plot(kr_weekly["Week_Ending_Date"], kr_weekly["Net_Cash_Flow"], 
                   "o-", color=C["blue"], lw=1.5, ms=4, alpha=0.7, label="Net Cash Flow")
        
        if not kr_anom.empty:
            ax_kr.scatter(kr_anom["Week_Ending_Date"], kr_anom["Net_Cash_Flow"], 
                         color=C["red"], s=100, marker="X", zorder=10, 
                         label=f"Anomaly ({len(kr_anom)})")
            
            # Annotate the most severe anomaly
            worst = kr_anom.loc[kr_anom["Anomaly_Score"].idxmin()]
            ax_kr.annotate(f"Severity: {worst['Anomaly_Score']:.3f}\n${worst['Net_Cash_Flow']:,.0f}",
                          xy=(worst["Week_Ending_Date"], worst["Net_Cash_Flow"]),
                          xytext=(10, 30), textcoords="offset points",
                          fontsize=7, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                          arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.5))
        
        ax_kr.axhline(0, color=C["gray"], lw=1, ls="--", alpha=0.5)
        ax_kr.set_title("Korea Cash Flow Timeline with Anomalies", fontsize=9, fontweight="bold", pad=5)
        ax_kr.set_ylabel("Net Cash Flow (USD)", fontsize=8)
        ax_kr.legend(loc="lower left", fontsize=7)
        ax_kr.grid(True, alpha=0.3)
        ax_kr.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(x, 0)))
        ax_kr.tick_params(labelsize=7)
        plt.setp(ax_kr.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=7)
    
    # Korea Stats Table (right)
    ax_kr_stats = fig.add_axes([ML + hw + 0.04, 0.40, hw, 0.26])
    ax_kr_stats.axis("off")
    
    # Calculate Korea stats
    kr_mape = am[am["Country"] == "KR"]["MAPE_percent"].values[0] if not am.empty else 0
    kr_anom_count = len(sa[(sa["Country"] == "KR") & (sa["Anomaly_Flag"] == -1)]) if not sa.empty else 0
    kr_cf_std = w[w["Country_Name"] == "KR"]["Net_Cash_Flow"].std() if not w.empty else 0
    kr_cf_mean = w[w["Country_Name"] == "KR"]["Net_Cash_Flow"].mean() if not w.empty else 0
    
    kr_stats = [
        ["MAPE (Accuracy)", f"{kr_mape:.1f}%", "UNUSABLE", "Model forecasting fails"],
        ["Struct. Anomalies", f"{kr_anom_count} weeks", "HIGH", "21% weeks anomalous"],
        ["CF Volatility", fmt(kr_cf_std), "HIGH", "Unpredictable swings"],
        ["Avg Weekly CF", fmt(kr_cf_mean), "NEGATIVE", "Consistent outflows"],
        ["Recommendation", "MANUAL REVIEW", "CRITICAL", "No model forecasts"],
        ["Action Required", "Weekly monitor", "IMMEDIATE", "Manual oversight"],
        ["Root Cause", "Investigate", "ANALYSIS", "Find volatility drivers"],
        ["Alternative", "Scenario plan", "STRATEGIC", "Range-based planning"]
    ]
    
    kr_table = ax_kr_stats.table(cellText=kr_stats,
                                 colLabels=["Metric", "Value", "Status", "Interpretation"],
                                 loc="center", cellLoc="left",
                                 colWidths=[0.26, 0.20, 0.18, 0.36])
    kr_table.auto_set_font_size(False)
    kr_table.set_fontsize(7)
    kr_table.scale(1.0, 1.3)
    
    for i in range(4):
        kr_table[(0, i)].set_facecolor(C["orange"])
        kr_table[(0, i)].set_text_props(fontweight="bold", color="white", ha="center")
    
    # Color code status column
    status_colors = [C["red"], C["orange"], C["orange"], C["blue"], C["red"], C["red"], C["purple"], C["green"]]
    for ri in range(1, 9):
        kr_table[(ri, 2)].set_text_props(color=status_colors[ri-1], fontweight="bold")
    
    # ---------------------------------------------------------
    # BOTTOM SECTION: Action Plan (condensed)
    # ---------------------------------------------------------
    fig.text(ML + hw/2, 0.30, "PRIORITIZED ACTIONS", fontsize=9, fontweight="bold", ha="center", color=C["blue"])
    
    ax_action = fig.add_axes([ML, 0.06, hw, 0.24])
    ax_action.axis("off")
    
    action_data = [
        ["CRITICAL", "Do NOT use KR forecasts", "Immediate"],
        ["CRITICAL", "Deploy forecasts: MY/PH/TH", "Immediate"],
        ["HIGH", "Weekly KR manual review", "Ongoing"],
        ["HIGH", "Set alerts: |Z| > 5", "Week 1"],
        ["MEDIUM", "Investigate KR volatility", "Month 1"],
        ["MEDIUM", "Build scenario models", "Month 2"],
        ["LOW", "Add FX regressors", "Month 3+"]
    ]
    
    action_table = ax_action.table(cellText=action_data,
                                   colLabels=["Priority", "Action", "Timeline"],
                                   loc="center", cellLoc="left",
                                   colWidths=[0.22, 0.50, 0.28])
    action_table.auto_set_font_size(False)
    action_table.set_fontsize(7.5)
    action_table.scale(1.0, 1.3)
    
    for i in range(3):
        action_table[(0, i)].set_facecolor(C["blue"])
        action_table[(0, i)].set_text_props(fontweight="bold", color="white", ha="center")
    
    for ri in range(1, len(action_data) + 1):
        priority = action_data[ri-1][0]
        color = C["red"] if priority == "CRITICAL" else C["orange"] if priority == "HIGH" else C["gold"] if priority == "MEDIUM" else C["green"]
        action_table[(ri, 0)].set_text_props(color=color, fontweight="bold")
    
    # Key Insights (bottom right)
    fig.text(ML + hw + 0.04 + hw/2, 0.30, "KEY INSIGHTS", fontsize=9, fontweight="bold", ha="center", color=C["purple"])
    
    ax_insights = fig.add_axes([ML + hw + 0.04, 0.06, hw, 0.24])
    ax_insights.axis("off")
    
    best_c = am.loc[am["MAPE_percent"].idxmin(), "Country"] if not am.empty else "N/A"
    best_mape = am.loc[am["MAPE_percent"].idxmin(), "MAPE_percent"] if not am.empty else 0
    
    insights_data = [
        ["Best Model", f"{best_c} ({best_mape:.1f}%)", "High confidence"],
        ["Worst Model", f"KR ({kr_mape:.1f}%)", "Do not use"],
        ["Risk: High", "KR, SS", "Manual review only"],
        ["Risk: Medium", "TW, VN", "Use with caution"],
        ["Risk: Low", "MY, PH, TH, ID", "Full automation OK"],
        ["Data Quality", "44 weeks, 84K txns", "Complete coverage"],
        ["Anomaly Rate", f"{n_struct}/352 weeks", f"{n_struct/3.52:.1f}% flagged"]
    ]
    
    insights_table = ax_insights.table(cellText=insights_data,
                                       colLabels=["Category", "Finding", "Status"],
                                       loc="center", cellLoc="left",
                                       colWidths=[0.28, 0.38, 0.34])
    insights_table.auto_set_font_size(False)
    insights_table.set_fontsize(7.5)
    insights_table.scale(1.0, 1.3)
    
    for i in range(3):
        insights_table[(0, i)].set_facecolor(C["purple"])
        insights_table[(0, i)].set_text_props(fontweight="bold", color="white", ha="center")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 5: Anomaly Detection & Korea Deep Dive")


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