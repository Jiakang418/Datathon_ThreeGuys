"""
Professional PDF Dashboard Generator
=====================================
AstraZeneca Cash Flow Challenge - Visual Storyboard

Creates a 5-page professional PDF dashboard with:
- Page 1: Executive Summary & KPIs
- Page 2: Cash Flow Trends & Category Analysis
- Page 3: Prophet Forecast Results
- Page 4: Model Validation & Accuracy
- Page 5: Insights & Recommendations

Author: Marcus (Data Engineer)
Date: 2025-12-19
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

# AstraZeneca Brand Colors
AZ_COLORS = {
    "purple": "#830051",       # Primary brand
    "dark_purple": "#5C0039",  # Darker shade
    "blue": "#00A0DF",         # Secondary
    "light_blue": "#68D2DF",   # Accent
    "green": "#00843D",        # Positive/Inflow
    "red": "#E40046",          # Negative/Outflow
    "orange": "#FF6B35",       # Warning
    "gray": "#6E6E6E",         # Neutral
    "light_gray": "#E8E8E8",   # Background
    "white": "#FFFFFF",
    "black": "#1A1A1A",
}

# Country colors for consistency
COUNTRY_COLORS = {
    "ID": "#830051",
    "KR": "#00A0DF", 
    "MY": "#68D2DF",
    "PH": "#00843D",
    "SS": "#E40046",
    "TH": "#FF6B35",
    "TW": "#6E6E6E",
    "VN": "#5C0039",
}


# --------------------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------------------

def load_all_data():
    """Load all required datasets for the dashboard."""
    data = {}
    
    # Weekly cash flow data
    weekly_file = DATA_DIR / "processed_weekly_cashflow.csv"
    if weekly_file.exists():
        data["weekly"] = pd.read_csv(weekly_file, parse_dates=["Week_Ending_Date"])
    
    # Prophet results
    prophet_forecast = MARCUS_DIR / "prophet_results" / "prophet_future_forecasts.csv"
    prophet_metrics = MARCUS_DIR / "prophet_results" / "prophet_backtest_metrics.csv"
    prophet_comparison = MARCUS_DIR / "prophet_results" / "prophet_actual_vs_predicted.csv"
    
    if prophet_forecast.exists():
        data["prophet_forecast"] = pd.read_csv(prophet_forecast, parse_dates=["Week_Ending_Date"])
    if prophet_metrics.exists():
        data["prophet_metrics"] = pd.read_csv(prophet_metrics)
    if prophet_comparison.exists():
        data["prophet_comparison"] = pd.read_csv(prophet_comparison, parse_dates=["Week_Ending_Date"])
    
    # Naive results for comparison
    naive_metrics = MARCUS_DIR / "naive_results" / "naive_backtest_metrics.csv"
    if naive_metrics.exists():
        data["naive_metrics"] = pd.read_csv(naive_metrics)
    
    # Cash balance data
    balance_file = RAW_DIR / "Datathon Dataset - Data - Cash Balance.csv"
    if balance_file.exists():
        data["cash_balance"] = pd.read_csv(balance_file)
    
    # Raw transaction data for category analysis
    main_file = RAW_DIR / "Datathon Dataset - Data - Main.csv"
    if main_file.exists():
        data["transactions"] = pd.read_csv(main_file, low_memory=False)
    
    return data


# --------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------------

def format_currency(value, decimals=1):
    """Format number as currency string."""
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.{decimals}f}K"
    else:
        return f"${value:.0f}"


def add_header(ax, title, subtitle=None):
    """Add a styled header to a subplot."""
    ax.set_facecolor(AZ_COLORS["white"])
    ax.text(0.5, 0.7, title, transform=ax.transAxes, fontsize=16, 
            fontweight="bold", color=AZ_COLORS["purple"], ha="center", va="center")
    if subtitle:
        ax.text(0.5, 0.3, subtitle, transform=ax.transAxes, fontsize=10,
                color=AZ_COLORS["gray"], ha="center", va="center")
    ax.axis("off")


def create_kpi_card(ax, title, value, subtitle=None, color=None, icon_text=None):
    """Create a KPI card visualization."""
    if color is None:
        color = AZ_COLORS["purple"]
    
    ax.set_facecolor(AZ_COLORS["light_gray"])
    
    # Add colored left border effect
    rect = mpatches.Rectangle((0, 0), 0.02, 1, transform=ax.transAxes, 
                               color=color, clip_on=False)
    ax.add_patch(rect)
    
    # Title
    ax.text(0.5, 0.85, title, transform=ax.transAxes, fontsize=9,
            color=AZ_COLORS["gray"], ha="center", va="center", fontweight="medium")
    
    # Value
    ax.text(0.5, 0.5, value, transform=ax.transAxes, fontsize=20,
            color=color, ha="center", va="center", fontweight="bold")
    
    # Subtitle
    if subtitle:
        ax.text(0.5, 0.15, subtitle, transform=ax.transAxes, fontsize=8,
                color=AZ_COLORS["gray"], ha="center", va="center")
    
    ax.axis("off")
    
    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AZ_COLORS["light_gray"])


# --------------------------------------------------------------------------------------
# PAGE 1: EXECUTIVE SUMMARY
# --------------------------------------------------------------------------------------

def create_page1_executive_summary(pdf, data):
    """Create Page 1: Executive Summary with KPIs."""
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape - consistent size
    fig.patch.set_facecolor(AZ_COLORS["white"])
    
    # Title header
    fig.text(0.5, 0.95, "ASTRAZENECA CASH FLOW DASHBOARD", fontsize=22, 
             fontweight="bold", color=AZ_COLORS["purple"], ha="center")
    fig.text(0.5, 0.91, "Weekly Cash Flow Analysis & Forecasting | Prophet Model Baseline", 
             fontsize=11, color=AZ_COLORS["gray"], ha="center")
    fig.text(0.5, 0.88, f"Report Generated: {datetime.now().strftime('%B %d, %Y')}", 
             fontsize=9, color=AZ_COLORS["gray"], ha="center")
    
    # Divider line
    line = plt.Line2D([0.05, 0.95], [0.85, 0.85], transform=fig.transFigure, 
                      color=AZ_COLORS["purple"], linewidth=2)
    fig.add_artist(line)
    
    # Calculate KPIs from data
    weekly_df = data.get("weekly", pd.DataFrame())
    
    if not weekly_df.empty:
        total_inflow = weekly_df[weekly_df["Net_Cash_Flow"] > 0]["Net_Cash_Flow"].sum()
        total_outflow = abs(weekly_df[weekly_df["Net_Cash_Flow"] < 0]["Net_Cash_Flow"].sum())
        net_cash = weekly_df["Net_Cash_Flow"].sum()
        num_countries = weekly_df["Country_Name"].nunique()
        num_weeks = weekly_df["Week_Ending_Date"].nunique()
        avg_weekly = weekly_df.groupby("Week_Ending_Date")["Net_Cash_Flow"].sum().mean()
    else:
        total_inflow = total_outflow = net_cash = avg_weekly = 0
        num_countries = num_weeks = 0
    
    # KPI Cards Row 1
    kpi_positions = [
        (0.08, 0.65, 0.18, 0.15),
        (0.30, 0.65, 0.18, 0.15),
        (0.52, 0.65, 0.18, 0.15),
        (0.74, 0.65, 0.18, 0.15),
    ]
    
    kpis = [
        ("COUNTRIES", str(num_countries), "Active Entities", AZ_COLORS["blue"]),
        ("DATA PERIOD", f"{num_weeks} Weeks", "Jan - Nov 2025", AZ_COLORS["purple"]),
        ("NET CASH FLOW", format_currency(net_cash), "Total USD", AZ_COLORS["red"] if net_cash < 0 else AZ_COLORS["green"]),
        ("AVG WEEKLY", format_currency(avg_weekly), "Per Week", AZ_COLORS["gray"]),
    ]
    
    for pos, (title, value, subtitle, color) in zip(kpi_positions, kpis):
        ax = fig.add_axes(pos)
        create_kpi_card(ax, title, value, subtitle, color)
    
    # Main chart: Net Cash Flow Trend
    ax_main = fig.add_axes([0.08, 0.12, 0.55, 0.45])
    
    if not weekly_df.empty:
        weekly_totals = weekly_df.groupby("Week_Ending_Date")["Net_Cash_Flow"].sum().reset_index()
        weekly_totals = weekly_totals.sort_values("Week_Ending_Date")
        
        colors = [AZ_COLORS["green"] if v >= 0 else AZ_COLORS["red"] 
                  for v in weekly_totals["Net_Cash_Flow"]]
        
        ax_main.bar(range(len(weekly_totals)), weekly_totals["Net_Cash_Flow"], 
                   color=colors, alpha=0.8, width=0.8)
        ax_main.axhline(y=0, color=AZ_COLORS["gray"], linewidth=1, linestyle="-")
        
        # Trend line
        z = np.polyfit(range(len(weekly_totals)), weekly_totals["Net_Cash_Flow"], 1)
        p = np.poly1d(z)
        ax_main.plot(range(len(weekly_totals)), p(range(len(weekly_totals))), 
                    color=AZ_COLORS["purple"], linewidth=2, linestyle="--", label="Trend")
        
        ax_main.set_xlabel("Week", fontsize=10, color=AZ_COLORS["gray"])
        ax_main.set_ylabel("Net Cash Flow (USD)", fontsize=10, color=AZ_COLORS["gray"])
        ax_main.set_title("Weekly Net Cash Flow Trend (All Countries)", fontsize=12, 
                         fontweight="bold", color=AZ_COLORS["black"], pad=10)
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x, 0)))
        ax_main.grid(True, alpha=0.3, axis="y")
        ax_main.legend(loc="upper right", fontsize=8)
    
    # Side panel: Cash Flow by Country
    ax_side = fig.add_axes([0.68, 0.12, 0.27, 0.45])
    
    if not weekly_df.empty:
        country_totals = weekly_df.groupby("Country_Name")["Net_Cash_Flow"].sum().sort_values()
        colors = [COUNTRY_COLORS.get(c, AZ_COLORS["gray"]) for c in country_totals.index]
        
        bars = ax_side.barh(country_totals.index, country_totals.values, color=colors, alpha=0.85)
        ax_side.axvline(x=0, color=AZ_COLORS["gray"], linewidth=1)
        
        ax_side.set_title("Net Cash Flow by Country", fontsize=11, fontweight="bold", 
                         color=AZ_COLORS["black"], pad=10)
        ax_side.set_xlabel("USD", fontsize=9, color=AZ_COLORS["gray"])
        ax_side.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x, 0)))
        ax_side.grid(True, alpha=0.3, axis="x")
    
    # Footer
    fig.text(0.05, 0.02, "Data Source: AstraZeneca Datathon Dataset", 
             fontsize=8, color=AZ_COLORS["gray"])
    fig.text(0.95, 0.02, "Page 1 of 5", fontsize=8, color=AZ_COLORS["gray"], ha="right")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 1: Executive Summary")


# --------------------------------------------------------------------------------------
# PAGE 2: CASH FLOW TRENDS & CATEGORY ANALYSIS
# --------------------------------------------------------------------------------------

def create_page2_trends_analysis(pdf, data):
    """Create Page 2: Cash Flow Trends & Category Analysis."""
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.patch.set_facecolor(AZ_COLORS["white"])
    
    # Header
    fig.text(0.5, 0.95, "CASH FLOW TRENDS & CATEGORY ANALYSIS", fontsize=18,
             fontweight="bold", color=AZ_COLORS["purple"], ha="center")
    fig.text(0.5, 0.91, "Historical patterns and breakdown by operating categories",
             fontsize=10, color=AZ_COLORS["gray"], ha="center")
    
    line = plt.Line2D([0.05, 0.95], [0.88, 0.88], transform=fig.transFigure,
                      color=AZ_COLORS["purple"], linewidth=2)
    fig.add_artist(line)
    
    weekly_df = data.get("weekly", pd.DataFrame())
    transactions_df = data.get("transactions", pd.DataFrame())
    
    # Chart 1: Time series by country (top left) - REDUCED SIZE
    ax1 = fig.add_axes([0.08, 0.56, 0.40, 0.28])
    
    if not weekly_df.empty:
        for country in weekly_df["Country_Name"].unique():
            country_data = weekly_df[weekly_df["Country_Name"] == country].sort_values("Week_Ending_Date")
            ax1.plot(country_data["Week_Ending_Date"], country_data["Net_Cash_Flow"],
                    label=country, color=COUNTRY_COLORS.get(country, AZ_COLORS["gray"]),
                    linewidth=1.2, alpha=0.8)
        
        ax1.axhline(y=0, color=AZ_COLORS["gray"], linewidth=0.8, linestyle="--")
        ax1.set_title("Weekly Cash Flow by Country", fontsize=10, fontweight="bold", pad=5)
        ax1.set_xlabel("Date", fontsize=8)
        ax1.set_ylabel("Net Cash Flow (USD)", fontsize=8)
        ax1.legend(loc="upper right", fontsize=6, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x, 0)))
        ax1.tick_params(axis='both', labelsize=7)
        # Reduce number of x-axis labels
        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=6)
    
    # Chart 2: Operating vs Financing (top right) - REDUCED SIZE
    ax2 = fig.add_axes([0.55, 0.56, 0.40, 0.28])
    
    if not weekly_df.empty:
        weekly_totals = weekly_df.groupby("Week_Ending_Date").agg({
            "Operating_Cash_Flow": "sum",
            "Financing_Cash_Flow": "sum"
        }).reset_index().sort_values("Week_Ending_Date")
        
        ax2.fill_between(range(len(weekly_totals)), weekly_totals["Operating_Cash_Flow"],
                        alpha=0.6, color=AZ_COLORS["blue"], label="Operating")
        ax2.fill_between(range(len(weekly_totals)), weekly_totals["Financing_Cash_Flow"],
                        alpha=0.6, color=AZ_COLORS["orange"], label="Financing")
        ax2.axhline(y=0, color=AZ_COLORS["gray"], linewidth=0.8)
        
        ax2.set_title("Operating vs Financing Cash Flow", fontsize=10, fontweight="bold", pad=5)
        ax2.set_xlabel("Week", fontsize=8)
        ax2.set_ylabel("USD", fontsize=8)
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=7)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x, 0)))
    
    # Chart 3: Category breakdown (bottom left) - Donut chart - ADJUSTED
    ax3 = fig.add_axes([0.12, 0.08, 0.32, 0.38])
    
    if not transactions_df.empty and "Category" in transactions_df.columns:
        # Ensure Amount in USD is numeric
        transactions_df["Amount in USD"] = pd.to_numeric(transactions_df["Amount in USD"], errors="coerce").fillna(0)
        category_totals = transactions_df.groupby("Category")["Amount in USD"].sum().abs()
        category_totals = category_totals.sort_values(ascending=False).head(8)
        
        colors = [AZ_COLORS["purple"], AZ_COLORS["blue"], AZ_COLORS["light_blue"],
                 AZ_COLORS["green"], AZ_COLORS["red"], AZ_COLORS["orange"],
                 AZ_COLORS["gray"], "#9B59B6"]
        
        wedges, texts, autotexts = ax3.pie(category_totals.values, labels=None,
                                           autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
                                           colors=colors[:len(category_totals)],
                                           wedgeprops=dict(width=0.5, edgecolor='white'),
                                           pctdistance=0.78,
                                           textprops={'fontsize': 7})
        
        ax3.set_title("Cash Flow by Category (Top 8)", fontsize=10, fontweight="bold", pad=5)
        
        # Legend - moved to the left side
        ax3.legend(wedges, [f"{cat[:12]}..." if len(cat) > 12 else cat 
                           for cat in category_totals.index],
                  loc="center left", bbox_to_anchor=(-0.45, 0.5), fontsize=6)
    
    # Chart 4: Monthly pattern (bottom right) - ADJUSTED
    ax4 = fig.add_axes([0.55, 0.08, 0.40, 0.38])
    
    if not weekly_df.empty:
        weekly_df["Month"] = weekly_df["Week_Ending_Date"].dt.month_name().str[:3]
        monthly = weekly_df.groupby("Month")["Net_Cash_Flow"].sum()
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
        monthly = monthly.reindex([m for m in month_order if m in monthly.index])
        
        colors = [AZ_COLORS["green"] if v >= 0 else AZ_COLORS["red"] for v in monthly.values]
        ax4.bar(monthly.index, monthly.values, color=colors, alpha=0.8, edgecolor="white")
        ax4.axhline(y=0, color=AZ_COLORS["gray"], linewidth=0.8)
        
        ax4.set_title("Monthly Net Cash Flow Pattern", fontsize=10, fontweight="bold", pad=5)
        ax4.set_xlabel("Month", fontsize=8)
        ax4.set_ylabel("USD", fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")
        ax4.tick_params(axis='both', labelsize=7)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x, 0)))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    
    # Footer
    fig.text(0.05, 0.02, "Analysis Period: January - November 2025",
             fontsize=8, color=AZ_COLORS["gray"])
    fig.text(0.95, 0.02, "Page 2 of 5", fontsize=8, color=AZ_COLORS["gray"], ha="right")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 2: Trends & Category Analysis")


# --------------------------------------------------------------------------------------
# PAGE 3: PROPHET FORECAST RESULTS
# --------------------------------------------------------------------------------------

def create_page3_prophet_forecast(pdf, data):
    """Create Page 3: Prophet Forecast Results."""
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.patch.set_facecolor(AZ_COLORS["white"])
    
    # Header
    fig.text(0.5, 0.95, "PROPHET FORECAST RESULTS", fontsize=18,
             fontweight="bold", color=AZ_COLORS["purple"], ha="center")
    fig.text(0.5, 0.91, "1-Week, 1-Month, and 6-Month forecasts using Prophet time series model",
             fontsize=10, color=AZ_COLORS["gray"], ha="center")
    
    line = plt.Line2D([0.05, 0.95], [0.88, 0.88], transform=fig.transFigure,
                      color=AZ_COLORS["purple"], linewidth=2)
    fig.add_artist(line)
    
    prophet_forecast = data.get("prophet_forecast", pd.DataFrame())
    weekly_df = data.get("weekly", pd.DataFrame())
    
    # Chart 1: 1-Month Forecast (top) - REDUCED SIZE
    ax1 = fig.add_axes([0.08, 0.58, 0.87, 0.26])
    
    if not prophet_forecast.empty:
        forecast_1m = prophet_forecast[prophet_forecast["Horizon"] == "1_month"]
        
        for country in forecast_1m["Country"].unique():
            country_data = forecast_1m[forecast_1m["Country"] == country].sort_values("Week_Ending_Date")
            
            ax1.plot(country_data["Week_Ending_Date"], country_data["Predicted_Cash_Flow"],
                    marker="o", markersize=4, linewidth=1.5, label=country,
                    color=COUNTRY_COLORS.get(country, AZ_COLORS["gray"]))
            
            # Confidence interval
            ax1.fill_between(country_data["Week_Ending_Date"],
                           country_data["Prediction_Lower_95CI"],
                           country_data["Prediction_Upper_95CI"],
                           alpha=0.15, color=COUNTRY_COLORS.get(country, AZ_COLORS["gray"]))
        
        ax1.axhline(y=0, color=AZ_COLORS["gray"], linewidth=0.8, linestyle="--")
        ax1.set_title("1-Month Forecast (4 Weeks Ahead) with 95% Confidence Interval", 
                     fontsize=10, fontweight="bold", pad=5)
        ax1.set_xlabel("Week Ending Date", fontsize=8)
        ax1.set_ylabel("Predicted Cash Flow (USD)", fontsize=8)
        ax1.legend(loc="upper right", fontsize=6, ncol=4)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=7)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x, 0)))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    
    # Chart 2: 6-Month Forecast Summary (bottom left) - REDUCED SIZE
    ax2 = fig.add_axes([0.08, 0.10, 0.52, 0.38])
    
    if not prophet_forecast.empty:
        forecast_6m = prophet_forecast[prophet_forecast["Horizon"] == "6_month"]
        
        for country in forecast_6m["Country"].unique():
            country_data = forecast_6m[forecast_6m["Country"] == country].sort_values("Week_Ending_Date")
            ax2.plot(country_data["Week_Ending_Date"], country_data["Predicted_Cash_Flow"],
                    linewidth=1.2, label=country, alpha=0.8,
                    color=COUNTRY_COLORS.get(country, AZ_COLORS["gray"]))
        
        ax2.axhline(y=0, color=AZ_COLORS["gray"], linewidth=0.8, linestyle="--")
        ax2.set_title("6-Month Forecast (26 Weeks Ahead)", fontsize=10, fontweight="bold", pad=5)
        ax2.set_xlabel("Week", fontsize=8)
        ax2.set_ylabel("Predicted USD", fontsize=8)
        ax2.legend(loc="lower left", fontsize=6, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=7)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x, 0)))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(8))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=6)
    
    # Summary table (bottom right) - ADJUSTED POSITION
    ax3 = fig.add_axes([0.65, 0.10, 0.30, 0.38])
    ax3.axis("off")
    
    if not prophet_forecast.empty:
        # Calculate summary statistics
        summary_data = []
        for country in prophet_forecast["Country"].unique():
            cf_1m = prophet_forecast[(prophet_forecast["Country"] == country) & 
                                     (prophet_forecast["Horizon"] == "1_month")]["Predicted_Cash_Flow"].sum()
            cf_6m = prophet_forecast[(prophet_forecast["Country"] == country) & 
                                     (prophet_forecast["Horizon"] == "6_month")]["Predicted_Cash_Flow"].sum()
            summary_data.append([country, format_currency(cf_1m), format_currency(cf_6m)])
        
        ax3.text(0.5, 0.98, "Forecast Summary", fontsize=10, fontweight="bold",
                transform=ax3.transAxes, ha="center", color=AZ_COLORS["purple"])
        
        # Table header
        col_labels = ["Country", "1-Month", "6-Month"]
        table = ax3.table(cellText=summary_data, colLabels=col_labels,
                         loc="upper center", cellLoc="center",
                         colColours=[AZ_COLORS["light_gray"]]*3)
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.4)
        
        # Style header
        for i in range(3):
            table[(0, i)].set_text_props(fontweight="bold", color=AZ_COLORS["purple"])
    
    # Footer
    fig.text(0.05, 0.02, "Model: Facebook Prophet | Seasonality: Monthly (30.5 days)",
             fontsize=8, color=AZ_COLORS["gray"])
    fig.text(0.95, 0.02, "Page 3 of 5", fontsize=8, color=AZ_COLORS["gray"], ha="right")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 3: Prophet Forecast Results")


# --------------------------------------------------------------------------------------
# PAGE 4: MODEL VALIDATION & ACCURACY
# --------------------------------------------------------------------------------------

def create_page4_model_validation(pdf, data):
    """Create Page 4: Model Validation & Accuracy."""
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.patch.set_facecolor(AZ_COLORS["white"])
    
    # Header
    fig.text(0.5, 0.95, "MODEL VALIDATION & PERFORMANCE", fontsize=18,
             fontweight="bold", color=AZ_COLORS["purple"], ha="center")
    fig.text(0.5, 0.91, "Backtesting results comparing actual vs predicted cash flows",
             fontsize=10, color=AZ_COLORS["gray"], ha="center")
    
    line = plt.Line2D([0.05, 0.95], [0.88, 0.88], transform=fig.transFigure,
                      color=AZ_COLORS["purple"], linewidth=2)
    fig.add_artist(line)
    
    prophet_metrics = data.get("prophet_metrics", pd.DataFrame())
    prophet_comparison = data.get("prophet_comparison", pd.DataFrame())
    naive_metrics = data.get("naive_metrics", pd.DataFrame())
    
    # Chart 1: Actual vs Predicted (top) - REDUCED SIZE
    ax1 = fig.add_axes([0.08, 0.56, 0.87, 0.28])
    
    if not prophet_comparison.empty:
        countries = prophet_comparison["Country"].unique()
        x_positions = []
        x_labels = []
        
        for i, country in enumerate(countries):
            country_data = prophet_comparison[prophet_comparison["Country"] == country].sort_values("Week_Ending_Date")
            n_points = len(country_data)
            positions = np.arange(i * (n_points + 1), i * (n_points + 1) + n_points)
            
            ax1.bar(positions - 0.2, country_data["Actual_Cash_Flow"], 0.35,
                   label="Actual" if i == 0 else "", color=AZ_COLORS["blue"], alpha=0.8)
            ax1.bar(positions + 0.2, country_data["Predicted_Cash_Flow"], 0.35,
                   label="Predicted" if i == 0 else "", color=AZ_COLORS["purple"], alpha=0.8)
            
            x_positions.extend(positions)
            x_labels.extend([f"{country}\nW{j+1}" for j in range(n_points)])
        
        ax1.axhline(y=0, color=AZ_COLORS["gray"], linewidth=0.8)
        ax1.set_title("Actual vs Prophet Predicted (4-Week Validation Period)", 
                     fontsize=10, fontweight="bold", pad=5)
        ax1.set_ylabel("Cash Flow (USD)", fontsize=8)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(x_labels, fontsize=6)
        ax1.legend(loc="upper right", fontsize=7)
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.tick_params(axis='y', labelsize=7)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x, 0)))
    
    # Chart 2: MAPE by Country (bottom left) - REDUCED SIZE
    ax2 = fig.add_axes([0.08, 0.10, 0.40, 0.36])
    
    if not prophet_metrics.empty:
        countries = prophet_metrics["Country"].values
        mape_values = prophet_metrics["MAPE_percent"].values
        
        colors = []
        for mape in mape_values:
            if mape < 25:
                colors.append(AZ_COLORS["green"])
            elif mape < 50:
                colors.append(AZ_COLORS["orange"])
            else:
                colors.append(AZ_COLORS["red"])
        
        bars = ax2.bar(countries, mape_values, color=colors, alpha=0.85, edgecolor="white")
        
        # Add value labels
        for bar, val in zip(bars, mape_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
        
        ax2.axhline(y=25, color=AZ_COLORS["green"], linewidth=1.5, linestyle="--", label="Good (<25%)")
        ax2.axhline(y=50, color=AZ_COLORS["orange"], linewidth=1.5, linestyle="--", label="Fair (<50%)")
        
        ax2.set_title("Prophet MAPE by Country", fontsize=10, fontweight="bold", pad=5)
        ax2.set_ylabel("MAPE (%)", fontsize=8)
        ax2.set_xlabel("Country", fontsize=8)
        ax2.tick_params(axis='both', labelsize=7)
        ax2.legend(loc="upper right", fontsize=6)
        ax2.grid(True, alpha=0.3, axis="y")
    
    # Chart 3: Model Comparison - Prophet vs Naive (bottom right) - REDUCED SIZE
    ax3 = fig.add_axes([0.55, 0.10, 0.40, 0.36])
    
    if not prophet_metrics.empty and not naive_metrics.empty:
        # Get simple naive metrics for comparison
        naive_simple = naive_metrics[naive_metrics["Method"] == "simple"]
        
        countries = prophet_metrics["Country"].values
        prophet_mape = prophet_metrics.set_index("Country")["MAPE_percent"]
        naive_mape = naive_simple.set_index("Country")["MAPE_percent"]
        
        x = np.arange(len(countries))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, [prophet_mape.get(c, 0) for c in countries], width,
                       label="Prophet", color=AZ_COLORS["purple"], alpha=0.85)
        bars2 = ax3.bar(x + width/2, [naive_mape.get(c, 0) for c in countries], width,
                       label="Naive (Simple)", color=AZ_COLORS["blue"], alpha=0.85)
        
        ax3.set_title("Prophet vs Naive Baseline", fontsize=10, fontweight="bold", pad=5)
        ax3.set_ylabel("MAPE (%)", fontsize=8)
        ax3.set_xlabel("Country", fontsize=8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(countries, fontsize=7)
        ax3.tick_params(axis='y', labelsize=7)
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(True, alpha=0.3, axis="y")
    
    # Footer
    fig.text(0.05, 0.02, "Validation: 4-week holdout backtesting | Metrics: RMSE, MAE, MAPE",
             fontsize=8, color=AZ_COLORS["gray"])
    fig.text(0.95, 0.02, "Page 4 of 5", fontsize=8, color=AZ_COLORS["gray"], ha="right")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 4: Model Validation & Performance")


# --------------------------------------------------------------------------------------
# PAGE 5: INSIGHTS & RECOMMENDATIONS
# --------------------------------------------------------------------------------------

def create_page5_insights(pdf, data):
    """Create Page 5: Insights & Recommendations."""
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape - same as other pages
    fig.patch.set_facecolor(AZ_COLORS["white"])
    
    # Header
    fig.text(0.5, 0.95, "INSIGHTS & RECOMMENDATIONS", fontsize=18,
             fontweight="bold", color=AZ_COLORS["purple"], ha="center")
    fig.text(0.5, 0.91, "Key findings and actionable recommendations for finance teams",
             fontsize=10, color=AZ_COLORS["gray"], ha="center")
    
    line = plt.Line2D([0.05, 0.95], [0.88, 0.88], transform=fig.transFigure,
                      color=AZ_COLORS["purple"], linewidth=2)
    fig.add_artist(line)
    
    prophet_metrics = data.get("prophet_metrics", pd.DataFrame())
    
    # Key Insights Box (Left) - adjusted position
    ax_insights = fig.add_axes([0.05, 0.42, 0.43, 0.42])
    ax_insights.set_facecolor(AZ_COLORS["light_gray"])
    ax_insights.axis("off")
    
    insights_text = """KEY INSIGHTS

[Cash Flow Patterns]
  - Consistent negative net cash flow across most countries
  - Operating cash flows dominate (AP payments, Payroll)
  - Month-end patterns visible in weekly data

[Forecast Performance]
  - Prophet model shows variable accuracy by country
  - Best performance: ID, TH (MAPE < 30%)
  - Challenging: KR, SS (high volatility)

[Liquidity Observations]
  - Short-term liquidity risk: All countries show
    negative projected cash flows
  - 6-month outlook: Stable negative trend expected
  - Key drivers: Accounts Payable, Payroll cycles

[Data Quality Notes]
  - 44 weeks of historical data (Jan-Nov 2025)
  - 8 countries/entities analyzed
  - Clean data with no missing values"""
    
    ax_insights.text(0.05, 0.98, insights_text, transform=ax_insights.transAxes,
                    fontsize=8.5, verticalalignment="top", fontfamily="monospace",
                    color=AZ_COLORS["black"], linespacing=1.3)
    
    for spine in ax_insights.spines.values():
        spine.set_visible(True)
        spine.set_color(AZ_COLORS["purple"])
        spine.set_linewidth(2)
    
    # Recommendations Box (Right) - adjusted position
    ax_recs = fig.add_axes([0.52, 0.42, 0.43, 0.42])
    ax_recs.set_facecolor("#F5F5F5")
    ax_recs.axis("off")
    
    recs_text = """RECOMMENDATIONS

[Short-Term Actions: 1-4 Weeks]
  - Monitor high-volatility countries (KR, SS)
  - Review AP payment timing for optimization
  - Set up alerts for cash flow deviations > 20%

[Medium-Term Actions: 1-6 Months]
  - Implement rolling forecast updates (weekly)
  - Consider ensemble model approach
  - Add external regressors (FX rates, seasonality)

[Model Improvements]
  - Collect more historical data for better training
  - Add ARIMA/XGBoost for comparison
  - Implement anomaly detection pipeline

[Dashboard Enhancements]
  - Add real-time data refresh capability
  - Include drill-down by transaction type
  - Integrate with Power BI for interactivity"""
    
    ax_recs.text(0.05, 0.98, recs_text, transform=ax_recs.transAxes,
                fontsize=8.5, verticalalignment="top", fontfamily="monospace",
                color=AZ_COLORS["black"], linespacing=1.3)
    
    for spine in ax_recs.spines.values():
        spine.set_visible(True)
        spine.set_color(AZ_COLORS["blue"])
        spine.set_linewidth(2)
    
    # Model Performance Summary Title (separate from table)
    fig.text(0.5, 0.38, "MODEL PERFORMANCE SUMMARY", fontsize=13, fontweight="bold",
             ha="center", color=AZ_COLORS["purple"])
    
    # Performance Summary Table (Bottom) - more space
    ax_table = fig.add_axes([0.08, 0.06, 0.84, 0.28])
    ax_table.axis("off")
    
    if not prophet_metrics.empty:
        table_data = []
        for _, row in prophet_metrics.iterrows():
            if row["MAPE_percent"] < 30:
                status = "Good"
            elif row["MAPE_percent"] < 60:
                status = "Fair"
            else:
                status = "Review"
            table_data.append([
                row["Country"],
                f"${row['RMSE_USD']:,.0f}",
                f"${row['MAE_USD']:,.0f}",
                f"{row['MAPE_percent']:.1f}%",
                status
            ])
        
        col_labels = ["Country", "RMSE (USD)", "MAE (USD)", "MAPE", "Status"]
        table = ax_table.table(cellText=table_data, colLabels=col_labels,
                              loc="center", cellLoc="center",
                              colWidths=[0.15, 0.2, 0.2, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)  # Increase row height
        
        # Style header row (row 0)
        for i in range(5):
            cell = table[(0, i)]
            cell.set_facecolor(AZ_COLORS["purple"])
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_height(0.08)  # Taller header
        
        # Style data rows with alternating colors
        for row_idx in range(1, len(table_data) + 1):
            for col_idx in range(5):
                cell = table[(row_idx, col_idx)]
                if row_idx % 2 == 0:
                    cell.set_facecolor("#F5F5F5")
                else:
                    cell.set_facecolor("white")
                
                # Color code Status column
                if col_idx == 4:
                    status_val = table_data[row_idx - 1][4]
                    if status_val == "Good":
                        cell.set_text_props(color=AZ_COLORS["green"], fontweight="bold")
                    elif status_val == "Fair":
                        cell.set_text_props(color=AZ_COLORS["orange"], fontweight="bold")
                    else:
                        cell.set_text_props(color=AZ_COLORS["red"], fontweight="bold")
    
    # Footer
    fig.text(0.05, 0.02, "Prepared by: Three Guys Team | AstraZeneca Datathon 2025",
             fontsize=8, color=AZ_COLORS["gray"])
    fig.text(0.95, 0.02, "Page 5 of 5", fontsize=8, color=AZ_COLORS["gray"], ha="right")
    
    pdf.savefig(fig)
    plt.close(fig)
    print("  [OK] Page 5: Insights & Recommendations")


# --------------------------------------------------------------------------------------
# MAIN DASHBOARD GENERATOR
# --------------------------------------------------------------------------------------

def generate_dashboard_pdf():
    """Generate the complete 5-page PDF dashboard."""
    
    print("\n" + "=" * 70)
    print("  GENERATING PROFESSIONAL PDF DASHBOARD")
    print("  AstraZeneca Cash Flow Challenge")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    print("\n[Step 1] Loading data...")
    data = load_all_data()
    
    for key, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"  - {key}: {len(df)} records")
    
    # Generate PDF
    output_file = OUTPUT_DIR / "AstraZeneca_CashFlow_Dashboard.pdf"
    
    print(f"\n[Step 2] Generating PDF pages...")
    
    with PdfPages(output_file) as pdf:
        create_page1_executive_summary(pdf, data)
        create_page2_trends_analysis(pdf, data)
        create_page3_prophet_forecast(pdf, data)
        create_page4_model_validation(pdf, data)
        create_page5_insights(pdf, data)
    
    print("\n" + "=" * 70)
    print("  DASHBOARD GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n[PDF] Output: {output_file}")
    print(f"      Size: {output_file.stat().st_size / 1024:.1f} KB")
    
    return output_file


# --------------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    generate_dashboard_pdf()

