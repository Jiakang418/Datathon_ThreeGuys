"""
Naive Cash Flow Forecasting Model
=================================
Phase 3: Baseline Forecasting Engine for AstraZeneca Datathon

This script implements Naive forecasting methods as baseline models:
- Simple Naive: Forecast = last observed value
- Drift Naive: Forecast = last value + average change (trend)
- Seasonal Naive: Forecast = value from same period in previous cycle

Author: Marcus
Date: 2025-12-19
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # Datathon_ThreeGuys root
DATA_DIR = BASE_DIR / "data" / "model_dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "naive_results"

WEEKLY_DATA_FILE = DATA_DIR / "processed_weekly_cashflow.csv"


# --------------------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------------------

def load_weekly_data() -> pd.DataFrame:
    """Load the weekly cashflow data prepared in Phase 1."""
    if not WEEKLY_DATA_FILE.exists():
        raise FileNotFoundError(f"Weekly data not found at {WEEKLY_DATA_FILE}")
    
    df = pd.read_csv(WEEKLY_DATA_FILE)
    df["Week_Ending_Date"] = pd.to_datetime(df["Week_Ending_Date"])
    return df


def prepare_naive_data(
    df: pd.DataFrame, 
    country: str, 
    target_col: str = "Net_Cash_Flow"
) -> pd.DataFrame:
    """
    Prepare data for naive forecasting.
    
    Args:
        df: Weekly cashflow dataframe
        country: Country code to filter (e.g., 'TW', 'ID')
        target_col: Column to forecast
    
    Returns:
        DataFrame with 'ds' (date) and 'y' (value) columns
    """
    country_df = df[df["Country_Name"] == country].copy()
    country_df = country_df.sort_values("Week_Ending_Date")
    
    naive_df = pd.DataFrame({
        "ds": country_df["Week_Ending_Date"],
        "y": country_df[target_col]
    })
    
    return naive_df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# NAIVE FORECASTING METHODS
# --------------------------------------------------------------------------------------

def simple_naive_forecast(series: pd.Series, periods: int) -> np.ndarray:
    """
    Simple Naive: Forecast = last observed value repeated.
    
    This is the simplest baseline. If last week's cash flow was $10,000,
    predict $10,000 for all future weeks.
    """
    last_value = series.iloc[-1]
    return np.full(periods, last_value)


def drift_naive_forecast(series: pd.Series, periods: int) -> np.ndarray:
    """
    Drift Naive: Forecast = last value + (average change per period × h)
    
    This accounts for an overall trend in the data.
    Formula: y_T+h = y_T + h * (y_T - y_1) / (T - 1)
    """
    n = len(series)
    if n < 2:
        return simple_naive_forecast(series, periods)
    
    last_value = series.iloc[-1]
    first_value = series.iloc[0]
    avg_drift = (last_value - first_value) / (n - 1)
    
    forecasts = np.array([last_value + (h + 1) * avg_drift for h in range(periods)])
    return forecasts


def seasonal_naive_forecast(
    series: pd.Series, 
    periods: int, 
    seasonal_period: int = 4
) -> np.ndarray:
    """
    Seasonal Naive: Forecast = value from the same period in the last season.
    
    For weekly data with monthly patterns, seasonal_period=4 (4 weeks = 1 month).
    """
    n = len(series)
    forecasts = []
    
    for h in range(periods):
        # Find corresponding seasonal index
        idx = n - seasonal_period + (h % seasonal_period)
        if idx >= 0:
            forecasts.append(series.iloc[idx])
        else:
            # Fallback to simple naive if not enough history
            forecasts.append(series.iloc[-1])
    
    return np.array(forecasts)


def mean_naive_forecast(series: pd.Series, periods: int) -> np.ndarray:
    """
    Mean Naive: Forecast = historical mean.
    
    Simple baseline using the average of all historical values.
    """
    mean_value = series.mean()
    return np.full(periods, mean_value)


# --------------------------------------------------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------------------------------------------------

def train_validation_split(
    df: pd.DataFrame, 
    validation_weeks: int = 4
) -> tuple:
    """
    Split time series for backtesting.
    Train: All data except last N weeks
    Validation: Last N weeks
    """
    cutoff_idx = len(df) - validation_weeks
    train_df = df.iloc[:cutoff_idx].copy()
    val_df = df.iloc[cutoff_idx:].copy()
    
    return train_df, val_df


# --------------------------------------------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------------------------------------------

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate RMSE, MAE, and MAPE for forecast evaluation."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    # MAPE (handle zero/near-zero values)
    non_zero_mask = np.abs(actual) > 1e-6
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) 
                              / actual[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE_percent": round(mape, 2) if not np.isnan(mape) else None
    }


# --------------------------------------------------------------------------------------
# BACKTESTING (VALIDATION)
# --------------------------------------------------------------------------------------

def backtest_naive(
    df: pd.DataFrame,
    country: str,
    method: str = "simple",
    validation_weeks: int = 4
) -> dict:
    """
    Perform backtesting: train on historical, validate on holdout.
    
    Args:
        df: DataFrame with 'ds' and 'y' columns
        country: Country code
        method: 'simple', 'drift', 'seasonal', or 'mean'
        validation_weeks: Number of weeks for validation
    
    Returns dict with forecasts, actuals, and metrics.
    """
    train_df, val_df = train_validation_split(df, validation_weeks)
    
    train_series = train_df["y"]
    
    # Generate forecasts based on method
    if method == "simple":
        predictions = simple_naive_forecast(train_series, validation_weeks)
    elif method == "drift":
        predictions = drift_naive_forecast(train_series, validation_weeks)
    elif method == "seasonal":
        predictions = seasonal_naive_forecast(train_series, validation_weeks, seasonal_period=4)
    elif method == "mean":
        predictions = mean_naive_forecast(train_series, validation_weeks)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    actuals = val_df["y"].values
    metrics = calculate_metrics(actuals, predictions)
    
    return {
        "method": method,
        "train_df": train_df,
        "val_df": val_df,
        "actuals": actuals,
        "predictions": predictions,
        "metrics": metrics,
        "country": country
    }


# --------------------------------------------------------------------------------------
# FUTURE FORECASTING
# --------------------------------------------------------------------------------------

def generate_future_forecast(
    df: pd.DataFrame,
    periods: int,
    country: str,
    method: str = "simple"
) -> pd.DataFrame:
    """
    Train on ALL available data and forecast into the future.
    
    Args:
        df: Full data (ds, y columns)
        periods: Number of weeks to forecast (1, 4, or 26)
        country: Country code
        method: 'simple', 'drift', 'seasonal', or 'mean'
    
    Returns:
        Forecast dataframe with predictions
    """
    series = df["y"]
    last_date = df["ds"].max()
    
    # Generate forecasts
    if method == "simple":
        predictions = simple_naive_forecast(series, periods)
    elif method == "drift":
        predictions = drift_naive_forecast(series, periods)
    elif method == "seasonal":
        predictions = seasonal_naive_forecast(series, periods, seasonal_period=4)
    elif method == "mean":
        predictions = mean_naive_forecast(series, periods)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create future dates (weekly, Sunday)
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=7),
        periods=periods,
        freq="W-SUN"
    )
    
    result_df = pd.DataFrame({
        "Week_Ending_Date": future_dates,
        "Predicted_Cash_Flow": predictions,
        "Country": country,
        "Method": method
    })
    
    return result_df


# --------------------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------------------

def run_naive_pipeline():
    """Main execution pipeline for Naive forecasting."""
    
    print("=" * 70)
    print("  NAIVE CASH FLOW FORECASTING - Baseline Models")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading weekly cashflow data...")
    df = load_weekly_data()
    countries = df["Country_Name"].unique()
    
    print(f"  - Countries found: {list(countries)}")
    print(f"  - Date range: {df['Week_Ending_Date'].min().date()} to {df['Week_Ending_Date'].max().date()}")
    print(f"  - Total rows: {len(df)}")
    
    # Methods to evaluate
    methods = ["simple", "drift", "seasonal", "mean"]
    
    # -------------------------------------------------------------------------
    # Step 2: Backtest Each Country & Method (Validation)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Running backtesting (4-week validation holdout)...")
    
    all_backtest_results = {}
    metrics_list = []
    
    for country in countries:
        print(f"\n  Processing {country}...")
        
        naive_df = prepare_naive_data(df, country, "Net_Cash_Flow")
        
        if len(naive_df) < 10:
            print(f"    [!] Skipping {country}: insufficient data ({len(naive_df)} weeks)")
            continue
        
        country_results = {}
        
        for method in methods:
            result = backtest_naive(naive_df, country, method=method, validation_weeks=4)
            country_results[method] = result
            
            metrics = result["metrics"]
            metrics_list.append({
                "Country": country,
                "Method": method,
                "RMSE_USD": metrics["RMSE"],
                "MAE_USD": metrics["MAE"],
                "MAPE_percent": metrics["MAPE_percent"],
                "Train_Weeks": len(result["train_df"]),
                "Validation_Weeks": len(result["val_df"])
            })
            
            print(f"    [{method.upper()}] RMSE: ${metrics['RMSE']:,.2f} | MAE: ${metrics['MAE']:,.2f} | MAPE: {metrics['MAPE_percent']}%")
        
        all_backtest_results[country] = country_results
    
    # Save metrics summary
    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = OUTPUT_DIR / "naive_backtest_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n  [OK] Saved backtest metrics to: {metrics_file}")
    
    # -------------------------------------------------------------------------
    # Step 3: Generate Future Forecasts (1-week, 1-month, 6-month)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Generating future forecasts...")
    
    all_forecasts = []
    
    # Use "simple" as the primary naive method (industry standard baseline)
    primary_method = "simple"
    
    for country in countries:
        naive_df = prepare_naive_data(df, country, "Net_Cash_Flow")
        
        if len(naive_df) < 10:
            continue
        
        # 1-week forecast
        forecast_1w = generate_future_forecast(naive_df, periods=1, country=country, method=primary_method)
        forecast_1w["Horizon"] = "1_week"
        
        # 1-month forecast (4 weeks)
        forecast_1m = generate_future_forecast(naive_df, periods=4, country=country, method=primary_method)
        forecast_1m["Horizon"] = "1_month"
        
        # 6-month forecast (26 weeks)
        forecast_6m = generate_future_forecast(naive_df, periods=26, country=country, method=primary_method)
        forecast_6m["Horizon"] = "6_month"
        
        all_forecasts.extend([forecast_1w, forecast_1m, forecast_6m])
        
        print(f"  [OK] {country}: Generated 1-week, 1-month, and 6-month forecasts")
    
    # Combine and save forecasts
    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    forecast_file = OUTPUT_DIR / "naive_future_forecasts.csv"
    forecast_df.to_csv(forecast_file, index=False)
    print(f"\n  [OK] Saved future forecasts to: {forecast_file}")
    
    # -------------------------------------------------------------------------
    # Step 4: Create Actual vs Predicted Comparison (for validation period)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Creating actual vs predicted comparison...")
    
    comparison_list = []
    for country, methods_results in all_backtest_results.items():
        # Use simple naive for primary comparison
        result = methods_results["simple"]
        val_dates = result["val_df"]["ds"].values
        
        for i, date in enumerate(val_dates):
            comparison_list.append({
                "Country": country,
                "Week_Ending_Date": pd.Timestamp(date),
                "Actual_Cash_Flow": result["actuals"][i],
                "Predicted_Cash_Flow": result["predictions"][i],
                "Error": result["actuals"][i] - result["predictions"][i],
                "Method": "simple"
            })
    
    comparison_df = pd.DataFrame(comparison_list)
    comparison_file = OUTPUT_DIR / "naive_actual_vs_predicted.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"  [OK] Saved actual vs predicted to: {comparison_file}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  NAIVE FORECASTING COMPLETE!")
    print("=" * 70)
    print("\nOutput files created:")
    print(f"  1. {metrics_file.name} - Backtest metrics per country & method")
    print(f"  2. {forecast_file.name} - Future predictions (1w, 1m, 6m)")
    print(f"  3. {comparison_file.name} - Validation actual vs predicted")
    
    # Best method summary
    print("\n" + "-" * 70)
    print("  BEST METHOD BY COUNTRY (Lowest MAPE)")
    print("-" * 70)
    
    best_methods = metrics_df.loc[metrics_df.groupby("Country")["MAPE_percent"].idxmin()]
    print(best_methods[["Country", "Method", "MAPE_percent"]].to_string(index=False))
    
    # Overall comparison
    print("\n" + "-" * 70)
    print("  AVERAGE MAPE BY METHOD")
    print("-" * 70)
    avg_by_method = metrics_df.groupby("Method")["MAPE_percent"].mean().sort_values()
    for method, mape in avg_by_method.items():
        print(f"  {method.upper():10} : {mape:.2f}%")
    
    return all_backtest_results, forecast_df, metrics_df


# --------------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    results, forecasts, metrics = run_naive_pipeline()

