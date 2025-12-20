"""
Prophet Cash Flow Forecasting Model
===================================
Phase 3: Forecasting Engine for AstraZeneca Datathon

This script:
- Loads the processed weekly cashflow data (from Steve's Phase 1)
- Trains Prophet models for each country
- Performs backtesting with train/validation split
- Generates 1-month (4 weeks) and 6-month (26 weeks) forecasts
- Calculates evaluation metrics (RMSE, MAE, MAPE)
- Saves results for Phase 4 (Evaluation) and Phase 5 (Dashboard)

Author: Marcus
Date: 2025-12-19
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # Datathon_ThreeGuys root
STEVE_DIR = BASE_DIR / "Steve"
OUTPUT_DIR = Path(__file__).resolve().parent / "prophet_results"

WEEKLY_DATA_FILE = BASE_DIR / "data" / "model_dataset" / "weekly_features.csv"


# --------------------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------------------

def load_weekly_data() -> pd.DataFrame:
    """Load the weekly cashflow data prepared by Steve in Phase 1."""
    if not WEEKLY_DATA_FILE.exists():
        raise FileNotFoundError(f"Weekly data not found at {WEEKLY_DATA_FILE}")
    
    df = pd.read_csv(WEEKLY_DATA_FILE)
    df["Week_Ending_Date"] = pd.to_datetime(df["Week_Ending_Date"])
    return df


def prepare_prophet_data(
    df: pd.DataFrame, 
    country: str, 
    target_col: str = "Net_Cash_Flow"
) -> pd.DataFrame:
    """
    Prepare data in Prophet's required format (ds, y).
    
    Args:
        df: Weekly cashflow dataframe
        country: Country code to filter (e.g., 'TW', 'ID')
        target_col: Column to forecast
    
    Returns:
        DataFrame with 'ds' (date) and 'y' (value) columns
    """
    country_df = df[df["Country_Name"] == country].copy()
    country_df = country_df.sort_values("Week_Ending_Date")
    
    prophet_df = pd.DataFrame({
        "ds": country_df["Week_Ending_Date"],
        "y": country_df[target_col]
    })
    
    return prophet_df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# PROPHET MODEL
# --------------------------------------------------------------------------------------

def build_prophet_model(
    yearly_seasonality: bool = False,
    weekly_seasonality: bool = False,
    seasonality_mode: str = "additive"
) -> Prophet:
    """
    Build a Prophet model configured for weekly cash flow forecasting.
    
    Key settings:
    - No yearly seasonality (data is < 1 year)
    - No weekly seasonality (data is already weekly aggregated)
    - Monthly seasonality added (captures month-end patterns)
    """
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
    )
    
    # Add monthly seasonality (~4 weeks cycle for payroll/tax patterns)
    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=5
    )
    
    return model


# --------------------------------------------------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------------------------------------------------

def train_validation_split(
    df: pd.DataFrame, 
    validation_weeks: int = 4
) -> tuple:
    """
    Split time series for backtesting.
    Train: Jan - Aug 2025
    Validation: Last N weeks (default 4 = Sept 2025)
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

def backtest_prophet(
    df: pd.DataFrame,
    country: str,
    target_col: str = "Net_Cash_Flow",
    validation_weeks: int = 4
) -> dict:
    """
    Perform backtesting: train on historical, validate on holdout.
    
    Returns dict with model, forecasts, actuals, and metrics.
    """
    train_df, val_df = train_validation_split(df, validation_weeks)
    
    # Build and train model
    model = build_prophet_model()
    model.fit(train_df)
    
    # Create future dataframe for validation period
    future = model.make_future_dataframe(periods=validation_weeks, freq="W-SUN")
    forecast = model.predict(future)
    
    # Extract validation predictions
    forecast_val = forecast.tail(validation_weeks)
    
    actuals = val_df["y"].values
    predictions = forecast_val["yhat"].values
    
    metrics = calculate_metrics(actuals, predictions)
    
    return {
        "model": model,
        "forecast_full": forecast,
        "train_df": train_df,
        "val_df": val_df,
        "actuals": actuals,
        "predictions": predictions,
        "prediction_lower": forecast_val["yhat_lower"].values,
        "prediction_upper": forecast_val["yhat_upper"].values,
        "metrics": metrics,
        "country": country,
        "target": target_col
    }


# --------------------------------------------------------------------------------------
# FUTURE FORECASTING
# --------------------------------------------------------------------------------------

def generate_future_forecast(
    df: pd.DataFrame,
    periods: int,
    country: str,
    target_col: str = "Net_Cash_Flow"
) -> pd.DataFrame:
    """
    Train on ALL available data and forecast into the future.
    
    Args:
        df: Full prophet-formatted data (ds, y)
        periods: Number of weeks to forecast (4 = 1 month, 26 = 6 months)
        country: Country code
        target_col: Target variable name
    
    Returns:
        Forecast dataframe with predictions
    """
    model = build_prophet_model()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=periods, freq="W-SUN")
    forecast = model.predict(future)
    
    # Extract only future predictions
    last_historical_date = df["ds"].max()
    future_only = forecast[forecast["ds"] > last_historical_date].copy()
    
    future_only["Country"] = country
    future_only["Target"] = target_col
    
    return future_only[[
        "ds", "yhat", "yhat_lower", "yhat_upper", "Country", "Target"
    ]].rename(columns={
        "ds": "Week_Ending_Date",
        "yhat": "Predicted_Cash_Flow",
        "yhat_lower": "Prediction_Lower_95CI",
        "yhat_upper": "Prediction_Upper_95CI"
    })


# --------------------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------------------

def run_prophet_pipeline():
    """Main execution pipeline for Prophet forecasting."""
    
    print("=" * 70)
    print("  PROPHET CASH FLOW FORECASTING - Phase 3")
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
    
    # -------------------------------------------------------------------------
    # Step 2: Backtest Each Country (Validation)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Running backtesting (4-week validation holdout)...")
    
    all_backtest_results = {}
    metrics_list = []
    
    for country in countries:
        print(f"\n  Processing {country}...")
        
        prophet_df = prepare_prophet_data(df, country, "Net_Cash_Flow")
        
        if len(prophet_df) < 10:
            print(f"    [!] Skipping {country}: insufficient data ({len(prophet_df)} weeks)")
            continue
        
        result = backtest_prophet(prophet_df, country, "Net_Cash_Flow", validation_weeks=4)
        all_backtest_results[country] = result
        
        metrics = result["metrics"]
        metrics_list.append({
            "Country": country,
            "RMSE_USD": metrics["RMSE"],
            "MAE_USD": metrics["MAE"],
            "MAPE_percent": metrics["MAPE_percent"],
            "Train_Weeks": len(result["train_df"]),
            "Validation_Weeks": len(result["val_df"])
        })
        
        print(f"    [OK] RMSE: ${metrics['RMSE']:,.2f} | MAE: ${metrics['MAE']:,.2f} | MAPE: {metrics['MAPE_percent']}%")
    
    # Save metrics summary
    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = OUTPUT_DIR / "prophet_backtest_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n  [OK] Saved backtest metrics to: {metrics_file}")
    
    # -------------------------------------------------------------------------
    # Step 3: Generate Future Forecasts (1-month & 6-month)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Generating future forecasts...")
    
    all_forecasts = []
    
    for country in countries:
        prophet_df = prepare_prophet_data(df, country, "Net_Cash_Flow")
        
        if len(prophet_df) < 10:
            continue
        
        # 1-month forecast (4 weeks)
        forecast_1m = generate_future_forecast(prophet_df, periods=4, country=country)
        forecast_1m["Horizon"] = "1_month"
        
        # 6-month forecast (26 weeks)
        forecast_6m = generate_future_forecast(prophet_df, periods=26, country=country)
        forecast_6m["Horizon"] = "6_month"
        
        all_forecasts.append(forecast_1m)
        all_forecasts.append(forecast_6m)
        
        print(f"  [OK] {country}: Generated 1-month and 6-month forecasts")
    
    # Combine and save forecasts
    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    forecast_file = OUTPUT_DIR / "prophet_future_forecasts.csv"
    forecast_df.to_csv(forecast_file, index=False)
    print(f"\n  [OK] Saved future forecasts to: {forecast_file}")
    
    # -------------------------------------------------------------------------
    # Step 4: Create Actual vs Predicted Comparison (for validation period)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Creating actual vs predicted comparison...")
    
    comparison_list = []
    for country, result in all_backtest_results.items():
        val_dates = result["val_df"]["ds"].values
        for i, date in enumerate(val_dates):
            comparison_list.append({
                "Country": country,
                "Week_Ending_Date": pd.Timestamp(date),
                "Actual_Cash_Flow": result["actuals"][i],
                "Predicted_Cash_Flow": result["predictions"][i],
                "Prediction_Lower_95CI": result["prediction_lower"][i],
                "Prediction_Upper_95CI": result["prediction_upper"][i],
                "Error": result["actuals"][i] - result["predictions"][i]
            })
    
    comparison_df = pd.DataFrame(comparison_list)
    comparison_file = OUTPUT_DIR / "prophet_actual_vs_predicted.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"  [OK] Saved actual vs predicted to: {comparison_file}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PROPHET FORECASTING COMPLETE!")
    print("=" * 70)
    print("\nOutput files created:")
    print(f"  1. {metrics_file.name} - Backtest metrics per country")
    print(f"  2. {forecast_file.name} - Future predictions (1m & 6m)")
    print(f"  3. {comparison_file.name} - Validation actual vs predicted")
    
    print("\nOverall Model Performance:")
    print(metrics_df.to_string(index=False))
    
    avg_mape = metrics_df["MAPE_percent"].mean()
    print(f"\nAverage MAPE across all countries: {avg_mape:.1f}%")
    
    return all_backtest_results, forecast_df, metrics_df


# --------------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    results, forecasts, metrics = run_prophet_pipeline()

