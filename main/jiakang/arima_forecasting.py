"""
ARIMA Cash Flow Forecasting
===========================
Phase 3: ARIMA Forecasting Engine for Cash Flow Data

This script:
- Loads the processed weekly cashflow data with lag features
- Trains ARIMA models for each country (with configurable ARIMA order)
- Performs backtesting with train/validation split
- Generates 1-month (4 weeks) and 6-month (26 weeks) forecasts with confidence intervals
- Calculates evaluation metrics (RMSE, MAE, MAPE)
- Saves results for visualization
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# --------------------------------------------------------------------------------------
# Resolve base directory: main/jiakang/arima_forecasting.py → Datathon_ThreeGuys/
BASE_DIR = Path(__file__).resolve().parents[2]

# Try multiple possible paths for the data file
POSSIBLE_DATA_PATHS = [
    BASE_DIR / "data" / "model_dataset" / "weekly_features.csv",
    BASE_DIR / "data" / "processed" / "weekly_features.csv",
    BASE_DIR / "Steve" / "processed_weekly_cashflow.csv"
]

# Find the first existing data file
DATA_FILE = None
for path in POSSIBLE_DATA_PATHS:
    if path.exists():
        DATA_FILE = path
        break

OUTPUT_DIR = Path(__file__).parent / "arima_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# Model configuration
ARIMA_ORDER = (1, 0, 0)  # (p, d, q)
MIN_DATA_POINTS = 10
VALIDATION_WEEKS = 4
FORECAST_1M_WEEKS = 4
FORECAST_6M_WEEKS = 26

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------------------

def load_weekly_data() -> pd.DataFrame:
    """Load weekly features data from CSV file."""
    if DATA_FILE is None or not DATA_FILE.exists():
        error_msg = "Weekly data file not found. Checked paths:\n"
        for path in POSSIBLE_DATA_PATHS:
            error_msg += f"  - {path}\n"
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Loading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, parse_dates=["Week_Ending_Date"])
    logger.info(f"Loaded {len(df)} rows, {len(df['Country_Name'].unique())} countries")
    return df

# --------------------------------------------------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------------------------------------------------

def train_validation_split(df: pd.DataFrame, validation_weeks: int = VALIDATION_WEEKS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_idx = len(df) - validation_weeks
    train_df = df.iloc[:cutoff_idx].copy()
    val_df = df.iloc[cutoff_idx:].copy()
    return train_df, val_df

# --------------------------------------------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------------------------------------------

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    threshold = max(np.median(np.abs(actual)) * 0.01, 1e-6)
    mask = np.abs(actual) > threshold
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else None
    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE_percent": round(mape, 2) if mape is not None else None
    }

# --------------------------------------------------------------------------------------
# ARIMA MODEL CREATION
# --------------------------------------------------------------------------------------

def fit_arima_model(endog: pd.Series, exog: pd.DataFrame = None, order: Tuple[int, int, int] = ARIMA_ORDER):
    try:
        model = ARIMA(endog=endog, exog=exog, order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        logger.error(f"ARIMA model fitting failed: {e}")
        return None

# --------------------------------------------------------------------------------------
# BACKTESTING
# --------------------------------------------------------------------------------------

def backtest_arima(df: pd.DataFrame, country: str, target_col: str = "Net_Cash_Flow", validation_weeks: int = VALIDATION_WEEKS) -> Dict:
    train_df, val_df = train_validation_split(df, validation_weeks)
    exog_cols = [c for c in df.columns if "Lag" in c or "Roll" in c]
    train_exog = train_df[exog_cols] if exog_cols else None
    val_exog = val_df[exog_cols] if exog_cols else None
    model_fit = fit_arima_model(train_df[target_col], train_exog, ARIMA_ORDER)
    if model_fit is None:
        logger.warning(f"Failed to fit model for {country}")
        return None
    try:
        predictions = model_fit.forecast(steps=validation_weeks, exog=val_exog)
        metrics = calculate_metrics(val_df[target_col].values, predictions.values)
        logger.info(f"  {country} - RMSE: {metrics['RMSE']}, MAE: {metrics['MAE']}, MAPE: {metrics['MAPE_percent']}%")
        return {
            "model": model_fit,
            "train_df": train_df,
            "val_df": val_df,
            "actuals": val_df[target_col].values,
            "predictions": predictions.values,
            "metrics": metrics,
            "country": country
        }
    except Exception as e:
        logger.error(f"Forecasting failed for {country}: {e}")
        return None

# --------------------------------------------------------------------------------------
# FUTURE FORECAST
# --------------------------------------------------------------------------------------

def generate_future_forecast(df: pd.DataFrame, country: str, periods: int = FORECAST_1M_WEEKS, target_col: str = "Net_Cash_Flow") -> pd.DataFrame:
    model_fit = fit_arima_model(df[target_col], exog=None, order=ARIMA_ORDER)
    if model_fit is None:
        logger.warning(f"Failed to generate forecast for {country}")
        return pd.DataFrame()
    try:
        forecast_result = model_fit.get_forecast(steps=periods)
        forecast_values = forecast_result.predicted_mean.values
        conf_int = forecast_result.conf_int()
        future_dates = pd.date_range(df["Week_Ending_Date"].max() + pd.Timedelta(days=7), periods=periods, freq="W-SUN")
        forecast_df = pd.DataFrame({
            "Week_Ending_Date": future_dates,
            "Predicted_Cash_Flow": forecast_values,
            "Lower_CI": conf_int.iloc[:, 0].values,
            "Upper_CI": conf_int.iloc[:, 1].values,
            "Country": country
        })
        logger.info(f"  Generated {periods}-week forecast for {country}")
        return forecast_df
    except Exception as e:
        logger.error(f"Future forecasting failed for {country}: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------------------

def run_arima_pipeline():
    """
    Main ARIMA pipeline execution function.
    
    Returns:
        Tuple of (results_dict, forecast_df, metrics_df)
    """
    logger.info("="*70)
    logger.info("  ARIMA CASH FLOW FORECASTING")
    logger.info("="*70)
    logger.info(f"Configuration: ARIMA{ARIMA_ORDER}, Validation: {VALIDATION_WEEKS} weeks")
    
    df = load_weekly_data()
    countries = df["Country_Name"].unique()
    
    all_results = {}
    all_forecasts = []
    metrics_list = []
    
    for country in countries:
        logger.info(f"\nProcessing {country}...")
        country_df = df[df["Country_Name"] == country].sort_values("Week_Ending_Date").reset_index(drop=True)
        
        # Data validation
        if len(country_df) < MIN_DATA_POINTS:
            logger.warning(f"  Skipping {country}: only {len(country_df)} data points (minimum: {MIN_DATA_POINTS})")
            continue
        
        if len(country_df) < VALIDATION_WEEKS + MIN_DATA_POINTS:
            logger.warning(f"  Skipping {country}: insufficient data for {VALIDATION_WEEKS}-week validation")
            continue
        
        # Backtest
        result = backtest_arima(country_df, country)
        if result is None:
            continue
            
        all_results[country] = result
        
        metrics = result["metrics"]
        metrics_list.append({
            "Country": country,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE_percent": metrics["MAPE_percent"],
            "Train_Weeks": len(result["train_df"]),
            "Validation_Weeks": len(result["val_df"])
        })
        
        # Generate future forecasts
        forecast_1m = generate_future_forecast(country_df, country, periods=FORECAST_1M_WEEKS)
        if not forecast_1m.empty:
            forecast_1m["Horizon"] = "1_month"
            all_forecasts.append(forecast_1m)
        
        forecast_6m = generate_future_forecast(country_df, country, periods=FORECAST_6M_WEEKS)
        if not forecast_6m.empty:
            forecast_6m["Horizon"] = "6_month"
            all_forecasts.append(forecast_6m)
    
    # Save results
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(OUTPUT_DIR / "arima_backtest_metrics.csv", index=False)
    logger.info(f"\nSaved metrics to {OUTPUT_DIR / 'arima_backtest_metrics.csv'}")
    
    if all_forecasts:
        forecast_df = pd.concat(all_forecasts, ignore_index=True)
        forecast_df.to_csv(OUTPUT_DIR / "arima_future_forecasts.csv", index=False)
        logger.info(f"Saved forecasts to {OUTPUT_DIR / 'arima_future_forecasts.csv'}")
    else:
        forecast_df = pd.DataFrame()
        logger.warning("No forecasts generated")
    
    # -----------------------------
    # Calculate and log averages
    # -----------------------------
    if not metrics_df.empty:
        avg_rmse = metrics_df["RMSE"].mean()
        avg_mape = metrics_df["MAPE_percent"].mean()

        # Format table nicely
        metrics_table = metrics_df.copy()
        metrics_table["RMSE_USD"] = metrics_table["RMSE"].apply(lambda x: f"${x:,.2f}")
        metrics_table["MAE_USD"] = metrics_table["MAE"].apply(lambda x: f"${x:,.2f}")
        metrics_table = metrics_table[["Country", "RMSE_USD", "MAE_USD", "MAPE_percent", "Train_Weeks", "Validation_Weeks"]]

        logger.info("\nOverall Model Performance:")
        logger.info("\n" + metrics_table.to_string(index=False))
        logger.info(f"\nAverage RMSE across all countries: ${avg_rmse:,.2f}")
        logger.info(f"Average MAPE across all countries: {avg_mape:.1f}%")
    
    logger.info("\n" + "="*70)
    logger.info("[OK] ARIMA pipeline complete!")
    logger.info("="*70)
    
    return all_results, forecast_df, metrics_df


