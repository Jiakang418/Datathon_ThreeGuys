"""
Phase 3: Naive Baseline Forecasting Model

This script implements a Naive forecasting baseline (Last Observation Carried Forward)
that produces the exact same 3 output files as the Prophet model for direct comparison.

Output Files:
1. naive_backtest_metrics.csv - RMSE, MAE, MAPE per country
2. naive_actual_vs_predicted.csv - Validation actuals vs predictions with CIs
3. naive_future_forecasts.csv - 1-month and 6-month future forecasts
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


def naive_forecast_with_ci(
    train_values: np.ndarray,
    n_periods: int,
    confidence_level: float = 0.95
) -> tuple:
    """
    Generate naive forecast (last value) with confidence intervals.
    
    Args:
        train_values: Training set values
        n_periods: Number of periods to forecast
        confidence_level: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (predictions, lower_ci, upper_ci, std_dev)
    """
    # Last observed value
    last_value = train_values[-1]
    
    # Calculate standard deviation of training residuals
    # For naive model, residuals = actual - last_value (shifted)
    if len(train_values) > 1:
        # Use all values except the last as "predictions" (using previous value)
        predictions_shifted = np.roll(train_values, 1)[1:]  # Shift by 1, remove first NaN
        actuals = train_values[1:]
        residuals = actuals - predictions_shifted
        std_dev = np.std(residuals)
    else:
        std_dev = 0.0
    
    # Z-score for confidence interval (1.96 for 95% CI)
    z_score = 1.96 if confidence_level == 0.95 else 2.576  # 2.576 for 99% CI
    
    # Generate forecasts (all same value)
    predictions = np.full(n_periods, last_value)
    lower_ci = predictions - (z_score * std_dev)
    upper_ci = predictions + (z_score * std_dev)
    
    return predictions, lower_ci, upper_ci, std_dev


def main():
    """Main execution function."""
    # -------------------------------------------------------------------------
    # 1. Configuration & Load Data
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "data" / "model_dataset" / "weekly_features.csv"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path, parse_dates=["Week_Ending_Date"])
    df = df.sort_values(["Country_Name", "Week_Ending_Date"])
    
    print("=" * 70)
    print("  NAIVE BASELINE FORECASTING - Phase 3")
    print("=" * 70)
    print(f"\nLoaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Countries: {list(df['Country_Name'].unique())}")
    print(f"Date range: {df['Week_Ending_Date'].min().date()} to {df['Week_Ending_Date'].max().date()}")
    
    # -------------------------------------------------------------------------
    # 2. Backtesting (Validation Phase)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Running backtesting (4-week validation holdout)...")
    
    validation_weeks = 4
    metrics_list = []
    actual_vs_predicted_list = []
    
    for country in df["Country_Name"].unique():
        print(f"\n  Processing {country}...")
        
        # Filter country data
        country_df = df[df["Country_Name"] == country].copy()
        country_df = country_df.sort_values("Week_Ending_Date")
        
        if len(country_df) < validation_weeks + 1:
            print(f"    [!] Skipping {country}: insufficient data ({len(country_df)} weeks)")
            continue
        
        # Split into train and validation
        split_idx = len(country_df) - validation_weeks
        train_df = country_df.iloc[:split_idx].copy()
        val_df = country_df.iloc[split_idx:].copy()
        
        # Get training values
        train_values = train_df["Net_Cash_Flow"].values
        val_actuals = val_df["Net_Cash_Flow"].values
        
        # Generate naive predictions for validation period
        val_predictions, val_lower_ci, val_upper_ci, std_dev = naive_forecast_with_ci(
            train_values, n_periods=validation_weeks
        )
        
        # Calculate metrics
        metrics = calculate_metrics(val_actuals, val_predictions)
        
        metrics_list.append({
            "Country": country,
            "RMSE_USD": metrics["RMSE"],
            "MAE_USD": metrics["MAE"],
            "MAPE_percent": metrics["MAPE_percent"],
            "Train_Weeks": len(train_df),
            "Validation_Weeks": len(val_df)
        })
        
        # Store actual vs predicted for validation period
        for i, (idx, row) in enumerate(val_df.iterrows()):
            actual_vs_predicted_list.append({
                "Country": country,
                "Week_Ending_Date": row["Week_Ending_Date"],
                "Actual_Cash_Flow": val_actuals[i],
                "Predicted_Cash_Flow": val_predictions[i],
                "Prediction_Lower_95CI": val_lower_ci[i],
                "Prediction_Upper_95CI": val_upper_ci[i],
                "Error": val_actuals[i] - val_predictions[i]
            })
        
        print(f"    [OK] RMSE: ${metrics['RMSE']:,.2f} | MAE: ${metrics['MAE']:,.2f} | MAPE: {metrics['MAPE_percent']}%")
    
    # Save backtest metrics
    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = output_dir / "naive_backtest_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n  [OK] Saved backtest metrics to: {metrics_file}")
    
    # Save actual vs predicted
    comparison_df = pd.DataFrame(actual_vs_predicted_list)
    comparison_file = output_dir / "naive_actual_vs_predicted.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"  [OK] Saved actual vs predicted to: {comparison_file}")
    
    # -------------------------------------------------------------------------
    # 3. Future Forecasting (Production Phase)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Generating future forecasts (1-month & 6-month)...")
    
    all_forecasts = []
    
    for country in df["Country_Name"].unique():
        country_df = df[df["Country_Name"] == country].copy()
        country_df = country_df.sort_values("Week_Ending_Date")
        
        if len(country_df) < 10:
            continue
        
        # Use ALL available data for future forecasting
        all_values = country_df["Net_Cash_Flow"].values
        last_date = country_df["Week_Ending_Date"].max()
        
        # 1-month forecast (4 weeks)
        forecast_1m_pred, forecast_1m_lower, forecast_1m_upper, _ = naive_forecast_with_ci(
            all_values, n_periods=4
        )
        
        # Generate future dates (weekly, ending on Sunday)
        # Start from next Sunday after last_date (which should already be a Sunday)
        next_sunday = last_date + pd.Timedelta(days=7)
        
        future_dates_1m = pd.date_range(
            start=next_sunday,
            periods=4,
            freq="W-SUN"
        )
        
        for i, date in enumerate(future_dates_1m):
            all_forecasts.append({
                "Week_Ending_Date": date,
                "Predicted_Cash_Flow": forecast_1m_pred[i],
                "Prediction_Lower_95CI": forecast_1m_lower[i],
                "Prediction_Upper_95CI": forecast_1m_upper[i],
                "Country": country,
                "Target": "Net_Cash_Flow",
                "Horizon": "1_month"
            })
        
        # 6-month forecast (26 weeks)
        forecast_6m_pred, forecast_6m_lower, forecast_6m_upper, _ = naive_forecast_with_ci(
            all_values, n_periods=26
        )
        
        future_dates_6m = pd.date_range(
            start=next_sunday,
            periods=26,
            freq="W-SUN"
        )
        
        for i, date in enumerate(future_dates_6m):
            all_forecasts.append({
                "Week_Ending_Date": date,
                "Predicted_Cash_Flow": forecast_6m_pred[i],
                "Prediction_Lower_95CI": forecast_6m_lower[i],
                "Prediction_Upper_95CI": forecast_6m_upper[i],
                "Country": country,
                "Target": "Net_Cash_Flow",
                "Horizon": "6_month"
            })
        
        print(f"  [OK] {country}: Generated 1-month and 6-month forecasts")
    
    # Combine and save forecasts
    forecast_df = pd.DataFrame(all_forecasts)
    forecast_file = output_dir / "naive_future_forecasts.csv"
    forecast_df.to_csv(forecast_file, index=False)
    print(f"\n  [OK] Saved future forecasts to: {forecast_file}")
    
    # -------------------------------------------------------------------------
    # 4. Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  NAIVE BASELINE FORECASTING COMPLETE!")
    print("=" * 70)
    print("\nOutput files created:")
    print(f"  1. {metrics_file.name} - Backtest metrics per country")
    print(f"  2. {comparison_file.name} - Validation actual vs predicted")
    print(f"  3. {forecast_file.name} - Future predictions (1m & 6m)")
    
    print("\nOverall Model Performance:")
    print(metrics_df.to_string(index=False))
    
    avg_rmse = metrics_df["RMSE_USD"].mean()
    avg_mape = metrics_df["MAPE_percent"].mean()
    print(f"\nAverage RMSE across all countries: ${avg_rmse:,.2f}")
    print(f"Average MAPE across all countries: {avg_mape:.1f}%")
    
    return metrics_df, comparison_df, forecast_df


if __name__ == "__main__":
    metrics, comparison, forecasts = main()
