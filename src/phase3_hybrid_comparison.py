"""
Phase 3: Hybrid Model Comparison

This script implements and compares three hybrid forecasting architectures:
1. Hybrid_Naive: Naive (Last Observation) + XGBoost
2. Hybrid_ARIMA: ARIMA(1,1,1) + XGBoost
3. Hybrid_Prophet: Prophet + XGBoost

Each model follows the hybrid workflow:
- Base model captures trend/seasonality
- XGBoost predicts residuals using engineered features
- Walk-forward forecasting with recursive feature updates

Output: 9 CSV files (3 per model type)
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------
# HELPER FUNCTIONS
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


def update_features(history_net_cf: np.ndarray, new_value: float) -> dict:
    """
    Update lag and rolling features after adding a new predicted value.
    
    Args:
        history_net_cf: Historical Net_Cash_Flow values (numpy array)
        new_value: The newly predicted Net_Cash_Flow value
    
    Returns:
        Dictionary with updated feature values
    """
    # Append new value
    updated_series = np.append(history_net_cf, new_value)
    
    # Calculate lags
    lag_1 = updated_series[-1] if len(updated_series) >= 1 else 0.0
    lag_2 = updated_series[-2] if len(updated_series) >= 2 else 0.0
    lag_4 = updated_series[-4] if len(updated_series) >= 4 else 0.0
    
    # Calculate rolling stats (last 4 values excluding current)
    if len(updated_series) >= 5:
        roll_window = updated_series[-5:-1]
        roll_mean = np.mean(roll_window)
        roll_std = np.std(roll_window) if len(roll_window) > 1 else 0.0
    elif len(updated_series) >= 2:
        roll_window = updated_series[:-1]
        roll_mean = np.mean(roll_window)
        roll_std = np.std(roll_window) if len(roll_window) > 1 else 0.0
    else:
        roll_mean = 0.0
        roll_std = 0.0
    
    return {
        "Lag_1": lag_1,
        "Lag_2": lag_2,
        "Lag_4": lag_4,
        "Roll_Mean_4w": roll_mean,
        "Roll_Std_4w": roll_std
    }


# --------------------------------------------------------------------------------------
# HYBRID MODEL CLASSES
# --------------------------------------------------------------------------------------

class HybridNaiveXGBoost:
    """Hybrid Naive + XGBoost model."""
    
    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.xgb_model = None
        self.last_value = None
        self.residual_std = None
    
    def fit(self, train_df: pd.DataFrame):
        """Train the hybrid model."""
        train_targets = train_df["Net_Cash_Flow"].values
        
        # Base model: Naive (last value)
        self.last_value = train_targets[-1]
        
        # Calculate residuals on training set
        base_predictions = np.roll(train_targets, 1)
        base_predictions[0] = train_targets[0]
        residuals = train_targets - base_predictions
        
        # Calculate residual std for confidence intervals
        self.residual_std = np.std(residuals)
        
        # Prepare features for XGBoost
        X_train = train_df[self.feature_cols].fillna(0.0).values
        y_train = residuals
        
        # Train XGBoost on residuals
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.xgb_model.fit(X_train, y_train)
    
    def predict_base(self, history: np.ndarray) -> float:
        """Generate base model prediction."""
        return history[-1] if len(history) > 0 else 0.0
    
    def predict_residual(self, features: np.ndarray) -> float:
        """Predict residual using XGBoost."""
        if self.xgb_model is None:
            return 0.0
        return self.xgb_model.predict(features.reshape(1, -1))[0]
    
    def get_confidence_interval(self, prediction: float) -> tuple:
        """Get 95% confidence interval."""
        z_score = 1.96
        lower = prediction - (z_score * self.residual_std)
        upper = prediction + (z_score * self.residual_std)
        return lower, upper


class HybridARIMAXGBoost:
    """Hybrid ARIMA + XGBoost model."""
    
    def __init__(self, feature_cols: list, arima_order=(1, 1, 1)):
        self.feature_cols = feature_cols
        self.arima_order = arima_order
        self.xgb_model = None
        self.arima_model = None
        self.residual_std = None
        self.arima_fitted = None
    
    def fit(self, train_df: pd.DataFrame):
        """Train the hybrid model."""
        train_targets = train_df["Net_Cash_Flow"].values
        
        # Base model: ARIMA
        try:
            self.arima_model = ARIMA(train_targets, order=self.arima_order)
            self.arima_fitted = self.arima_model.fit()
            base_predictions = self.arima_fitted.fittedvalues.values
        except Exception as e:
            print(f"    [!] ARIMA failed to converge, using Naive fallback: {e}")
            # Fallback to naive
            base_predictions = np.roll(train_targets, 1)
            base_predictions[0] = train_targets[0]
            self.arima_fitted = None
        
        # Calculate residuals
        residuals = train_targets - base_predictions
        self.residual_std = np.std(residuals)
        
        # Train XGBoost on residuals
        X_train = train_df[self.feature_cols].fillna(0.0).values
        y_train = residuals
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.xgb_model.fit(X_train, y_train)
    
    def predict_base(self, history: np.ndarray) -> float:
        """Generate base model prediction."""
        if self.arima_fitted is None or len(history) < 3:
            # Fallback to naive
            return history[-1] if len(history) > 0 else 0.0
        
        try:
            # Refit ARIMA on current history
            arima_temp = ARIMA(history, order=self.arima_order)
            arima_fitted_temp = arima_temp.fit()
            forecast = arima_fitted_temp.forecast(steps=1)
            return float(forecast.iloc[0])
        except Exception:
            # Fallback to naive
            return history[-1] if len(history) > 0 else 0.0
    
    def predict_residual(self, features: np.ndarray) -> float:
        """Predict residual using XGBoost."""
        if self.xgb_model is None:
            return 0.0
        return self.xgb_model.predict(features.reshape(1, -1))[0]
    
    def get_confidence_interval(self, prediction: float) -> tuple:
        """Get 95% confidence interval."""
        z_score = 1.96
        lower = prediction - (z_score * self.residual_std)
        upper = prediction + (z_score * self.residual_std)
        return lower, upper


class HybridProphetXGBoost:
    """Hybrid Prophet + XGBoost model."""
    
    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.xgb_model = None
        self.prophet_model = None
        self.residual_std = None
        self.train_dates = None
    
    def fit(self, train_df: pd.DataFrame):
        """Train the hybrid model."""
        train_targets = train_df["Net_Cash_Flow"].values
        self.train_dates = train_df["Week_Ending_Date"].values
        
        # Base model: Prophet
        prophet_df = pd.DataFrame({
            "ds": self.train_dates,
            "y": train_targets
        })
        
        try:
            self.prophet_model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=False,
                daily_seasonality=False
            )
            self.prophet_model.fit(prophet_df)
            
            # Get fitted values
            forecast = self.prophet_model.predict(prophet_df)
            base_predictions = forecast["yhat"].values
        except Exception as e:
            print(f"    [!] Prophet failed, using Naive fallback: {e}")
            # Fallback to naive
            base_predictions = np.roll(train_targets, 1)
            base_predictions[0] = train_targets[0]
            self.prophet_model = None
        
        # Calculate residuals
        residuals = train_targets - base_predictions
        self.residual_std = np.std(residuals)
        
        # Train XGBoost on residuals
        X_train = train_df[self.feature_cols].fillna(0.0).values
        y_train = residuals
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.xgb_model.fit(X_train, y_train)
    
    def predict_base(self, history: np.ndarray, history_dates: np.ndarray) -> float:
        """Generate base model prediction."""
        if self.prophet_model is None or len(history) < 3:
            # Fallback to naive
            return history[-1] if len(history) > 0 else 0.0
        
        try:
            # Refit Prophet on current history
            prophet_df = pd.DataFrame({
                "ds": history_dates,
                "y": history
            })
            prophet_temp = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=False,
                daily_seasonality=False
            )
            prophet_temp.fit(prophet_df)
            
            # Forecast next period
            future = prophet_temp.make_future_dataframe(periods=1, freq="W-SUN")
            forecast = prophet_temp.predict(future)
            return float(forecast["yhat"].iloc[-1])
        except Exception:
            # Fallback to naive
            return history[-1] if len(history) > 0 else 0.0
    
    def predict_residual(self, features: np.ndarray) -> float:
        """Predict residual using XGBoost."""
        if self.xgb_model is None:
            return 0.0
        return self.xgb_model.predict(features.reshape(1, -1))[0]
    
    def get_confidence_interval(self, prediction: float) -> tuple:
        """Get 95% confidence interval."""
        z_score = 1.96
        lower = prediction - (z_score * self.residual_std)
        upper = prediction + (z_score * self.residual_std)
        return lower, upper


# --------------------------------------------------------------------------------------
# WALK-FORWARD FORECASTING
# --------------------------------------------------------------------------------------

def walk_forward_forecast(
    model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    is_prophet: bool = False
) -> tuple:
    """
    Perform walk-forward forecasting with recursive feature updates.
    
    Returns:
        Tuple of (predictions, lower_ci, upper_ci)
    """
    predictions = []
    lower_cis = []
    upper_cis = []
    
    # Initialize history (maintain full dataframe for features)
    history_df = train_df.copy()
    history_net_cf = train_df["Net_Cash_Flow"].values.copy()
    history_dates = train_df["Week_Ending_Date"].values.copy() if is_prophet else None
    
    for idx, row in val_df.iterrows():
        # Get current features from last row of history
        if len(history_df) > 0:
            last_row = history_df.iloc[-1]
            
            # Build feature vector using updated lags/rolling from history
            feature_dict = update_features(history_net_cf, history_net_cf[-1])
            feature_vector = np.array([
                feature_dict.get("Lag_1", 0.0),
                feature_dict.get("Lag_2", 0.0),
                feature_dict.get("Lag_4", 0.0),
                feature_dict.get("Roll_Mean_4w", 0.0),
                feature_dict.get("Roll_Std_4w", 0.0),
                last_row.get("Implied_Investing_CF", 0.0)
            ])
        else:
            feature_vector = np.zeros(len(feature_cols))
        
        # Step 1: Generate base forecast
        if is_prophet:
            base_pred = model.predict_base(history_net_cf, history_dates)
        else:
            base_pred = model.predict_base(history_net_cf)
        
        # Step 2: Predict residual
        residual_pred = model.predict_residual(feature_vector)
        
        # Step 3: Final prediction
        final_pred = base_pred + residual_pred
        predictions.append(final_pred)
        
        # Get confidence intervals
        lower, upper = model.get_confidence_interval(final_pred)
        lower_cis.append(lower)
        upper_cis.append(upper)
        
        # Step 4: Update history with predicted value
        history_net_cf = np.append(history_net_cf, final_pred)
        if is_prophet:
            history_dates = np.append(history_dates, row["Week_Ending_Date"])
        
        # Create new row for history dataframe
        new_row = row.copy()
        new_row["Net_Cash_Flow"] = final_pred
        
        # Update lag and rolling features in the new row
        updated_features = update_features(history_net_cf[:-1], final_pred)
        new_row["Lag_1"] = updated_features.get("Lag_1", 0.0)
        new_row["Lag_2"] = updated_features.get("Lag_2", 0.0)
        new_row["Lag_4"] = updated_features.get("Lag_4", 0.0)
        new_row["Roll_Mean_4w"] = updated_features.get("Roll_Mean_4w", 0.0)
        new_row["Roll_Std_4w"] = updated_features.get("Roll_Std_4w", 0.0)
        
        # Append to history dataframe
        history_df = pd.concat([history_df, new_row.to_frame().T], ignore_index=True)
    
    return np.array(predictions), np.array(lower_cis), np.array(upper_cis)


# --------------------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------------------

def main():
    """Main execution function."""
    # Configuration
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "data" / "model_dataset" / "weekly_features.csv"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load data
    df = pd.read_csv(input_path, parse_dates=["Week_Ending_Date"])
    df = df.sort_values(["Country_Name", "Week_Ending_Date"])
    
    print("=" * 70)
    print("  HYBRID MODEL COMPARISON - Phase 3")
    print("=" * 70)
    print(f"\nLoaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Countries: {list(df['Country_Name'].unique())}")
    
    # Feature columns for XGBoost
    feature_cols = [
        "Lag_1",
        "Lag_2",
        "Lag_4",
        "Roll_Mean_4w",
        "Roll_Std_4w",
        "Implied_Investing_CF"
    ]
    
    validation_weeks = 4
    models = {
        "Naive": HybridNaiveXGBoost,
        "ARIMA": HybridARIMAXGBoost,
        "Prophet": HybridProphetXGBoost
    }
    
    # Storage for all results
    all_results = {model_name: {
        "metrics": [],
        "actual_vs_pred": [],
        "future_forecasts": []
    } for model_name in models.keys()}
    
    # Process each country
    for country in df["Country_Name"].unique():
        print(f"\n{'='*70}")
        print(f"Processing Country: {country}")
        print(f"{'='*70}")
        
        country_df = df[df["Country_Name"] == country].copy()
        country_df = country_df.sort_values("Week_Ending_Date")
        
        if len(country_df) < validation_weeks + 10:
            print(f"  [!] Skipping {country}: insufficient data ({len(country_df)} weeks)")
            continue
        
        # Split data
        split_idx = len(country_df) - validation_weeks
        train_df = country_df.iloc[:split_idx].copy()
        val_df = country_df.iloc[split_idx:].copy()
        
        # Train and evaluate each model
        for model_name, ModelClass in models.items():
            print(f"\n  Training {model_name} + XGBoost...")
            
            try:
                # Initialize and train model
                if model_name == "ARIMA":
                    model = ModelClass(feature_cols, arima_order=(1, 1, 1))
                else:
                    model = ModelClass(feature_cols)
                
                model.fit(train_df)
                
                # Walk-forward forecasting on validation set
                is_prophet = (model_name == "Prophet")
                val_predictions, val_lower, val_upper = walk_forward_forecast(
                    model, train_df, val_df, feature_cols, is_prophet=is_prophet
                )
                
                val_actuals = val_df["Net_Cash_Flow"].values
                
                # Calculate metrics
                metrics = calculate_metrics(val_actuals, val_predictions)
                
                # Store metrics
                all_results[model_name]["metrics"].append({
                    "Country": country,
                    "RMSE_USD": metrics["RMSE"],
                    "MAE_USD": metrics["MAE"],
                    "MAPE_percent": metrics["MAPE_percent"],
                    "Train_Weeks": len(train_df),
                    "Validation_Weeks": len(val_df)
                })
                
                # Store actual vs predicted
                for i, (idx, row) in enumerate(val_df.iterrows()):
                    all_results[model_name]["actual_vs_pred"].append({
                        "Country": country,
                        "Week_Ending_Date": row["Week_Ending_Date"],
                        "Actual_Cash_Flow": val_actuals[i],
                        "Predicted_Cash_Flow": val_predictions[i],
                        "Prediction_Lower_95CI": val_lower[i],
                        "Prediction_Upper_95CI": val_upper[i],
                        "Error": val_actuals[i] - val_predictions[i]
                    })
                
                # Future forecasting (1-month and 6-month)
                all_values = country_df["Net_Cash_Flow"].values
                all_dates = country_df["Week_Ending_Date"].values
                last_date = country_df["Week_Ending_Date"].max()
                
                # Retrain on all data for future forecasting
                if model_name == "ARIMA":
                    future_model = ModelClass(feature_cols, arima_order=(1, 1, 1))
                else:
                    future_model = ModelClass(feature_cols)
                future_model.fit(country_df)
                
                # 1-month forecast (4 weeks)
                future_history_net_cf = all_values.copy()
                future_history_dates = all_dates.copy() if is_prophet else None
                
                for weeks_ahead in [4, 26]:
                    horizon_name = "1_month" if weeks_ahead == 4 else "6_month"
                    future_predictions = []
                    future_lower = []
                    future_upper = []
                    
                    temp_history = future_history_net_cf.copy()
                    temp_dates = future_history_dates.copy() if is_prophet else None
                    
                    for week in range(weeks_ahead):
                        # Get features
                        if len(country_df) > 0:
                            last_row = country_df.iloc[-1]
                            feature_dict = update_features(temp_history, temp_history[-1])
                            feature_vector = np.array([
                                feature_dict.get("Lag_1", 0.0),
                                feature_dict.get("Lag_2", 0.0),
                                feature_dict.get("Lag_4", 0.0),
                                feature_dict.get("Roll_Mean_4w", 0.0),
                                feature_dict.get("Roll_Std_4w", 0.0),
                                last_row.get("Implied_Investing_CF", 0.0)
                            ])
                        else:
                            feature_vector = np.zeros(len(feature_cols))
                        
                        # Predict
                        if is_prophet:
                            base_pred = future_model.predict_base(temp_history, temp_dates)
                        else:
                            base_pred = future_model.predict_base(temp_history)
                        residual_pred = future_model.predict_residual(feature_vector)
                        final_pred = base_pred + residual_pred
                        
                        lower, upper = future_model.get_confidence_interval(final_pred)
                        future_predictions.append(final_pred)
                        future_lower.append(lower)
                        future_upper.append(upper)
                        
                        # Update history
                        temp_history = np.append(temp_history, final_pred)
                        if is_prophet:
                            next_date = last_date + pd.Timedelta(days=7 * (week + 1))
                            temp_dates = np.append(temp_dates, next_date)
                    
                    # Generate future dates
                    next_sunday = last_date + pd.Timedelta(days=7)
                    future_dates = pd.date_range(start=next_sunday, periods=weeks_ahead, freq="W-SUN")
                    
                    # Store future forecasts
                    for i, date in enumerate(future_dates):
                        all_results[model_name]["future_forecasts"].append({
                            "Week_Ending_Date": date,
                            "Predicted_Cash_Flow": future_predictions[i],
                            "Prediction_Lower_95CI": future_lower[i],
                            "Prediction_Upper_95CI": future_upper[i],
                            "Country": country,
                            "Target": "Net_Cash_Flow",
                            "Horizon": horizon_name
                        })
                
                print(f"    [OK] RMSE: ${metrics['RMSE']:,.2f} | MAE: ${metrics['MAE']:,.2f} | MAPE: {metrics['MAPE_percent']}%")
                
            except Exception as e:
                print(f"    [!] Error training {model_name} for {country}: {e}")
                continue
    
    # Save all results
    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)
    
    for model_name in models.keys():
        # Save metrics
        metrics_df = pd.DataFrame(all_results[model_name]["metrics"])
        if len(metrics_df) > 0:
            metrics_file = output_dir / f"hybrid_{model_name.lower()}_backtest_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)
            print(f"\n  Saved: {metrics_file.name}")
            print(f"    Average RMSE: ${metrics_df['RMSE_USD'].mean():,.2f}")
        
        # Save actual vs predicted
        comparison_df = pd.DataFrame(all_results[model_name]["actual_vs_pred"])
        if len(comparison_df) > 0:
            comparison_file = output_dir / f"hybrid_{model_name.lower()}_actual_vs_predicted.csv"
            comparison_df.to_csv(comparison_file, index=False)
            print(f"  Saved: {comparison_file.name}")
        
        # Save future forecasts
        forecast_df = pd.DataFrame(all_results[model_name]["future_forecasts"])
        if len(forecast_df) > 0:
            forecast_file = output_dir / f"hybrid_{model_name.lower()}_future_forecasts.csv"
            forecast_df.to_csv(forecast_file, index=False)
            print(f"  Saved: {forecast_file.name}")
    
    print("\n" + "=" * 70)
    print("  HYBRID MODEL COMPARISON COMPLETE!")
    print("=" * 70)
    
    # Print summary comparison
    print("\nModel Performance Summary:")
    print("-" * 70)
    for model_name in models.keys():
        metrics_df = pd.DataFrame(all_results[model_name]["metrics"])
        if len(metrics_df) > 0:
            print(f"\n{model_name} + XGBoost:")
            print(f"  Average RMSE: ${metrics_df['RMSE_USD'].mean():,.2f}")
            print(f"  Average MAE: ${metrics_df['MAE_USD'].mean():,.2f}")
            print(f"  Average MAPE: {metrics_df['MAPE_percent'].mean():.1f}%")


if __name__ == "__main__":
    main()

