"""
Phase 3: Hybrid Forecasting Models for Weekly Cash Flow Prediction

This script implements three hybrid forecasting models:
1. Naive + XGBoost
2. ARIMA + XGBoost
3. Prophet + XGBoost

Each model follows the hybrid workflow:
- Base model captures trend/seasonality
- XGBoost predicts residuals using engineered features
- Walk-forward forecasting with recursive feature updates
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb


class HybridForecaster:
    """Base class for hybrid forecasting models."""

    def __init__(self, country_data: pd.DataFrame, test_size: int = 4):
        """
        Initialize forecaster for a single country.

        Args:
            country_data: DataFrame with Week_Ending_Date as index, sorted chronologically
            test_size: Number of weeks to hold out as test set
        """
        self.country_data = country_data.sort_index()
        self.test_size = test_size
        self.train_data = None
        self.test_data = None
        self.base_model = None
        self.residual_model = None
        self.feature_cols = [
            "Lag_1",
            "Lag_2",
            "Lag_4",
            "Roll_Mean_4w",
            "Roll_Std_4w",
            "Implied_Investing_CF",
            "Operating_Cash_Flow",
            "Financing_Cash_Flow",
        ]

    def _split_data(self):
        """Split data into train (all but last test_size weeks) and test (last test_size weeks)."""
        n = len(self.country_data)
        split_idx = n - self.test_size
        self.train_data = self.country_data.iloc[:split_idx].copy()
        self.test_data = self.country_data.iloc[split_idx:].copy()

    def _update_features(self, history: pd.DataFrame, new_value: float) -> pd.Series:
        """
        Update lag and rolling features after adding a new predicted value.

        Args:
            history: DataFrame with historical Net_Cash_Flow values (including new prediction)
            new_value: The newly predicted Net_Cash_Flow value

        Returns:
            Series with updated feature values for the next time step
        """
        # Append new value to history
        net_cf_series = history["Net_Cash_Flow"].copy()
        net_cf_series = pd.concat([net_cf_series, pd.Series([new_value])])

        # Calculate new lags
        lag_1 = net_cf_series.iloc[-1] if len(net_cf_series) >= 1 else np.nan
        lag_2 = net_cf_series.iloc[-2] if len(net_cf_series) >= 2 else np.nan
        lag_4 = net_cf_series.iloc[-4] if len(net_cf_series) >= 4 else np.nan

        # Calculate rolling stats (using last 4 values, excluding current)
        if len(net_cf_series) >= 5:
            roll_window = net_cf_series.iloc[-5:-1]  # Last 4 excluding current
            roll_mean = roll_window.mean()
            roll_std = roll_window.std()
        elif len(net_cf_series) >= 2:
            roll_window = net_cf_series.iloc[:-1]  # All but current
            roll_mean = roll_window.mean()
            roll_std = roll_window.std() if len(roll_window) > 1 else 0.0
        else:
            roll_mean = np.nan
            roll_std = np.nan

        # Get other features from the last row of history
        last_row = history.iloc[-1]
        operating_cf = last_row.get("Operating_Cash_Flow", 0.0)
        financing_cf = last_row.get("Financing_Cash_Flow", 0.0)
        
        # Recalculate Implied_Investing_CF if Operating/Financing are available
        if not pd.isna(operating_cf) and not pd.isna(financing_cf):
            implied_investing = new_value - (operating_cf + financing_cf)
        else:
            implied_investing = last_row.get("Implied_Investing_CF", 0.0)
        
        features = pd.Series(
            {
                "Lag_1": lag_1,
                "Lag_2": lag_2,
                "Lag_4": lag_4,
                "Roll_Mean_4w": roll_mean,
                "Roll_Std_4w": roll_std,
                "Implied_Investing_CF": implied_investing,
                "Operating_Cash_Flow": operating_cf if not pd.isna(operating_cf) else 0.0,
                "Financing_Cash_Flow": financing_cf if not pd.isna(financing_cf) else 0.0,
            }
        )
        return features

    def fit(self):
        """Train base model and residual XGBoost model on training data."""
        raise NotImplementedError("Subclasses must implement fit()")

    def predict_walk_forward(self) -> np.ndarray:
        """
        Perform walk-forward prediction on test set with recursive feature updates.

        Returns:
            Array of predictions for test period
        """
        predictions = []
        history = self.train_data.copy()

        for idx, test_row in self.test_data.iterrows():
            # Step A: Generate base forecast for t+1
            base_pred = self._predict_base_next(history)

            # Step B: Calculate features for t+1 based on current history
            # Features should reflect the state at time t (last known point)
            if len(history) > 0:
                # Get features from last row (these are already calculated for time t)
                # But we need to ensure they're valid for predicting t+1
                feature_values = []
                for feat in self.feature_cols:
                    if feat in history.columns:
                        val = history.iloc[-1][feat]
                        feature_values.append(val if not pd.isna(val) else 0.0)
                    else:
                        feature_values.append(0.0)
                features = np.array(feature_values).reshape(1, -1)
            else:
                features = np.zeros((1, len(self.feature_cols)))

            # Predict residual using XGBoost
            residual_pred = self.residual_model.predict(features)[0]

            # Step C: Final prediction = Base + Residual
            final_pred = base_pred + residual_pred
            predictions.append(final_pred)

            # Step D: Update history with predicted value for next iteration
            # Create a new row with predicted Net_Cash_Flow and copy other columns
            new_row = test_row.copy()
            new_row["Net_Cash_Flow"] = final_pred
            
            # Convert to DataFrame and append
            new_row_df = pd.DataFrame([new_row])
            new_row_df.index = [idx]  # Use the test date as index
            history = pd.concat([history, new_row_df])

            # Recalculate and update features for the new row (for next iteration)
            updated_features = self._update_features(history, final_pred)
            for feat in self.feature_cols:
                if feat in updated_features.index and feat in history.columns:
                    history.loc[idx, feat] = updated_features[feat]

        return np.array(predictions)

    def _predict_base_next(self, history: pd.DataFrame) -> float:
        """Generate base model prediction for next time step."""
        raise NotImplementedError("Subclasses must implement _predict_base_next()")


class NaiveXGBoostForecaster(HybridForecaster):
    """Naive (last value) + XGBoost hybrid model."""

    def fit(self):
        """Train naive base model and XGBoost residual model."""
        self._split_data()

        # Base model: Naive (last value = next prediction)
        # No training needed for naive model

        # Calculate residuals on training set
        train_targets = self.train_data["Net_Cash_Flow"].values
        base_predictions = np.roll(train_targets, 1)  # Shift by 1 (t-1 predicts t)
        base_predictions[0] = train_targets[0]  # First value uses itself
        residuals = train_targets - base_predictions

        # Prepare features for XGBoost
        X_train = self.train_data[self.feature_cols].fillna(0.0).values
        y_train = residuals

        # Train XGBoost on residuals
        self.residual_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.residual_model.fit(X_train, y_train)

    def _predict_base_next(self, history: pd.DataFrame) -> float:
        """Naive: return last value."""
        if len(history) == 0:
            return 0.0
        return history["Net_Cash_Flow"].iloc[-1]


class ARIMAXGBoostForecaster(HybridForecaster):
    """ARIMA + XGBoost hybrid model."""

    def __init__(self, *args, arima_order=(1, 1, 1), **kwargs):
        """
        Initialize ARIMA+XGBoost forecaster.

        Args:
            arima_order: ARIMA(p, d, q) order tuple
        """
        super().__init__(*args, **kwargs)
        self.arima_order = arima_order

    def fit(self):
        """Train ARIMA base model and XGBoost residual model."""
        self._split_data()

        # Base model: ARIMA
        train_series = self.train_data["Net_Cash_Flow"]
        self.base_model = ARIMA(train_series, order=self.arima_order)
        self.base_model = self.base_model.fit()

        # Calculate residuals on training set
        base_predictions = self.base_model.fittedvalues
        residuals = train_series.values - base_predictions.values

        # Prepare features for XGBoost
        X_train = self.train_data[self.feature_cols].fillna(0.0).values
        y_train = residuals

        # Train XGBoost on residuals
        self.residual_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.residual_model.fit(X_train, y_train)

    def _predict_base_next(self, history: pd.DataFrame) -> float:
        """ARIMA: forecast next value."""
        if len(history) == 0:
            return 0.0

        # Refit ARIMA on current history (or use last fitted model)
        try:
            arima_model = ARIMA(history["Net_Cash_Flow"], order=self.arima_order)
            arima_fitted = arima_model.fit()
            forecast = arima_fitted.forecast(steps=1)
            return float(forecast.iloc[0])
        except Exception:
            # Fallback to naive if ARIMA fails
            return history["Net_Cash_Flow"].iloc[-1]


class ProphetXGBoostForecaster(HybridForecaster):
    """Prophet + XGBoost hybrid model."""

    def fit(self):
        """Train Prophet base model and XGBoost residual model."""
        self._split_data()

        # Base model: Prophet (requires 'ds' and 'y' columns)
        prophet_df = pd.DataFrame(
            {
                "ds": self.train_data.index,
                "y": self.train_data["Net_Cash_Flow"].values,
            }
        )
        self.base_model = Prophet(weekly_seasonality=True, yearly_seasonality=False)
        self.base_model.fit(prophet_df)

        # Calculate residuals on training set
        train_forecast = self.base_model.predict(prophet_df)
        base_predictions = train_forecast["yhat"].values
        residuals = self.train_data["Net_Cash_Flow"].values - base_predictions

        # Prepare features for XGBoost
        X_train = self.train_data[self.feature_cols].fillna(0.0).values
        y_train = residuals

        # Train XGBoost on residuals
        self.residual_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.residual_model.fit(X_train, y_train)

    def _predict_base_next(self, history: pd.DataFrame) -> float:
        """Prophet: forecast next value."""
        if len(history) == 0:
            return 0.0

        # Create prophet dataframe with current history
        prophet_df = pd.DataFrame(
            {
                "ds": history.index,
                "y": history["Net_Cash_Flow"].values,
            }
        )

        # Refit Prophet on current history
        try:
            prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=False)
            prophet_model.fit(prophet_df)

            # Forecast next period
            future = prophet_model.make_future_dataframe(periods=1, freq="W")
            forecast = prophet_model.predict(future)
            return float(forecast["yhat"].iloc[-1])
        except Exception:
            # Fallback to naive if Prophet fails
            return history["Net_Cash_Flow"].iloc[-1]


def main():
    """Main execution function."""
    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "data" / "model_dataset" / "weekly_features.csv"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, parse_dates=["Week_Ending_Date"], index_col="Week_Ending_Date")
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Countries: {df['Country_Name'].unique()}")

    # -------------------------------------------------------------------------
    # 2. Process Each Country
    # -------------------------------------------------------------------------
    results = {}
    test_size = 4

    for country in df["Country_Name"].unique():
        print(f"\n{'='*60}")
        print(f"Processing Country: {country}")
        print(f"{'='*60}")

        country_data = df[df["Country_Name"] == country].copy()

        # Ensure we have enough data
        if len(country_data) < test_size + 10:
            print(f"  Skipping {country}: insufficient data ({len(country_data)} rows)")
            continue

        # Initialize forecasters
        naive_forecaster = NaiveXGBoostForecaster(country_data, test_size=test_size)
        arima_forecaster = ARIMAXGBoostForecaster(country_data, test_size=test_size)
        prophet_forecaster = ProphetXGBoostForecaster(country_data, test_size=test_size)

        # Train models
        print("  Training Naive+XGBoost...")
        naive_forecaster.fit()

        print("  Training ARIMA+XGBoost...")
        try:
            arima_forecaster.fit()
        except Exception as e:
            print(f"    ARIMA training failed: {e}, skipping...")
            continue

        print("  Training Prophet+XGBoost...")
        try:
            prophet_forecaster.fit()
        except Exception as e:
            print(f"    Prophet training failed: {e}, skipping...")
            continue

        # Walk-forward predictions
        print("  Generating walk-forward predictions...")
        naive_preds = naive_forecaster.predict_walk_forward()
        arima_preds = arima_forecaster.predict_walk_forward()
        prophet_preds = prophet_forecaster.predict_walk_forward()

        # Get actuals
        actuals = naive_forecaster.test_data["Net_Cash_Flow"].values

        # Calculate RMSE
        naive_rmse = np.sqrt(mean_squared_error(actuals, naive_preds))
        arima_rmse = np.sqrt(mean_squared_error(actuals, arima_preds))
        prophet_rmse = np.sqrt(mean_squared_error(actuals, prophet_preds))

        print(f"\n  RMSE Results for {country}:")
        print(f"    Naive+XGBoost:  {naive_rmse:.2f}")
        print(f"    ARIMA+XGBoost:  {arima_rmse:.2f}")
        print(f"    Prophet+XGBoost: {prophet_rmse:.2f}")

        results[country] = {
            "actuals": actuals,
            "naive_preds": naive_preds,
            "arima_preds": arima_preds,
            "prophet_preds": prophet_preds,
            "test_dates": naive_forecaster.test_data.index,
            "naive_rmse": naive_rmse,
            "arima_rmse": arima_rmse,
            "prophet_rmse": prophet_rmse,
        }

    # -------------------------------------------------------------------------
    # 3. Aggregate Results and Plot
    # -------------------------------------------------------------------------
    if not results:
        print("\nNo results to plot. Check data availability.")
        return

    # Create comparison plot
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 5 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for idx, (country, result) in enumerate(results.items()):
        ax = axes[idx]
        test_dates = result["test_dates"]

        ax.plot(test_dates, result["actuals"], "ko-", label="Actual", linewidth=2, markersize=8)
        ax.plot(
            test_dates,
            result["naive_preds"],
            "b--",
            label=f"Naive+XGBoost (RMSE={result['naive_rmse']:.2f})",
            linewidth=2,
        )
        ax.plot(
            test_dates,
            result["arima_preds"],
            "r--",
            label=f"ARIMA+XGBoost (RMSE={result['arima_rmse']:.2f})",
            linewidth=2,
        )
        ax.plot(
            test_dates,
            result["prophet_preds"],
            "g--",
            label=f"Prophet+XGBoost (RMSE={result['prophet_rmse']:.2f})",
            linewidth=2,
        )

        ax.set_title(f"Hybrid Model Comparison - {country}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Week Ending Date", fontsize=12)
        ax.set_ylabel("Net Cash Flow (USD)", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plot_path = output_dir / "model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n{'='*60}")
    print(f"Plot saved to: {plot_path}")
    print(f"{'='*60}")

    # Print summary RMSE
    print("\n" + "=" * 60)
    print("SUMMARY RMSE ACROSS ALL COUNTRIES:")
    print("=" * 60)
    for country, result in results.items():
        print(f"{country}:")
        print(f"  Naive+XGBoost:  {result['naive_rmse']:.2f}")
        print(f"  ARIMA+XGBoost:  {result['arima_rmse']:.2f}")
        print(f"  Prophet+XGBoost: {result['prophet_rmse']:.2f}")
        print()


if __name__ == "__main__":
    main()

