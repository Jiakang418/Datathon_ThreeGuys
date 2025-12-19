"""
Phase 2: Feature Engineering & Time Series Preparation for Weekly Cash Flow

Input:
    weekly_cashflow.csv
        - Country_Name
        - Week_Ending_Date
        - Net_Cash_Flow
        - Operating_Cash_Flow
        - Financing_Cash_Flow

Output:
    data/processed/weekly_features.csv

Features created:
    - Implied_Investing_CF      = Net_Cash_Flow - (Operating_Cash_Flow + Financing_Cash_Flow)
    - Lag_1, Lag_2, Lag_4       lags of Net_Cash_Flow (1, 2, and 4 weeks)
    - Roll_Mean_4w, Roll_Std_4w 4-week rolling mean/std of Net_Cash_Flow (history-only)
"""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Paths & Loading
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parents[1]  # project root

    # Use the Phase 1 output as input to Phase 2
    input_path = base_dir / "Steve" / "processed_weekly_cashflow.csv"
    output_dir = base_dir / "data" / "processed"
    output_path = output_dir / "weekly_features.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found at {input_path}. "
            "Ensure 'weekly_cashflow.csv' is in the project root."
        )

    df = pd.read_csv(input_path)

    # Basic schema sanity check
    required_cols = [
        "Country_Name",
        "Week_Ending_Date",
        "Net_Cash_Flow",
        "Operating_Cash_Flow",
        "Financing_Cash_Flow",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"weekly_cashflow.csv is missing required columns: {missing}")

    # -------------------------------------------------------------------------
    # 2. Implied Investing Cash Flow
    # -------------------------------------------------------------------------
    df["Implied_Investing_CF"] = (
        df["Net_Cash_Flow"]
        - (df["Operating_Cash_Flow"] + df["Financing_Cash_Flow"])
    )

    # -------------------------------------------------------------------------
    # 3. Date handling: convert and index
    # -------------------------------------------------------------------------
    df["Week_Ending_Date"] = pd.to_datetime(df["Week_Ending_Date"], errors="coerce")
    if df["Week_Ending_Date"].isna().any():
        raise ValueError("Some Week_Ending_Date values could not be parsed to datetime.")

    # Sort and set index by date
    df = df.sort_values(["Country_Name", "Week_Ending_Date"])
    df = df.set_index("Week_Ending_Date")

    # -------------------------------------------------------------------------
    # 4. Lag features on Net_Cash_Flow
    # -------------------------------------------------------------------------
    # We create lags per country to avoid leakage across entities.
    group = df.groupby("Country_Name")["Net_Cash_Flow"]
    df["Lag_1"] = group.shift(1)
    df["Lag_2"] = group.shift(2)
    df["Lag_4"] = group.shift(4)  # roughly one month back

    # -------------------------------------------------------------------------
    # 5. Rolling statistics (4-week history)
    # -------------------------------------------------------------------------
    # Use shifted series so rolling window only uses past data (no look-ahead).
    shifted = group.shift(1)
    df["Roll_Mean_4w"] = shifted.groupby(df["Country_Name"]).rolling(window=4).mean().reset_index(level=0, drop=True)
    df["Roll_Std_4w"] = shifted.groupby(df["Country_Name"]).rolling(window=4).std().reset_index(level=0, drop=True)

    # Optional: drop initial rows with NaNs in lag/rolling features
    df_features = df.dropna(
        subset=["Lag_1", "Lag_2", "Lag_4", "Roll_Mean_4w", "Roll_Std_4w"]
    ).copy()

    # -------------------------------------------------------------------------
    # 6. Output
    # -------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=True)  # index = Week_Ending_Date

    print("Phase 2 feature engineering complete.")
    print(f"Saved weekly_features.csv to: {output_path}")
    print("Resulting shape:", df_features.shape)


if __name__ == "__main__":
    main()


