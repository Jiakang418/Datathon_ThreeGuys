"""
Phase 1: Data Engineering for Weekly Cash Flow Forecasting

This script:
- Loads raw transaction data and mapping tables
- Cleans and enriches the data (dates, categories, countries, flow types)
- Engineers weekly features (week-ending Sunday, operating/financing cash flows)
- Aggregates to a weekly time series per country
- Fills missing country-week combinations with zeros
- Saves the result to `Steve/processed_weekly_cashflow.csv`
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # project root: Datathon_ThreeGuys
RAW_DIR = BASE_DIR / "Raw_Dataset"

MAIN_FILE = RAW_DIR / "Datathon Dataset - Data - Main.csv"
CATEGORY_LINKAGE_FILE = RAW_DIR / "Datathon Dataset - Others - Category Linkage.csv"
COUNTRY_MAPPING_FILE = RAW_DIR / "Datathon Dataset - Others - Country Mapping.csv"

CLEANED_OUTPUT_FILE = BASE_DIR / "Steve" / "cleaned_transactions.csv"
WEEKLY_OUTPUT_FILE = BASE_DIR / "Steve" / "processed_weekly_cashflow.csv"

NOISE_COLUMNS = ["Cost Ctr", "WBS element", "Ref.key (header) 1", "Clrng doc."]

OPERATING_CATEGORIES = [
    "AP",
    "Payroll",
    "Tax payable",
    "Statutory contribution",
    "AR",
    "Netting AP",
    "Netting AR",
]

FINANCING_CATEGORIES = [
    "Loan payment",
    "Interest charges",
    "Dividend Payout",
    "Loan receipt",
]


# --------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------------

def upcoming_sunday(date_series: pd.Series) -> pd.Series:
    """Map each date to the upcoming Sunday (including same day if Sunday)."""
    weekday = date_series.dt.weekday  # Monday=0, Sunday=6
    offset_days = (6 - weekday) % 7
    return date_series + pd.to_timedelta(offset_days, unit="D")


def main() -> None:
    # ------------------------------------------------------------------------------
    # 1. INPUT: LOAD CSV FILES
    # ------------------------------------------------------------------------------
    for f in [MAIN_FILE, CATEGORY_LINKAGE_FILE, COUNTRY_MAPPING_FILE]:
        if not f.exists():
            raise FileNotFoundError(f"Required input file not found: {f}")

    # Read main with low_memory=False to avoid mixed-type warnings
    df_main = pd.read_csv(MAIN_FILE, low_memory=False)
    df_cat = pd.read_csv(CATEGORY_LINKAGE_FILE)
    df_country = pd.read_csv(COUNTRY_MAPPING_FILE)

    # ------------------------------------------------------------------------------
    # 2. CLEANING & SANITIZATION
    # ------------------------------------------------------------------------------

    # 2.1 Drop known noise columns
    for c in NOISE_COLUMNS:
        if c in df_main.columns:
            df_main.drop(columns=c, inplace=True)

    # 2.2 Date parsing
    if "Pstng Date" not in df_main.columns:
        raise KeyError("Expected column 'Pstng Date' is missing from main dataset.")
    df_main["Pstng Date"] = pd.to_datetime(df_main["Pstng Date"], errors="coerce")
    df_main = df_main.dropna(subset=["Pstng Date"])

    # 2.3 Impute Category Index with mode
    if "Category Index" in df_main.columns and not df_main["Category Index"].dropna().empty:
        mode_val = df_main["Category Index"].mode().iloc[0]
        df_main["Category Index"] = df_main["Category Index"].fillna(mode_val)

    # 2.4 Fix Category label typo
    if "Category" not in df_main.columns:
        raise KeyError("Expected column 'Category' is missing from main dataset.")
    df_main["Category"] = df_main["Category"].replace({"Non Netting AP": "Non-Netting AP"})

    # Ensure Amount in USD is numeric
    if "Amount in USD" not in df_main.columns:
        raise KeyError("Expected column 'Amount in USD' is missing from main dataset.")
    df_main["Amount in USD"] = pd.to_numeric(
        df_main["Amount in USD"], errors="coerce"
    ).fillna(0.0)

    # ------------------------------------------------------------------------------
    # 3. MERGING & ENRICHMENT
    # ------------------------------------------------------------------------------

    # 3.1 Country Names merge
    # Main file uses 'Name' for entity code (e.g. TW10); mapping uses 'Code' and 'Country'
    if "Name" not in df_main.columns:
        raise KeyError("Expected 'Name' column in main dataset for country mapping join.")
    if "Code" not in df_country.columns or "Country" not in df_country.columns:
        raise KeyError(
            "Expected 'Code' and 'Country' columns in country mapping dataset."
        )

    df_country_renamed = df_country.rename(columns={"Code": "Name"})
    df_main = df_main.merge(
        df_country_renamed[["Name", "Country"]],
        on="Name",
        how="left",
    )
    df_main = df_main.rename(columns={"Country": "Country_Name"})

    # 3.2 Flow Types merge from Category Linkage
    # Category linkage columns: 'Category Names', 'ID', 'Category', 'Cat Order'
    # Here 'Category Names' = transaction category, 'Category' = flow type (Inflow/Outflow)
    if "Category Names" not in df_cat.columns or "Category" not in df_cat.columns:
        raise KeyError(
            "Expected 'Category Names' and 'Category' columns in Category Linkage dataset."
        )

    flow_map = df_cat.set_index("Category Names")["Category"].to_dict()
    df_main["Flow_Type"] = df_main["Category"].map(flow_map)

    # ------------------------------------------------------------------------------
    # 4. FEATURE ENGINEERING
    # ------------------------------------------------------------------------------

    # 4.1 Week_Ending_Date: upcoming Sunday
    df_main["Week_Ending_Date"] = upcoming_sunday(df_main["Pstng Date"])

    # 4.2 Cash Flow buckets
    df_main["Operating_CF"] = np.where(
        df_main["Category"].isin(OPERATING_CATEGORIES),
        df_main["Amount in USD"],
        0.0,
    )
    df_main["Financing_CF"] = np.where(
        df_main["Category"].isin(FINANCING_CATEGORIES),
        df_main["Amount in USD"],
        0.0,
    )

    # ------------------------------------------------------------------------------
    # 5. SAVE 1: CLEANED TRANSACTION-LEVEL DATA (PRE-AGGREGATION)
    # ------------------------------------------------------------------------------

    # This is the cleaned & enriched transaction-level dataset (dashboard / anomalies).
    merged_df = df_main.copy()
    CLEANED_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(CLEANED_OUTPUT_FILE, index=False)
    print(
        f"Saved cleaned_transactions.csv (Row count: {len(merged_df)}) "
        f"to {CLEANED_OUTPUT_FILE}"
    )

    # ------------------------------------------------------------------------------
    # 6. AGGREGATION & RESAMPLING (WEEKLY TIME SERIES)
    # ------------------------------------------------------------------------------

    group_cols = ["Country_Name", "Week_Ending_Date"]

    df_grouped = (
        df_main.groupby(group_cols, dropna=False)
        .agg(
            {
                "Amount in USD": "sum",
                "Operating_CF": "sum",
                "Financing_CF": "sum",
            }
        )
        .rename(
            columns={
                "Amount in USD": "Net_Cash_Flow",
                "Operating_CF": "Operating_Cash_Flow",
                "Financing_CF": "Financing_Cash_Flow",
            }
        )
        .reset_index()
    )

    if df_grouped.empty:
        raise ValueError("Aggregated dataframe is empty; check input data.")

    # Build full country-week grid
    min_week = df_grouped["Week_Ending_Date"].min()
    max_week = df_grouped["Week_Ending_Date"].max()
    all_weeks = pd.date_range(start=min_week, end=max_week, freq="W-SUN")
    all_countries = df_grouped["Country_Name"].drop_duplicates()

    full_index = pd.MultiIndex.from_product(
        [all_countries, all_weeks],
        names=["Country_Name", "Week_Ending_Date"],
    )

    df_final = (
        df_grouped.set_index(["Country_Name", "Week_Ending_Date"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
    )

    # For clarity with your snippet naming, treat df_final as weekly_df
    weekly_df = df_final

    # ------------------------------------------------------------------------------
    # 7. OUTPUT: WEEKLY TIME SERIES
    # ------------------------------------------------------------------------------

    print("Final weekly cashflow dataframe (head):")
    print(weekly_df.head())
    print("\nFinal dataframe shape (rows, columns):", weekly_df.shape)

    WEEKLY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    weekly_df.to_csv(WEEKLY_OUTPUT_FILE, index=False)
    print(
        f"Saved processed_weekly_cashflow.csv (Row count: {len(weekly_df)}) "
        f"to {WEEKLY_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()


