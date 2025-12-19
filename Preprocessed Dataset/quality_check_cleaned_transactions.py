"""
Quality checks for cleaned_transactions.csv

Checks performed:
- Duplicate row count
- Missing values in Amount in USD
- Sanity check on Indonesia (ID) transaction magnitudes
"""

import pandas as pd
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    csv_path = base_dir / "Preprocessed Dataset" / "cleaned_transactions.csv"

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    print("--- QUALITY CHECK REPORT ---")

    # 1. Check for duplicates
    dupes = df.duplicated().sum()
    print(f"Duplicate Rows: {dupes} (Should be 0)")

    # 2. Check for missing USD values
    missing_amt = df["Amount in USD"].isna().sum()
    print(f"Missing Amounts: {missing_amt} (Should be 0)")

    # 3. Verify Currency Conversion (Sanity Check)
    max_id_val = df[df["Country_Name"] == "ID"]["Amount in USD"].abs().max()
    print(f"Max Transaction for Indonesia: ${max_id_val:,.2f}")

    if max_id_val > 1_000_000_000:
        print("WARNING: Value is too high! You might be using IDR instead of USD.")
    else:
        print("SUCCESS: Values look like valid USD figures.")


if __name__ == "__main__":
    main()


