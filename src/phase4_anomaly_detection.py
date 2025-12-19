"""
Phase 4: Dual-Layer Anomaly Detection

This script performs anomaly detection at two levels:
1. Transaction-Level: Z-score based detection within Entity-Category groups
2. Structural-Level: Isolation Forest detection on weekly cash flow patterns

Output Files:
- outputs/anomalies_transaction_level.csv - Individual transaction anomalies
- outputs/anomalies_structural_level.csv - Weekly structural anomalies
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")


def calculate_z_score(series: pd.Series) -> pd.Series:
    """
    Calculate Z-scores for a series.
    
    Formula: Z = (x - μ) / σ
    """
    mean = series.mean()
    std = series.std()
    
    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index)
    
    return (series - mean) / std


def detect_transaction_anomalies(df: pd.DataFrame, z_threshold: float = 3.5) -> pd.DataFrame:
    """
    Detect transaction-level anomalies using Z-scores.
    
    Args:
        df: Transaction dataframe with columns: Name (Entity), Category, Amount in USD
        z_threshold: Z-score threshold for flagging anomalies (default 3.5)
    
    Returns:
        DataFrame containing only anomalous transactions
    """
    print("\n[Part 1] Detecting Transaction-Level Anomalies...")
    print(f"  Total transactions: {len(df):,}")
    
    # Ensure required columns exist
    if "Name" not in df.columns:
        raise KeyError("Column 'Name' (Entity) not found in transaction data")
    if "Category" not in df.columns:
        raise KeyError("Column 'Category' not found in transaction data")
    if "Amount in USD" not in df.columns:
        raise KeyError("Column 'Amount in USD' not found in transaction data")
    
    # Calculate Z-scores within each Entity-Category group
    df = df.copy()
    df["Z_Score"] = df.groupby(["Name", "Category"])["Amount in USD"].transform(calculate_z_score)
    
    # Flag anomalies (absolute Z-score > threshold)
    df["Is_Anomaly"] = np.abs(df["Z_Score"]) > z_threshold
    
    # Filter for anomalies only
    anomalies = df[df["Is_Anomaly"]].copy()
    
    print(f"  Anomalies detected: {len(anomalies):,} ({len(anomalies)/len(df)*100:.2f}%)")
    
    # Select output columns
    output_cols = []
    col_mapping = {
        "Pstng Date": "Date",
        "Name": "Entity",
        "Category": "Category",
        "Amount in USD": "Amount_USD",
        "Z_Score": "Z_Score",
        "Country_Name": "Country",
        "Week_Ending_Date": "Week_Ending_Date"
    }
    
    for col, alias in col_mapping.items():
        if col in anomalies.columns:
            output_cols.append(col)
    
    # Rename columns for output
    anomalies_output = anomalies[output_cols].copy()
    anomalies_output = anomalies_output.rename(columns=col_mapping)
    
    # Sort by absolute Z-score (most extreme first)
    anomalies_output = anomalies_output.sort_values("Z_Score", key=abs, ascending=False)
    
    return anomalies_output


def detect_structural_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detect structural anomalies in weekly cash flow using Isolation Forest.
    
    Args:
        df: Weekly dataframe with columns: Country_Name, Week_Ending_Date, Net_Cash_Flow, Roll_Mean_4w, Roll_Std_4w
        contamination: Expected proportion of anomalies (default 0.05 = 5%)
    
    Returns:
        DataFrame with anomaly flags and scores for all weeks
    """
    print("\n[Part 2] Detecting Structural-Level Anomalies...")
    print(f"  Total weekly records: {len(df):,}")
    
    # Ensure required columns exist
    required_cols = ["Country_Name", "Week_Ending_Date", "Net_Cash_Flow", "Roll_Mean_4w", "Roll_Std_4w"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    # Feature columns for Isolation Forest
    feature_cols = ["Net_Cash_Flow", "Roll_Mean_4w", "Roll_Std_4w"]
    
    all_results = []
    
    # Process each country independently
    for country in df["Country_Name"].unique():
        country_df = df[df["Country_Name"] == country].copy()
        country_df = country_df.sort_values("Week_Ending_Date")
        
        if len(country_df) < 10:
            print(f"  [!] Skipping {country}: insufficient data ({len(country_df)} weeks)")
            continue
        
        # Prepare features
        X = country_df[feature_cols].fillna(0.0).values
        
        # Check for constant features (would cause issues)
        if np.std(X, axis=0).sum() == 0:
            print(f"  [!] Skipping {country}: constant features")
            continue
        
        try:
            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            iso_forest.fit(X)
            
            # Predict anomalies (-1 = anomaly, 1 = normal)
            anomaly_flags = iso_forest.predict(X)
            
            # Get anomaly scores (lower = more anomalous)
            anomaly_scores = iso_forest.score_samples(X)
            
            # Store results
            for i, (idx, row) in enumerate(country_df.iterrows()):
                all_results.append({
                    "Country": country,
                    "Week_Ending_Date": row["Week_Ending_Date"],
                    "Net_Cash_Flow": row["Net_Cash_Flow"],
                    "Anomaly_Score": anomaly_scores[i],
                    "Anomaly_Flag": anomaly_flags[i],
                    "Roll_Mean_4w": row["Roll_Mean_4w"],
                    "Roll_Std_4w": row["Roll_Std_4w"]
                })
            
            n_anomalies = (anomaly_flags == -1).sum()
            print(f"  [OK] {country}: {n_anomalies} anomalies detected ({n_anomalies/len(country_df)*100:.1f}%)")
            
        except Exception as e:
            print(f"  [!] Error processing {country}: {e}")
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("  [!] No results generated. Check data availability.")
        return pd.DataFrame()
    
    # Sort by anomaly score (most anomalous first)
    results_df = results_df.sort_values("Anomaly_Score")
    
    print(f"\n  Total anomalies detected: {(results_df['Anomaly_Flag'] == -1).sum():,}")
    
    return results_df


def main():
    """Main execution function."""
    # Configuration
    base_dir = Path(__file__).resolve().parents[1]
    
    # Input paths (check multiple possible locations)
    possible_transaction_paths = [
        base_dir / "data" / "processed" / "cleaned_transactions.csv",
        base_dir / "Preprocessed Dataset" / "cleaned_transactions.csv",
        base_dir / "Steve" / "cleaned_transactions.csv"
    ]
    
    possible_weekly_paths = [
        base_dir / "data" / "processed" / "weekly_features.csv",
        base_dir / "data" / "model_dataset" / "weekly_features.csv"
    ]
    
    # Find existing transaction file
    transaction_path = None
    for path in possible_transaction_paths:
        if path.exists():
            transaction_path = path
            break
    
    # Find existing weekly file
    weekly_path = None
    for path in possible_weekly_paths:
        if path.exists():
            weekly_path = path
            break
    
    # Output directory
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  PHASE 4: DUAL-LAYER ANOMALY DETECTION")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Part 1: Transaction-Level Anomalies
    # -------------------------------------------------------------------------
    if transaction_path is None or not transaction_path.exists():
        print(f"\n[!] Transaction file not found in any expected location.")
        print("    Checked paths:")
        for path in possible_transaction_paths:
            print(f"      - {path}")
        print("    Skipping transaction-level anomaly detection.")
        transaction_anomalies = pd.DataFrame()
    else:
        print(f"\n[Loading] Transaction data from: {transaction_path}")
        transactions_df = pd.read_csv(transaction_path, parse_dates=["Pstng Date"], low_memory=False)
        
        # Ensure Amount in USD is numeric
        transactions_df["Amount in USD"] = pd.to_numeric(
            transactions_df["Amount in USD"], errors="coerce"
        )
        transactions_df = transactions_df.dropna(subset=["Amount in USD"])
        
        transaction_anomalies = detect_transaction_anomalies(transactions_df, z_threshold=3.5)
        
        # Save transaction anomalies
        if len(transaction_anomalies) > 0:
            output_file = output_dir / "anomalies_transaction_level.csv"
            transaction_anomalies.to_csv(output_file, index=False)
            print(f"\n  [OK] Saved transaction anomalies to: {output_file}")
            print(f"       Rows: {len(transaction_anomalies):,}")
        else:
            print("\n  [OK] No transaction anomalies detected.")
    
    # -------------------------------------------------------------------------
    # Part 2: Structural-Level Anomalies
    # -------------------------------------------------------------------------
    if weekly_path is None or not weekly_path.exists():
        print(f"\n[!] Weekly features file not found in any expected location.")
        print("    Checked paths:")
        for path in possible_weekly_paths:
            print(f"      - {path}")
        print("    Skipping structural-level anomaly detection.")
        structural_anomalies = pd.DataFrame()
    else:
        print(f"\n[Loading] Weekly features from: {weekly_path}")
        weekly_df = pd.read_csv(weekly_path, parse_dates=["Week_Ending_Date"])
        
        structural_anomalies = detect_structural_anomalies(weekly_df, contamination=0.05)
        
        # Save structural anomalies
        if len(structural_anomalies) > 0:
            output_file = output_dir / "anomalies_structural_level.csv"
            structural_anomalies.to_csv(output_file, index=False)
            print(f"\n  [OK] Saved structural anomalies to: {output_file}")
            print(f"       Rows: {len(structural_anomalies):,}")
            
            # Print summary by country
            print("\n  Anomaly Summary by Country:")
            print("-" * 70)
            for country in structural_anomalies["Country"].unique():
                country_data = structural_anomalies[structural_anomalies["Country"] == country]
                n_anomalies = (country_data["Anomaly_Flag"] == -1).sum()
                total = len(country_data)
                print(f"    {country}: {n_anomalies}/{total} weeks flagged ({n_anomalies/total*100:.1f}%)")
        else:
            print("\n  [OK] No structural anomalies detected.")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ANOMALY DETECTION COMPLETE!")
    print("=" * 70)
    
    print("\nOutput Files:")
    if len(transaction_anomalies) > 0:
        print(f"  1. anomalies_transaction_level.csv - {len(transaction_anomalies):,} anomalous transactions")
    else:
        print("  1. anomalies_transaction_level.csv - No anomalies detected")
    
    if len(structural_anomalies) > 0:
        n_structural = (structural_anomalies["Anomaly_Flag"] == -1).sum()
        print(f"  2. anomalies_structural_level.csv - {n_structural:,} anomalous weeks")
    else:
        print("  2. anomalies_structural_level.csv - No anomalies detected")
    
    return transaction_anomalies, structural_anomalies


if __name__ == "__main__":
    transaction_results, structural_results = main()

