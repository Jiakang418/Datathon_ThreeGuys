"""
Runner Script for Naive Forecasting
====================================
Executes the Naive forecasting pipeline and generates visualizations.

Usage:
    python run_naive.py

Author: Marcus
Date: 2025-12-19
"""

from naive_forecasting import run_naive_pipeline
from naive_visualization import generate_all_visualizations


def main():
    print("\n" + "=" * 70)
    print("  NAIVE FORECASTING - COMPLETE PIPELINE")
    print("=" * 70)
    
    # Step 1: Run forecasting
    print("\n>>> PHASE 1: Running Naive Forecasting...")
    results, forecasts, metrics = run_naive_pipeline()
    
    # Step 2: Generate visualizations
    print("\n>>> PHASE 2: Generating Visualizations...")
    generate_all_visualizations()
    
    print("\n" + "=" * 70)
    print("  ALL DONE! Naive forecasting pipeline complete.")
    print("=" * 70)
    print("\nCheck the following folders for outputs:")
    print("  - marcus/naive_results/  (CSV files)")
    print("  - marcus/naive_plots/    (PNG charts)")


if __name__ == "__main__":
    main()

