"""
Quick Runner Script for Prophet Forecasting
============================================
Run this file to execute the full Prophet pipeline.

Usage:
    python run_prophet.py
    
    or 
    
    python run_prophet.py --visualize  (to also generate plots)
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("\n>>> Starting Prophet Cash Flow Forecasting...\n")
    
    # Step 1: Run forecasting
    from prophet_forecasting import run_prophet_pipeline
    results, forecasts, metrics = run_prophet_pipeline()
    
    # Step 2: Generate visualizations (optional)
    if "--visualize" in sys.argv or "-v" in sys.argv:
        print("\n" + "=" * 70)
        print("\n>>> Generating visualizations...\n")
        from prophet_visualization import generate_all_visualizations
        generate_all_visualizations()
    else:
        print("\nTip: Run with --visualize flag to generate plots")
        print("   Example: python run_prophet.py --visualize")
    
    print("\n[DONE] All done!")
    return results, forecasts, metrics


if __name__ == "__main__":
    main()

