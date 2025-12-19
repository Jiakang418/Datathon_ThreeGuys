import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("\n>>> Starting ARIMA Cash Flow Forecasting...\n")
    
    from arima_forecasting import run_arima_pipeline
    results, forecasts, metrics = run_arima_pipeline()
    
    # Save validation comparison CSV
    import pandas as pd
    output_dir = Path(__file__).parent / "arima_results"
    output_dir.mkdir(exist_ok=True)
    
    comparison_list = []
    for country, result in results.items():
        val_dates = result["val_df"]["Week_Ending_Date"].values
        for i, date in enumerate(val_dates):
            comparison_list.append({
                "Country": country,
                "Week_Ending_Date": pd.Timestamp(date),
                "Actual_Cash_Flow": result["actuals"][i],
                "Predicted_Cash_Flow": result["predictions"][i]  # Fixed: was 'Predictions'
            })
    comparison_df = pd.DataFrame(comparison_list)
    comparison_df.to_csv(output_dir / "arima_actual_vs_predicted.csv", index=False)
    
    # Optional visualizations
    if "--visualize" in sys.argv or "-v" in sys.argv:
        from arima_visualization import generate_all_visualizations
        generate_all_visualizations()
    
    print("\n[DONE] ARIMA pipeline complete!")

if __name__ == "__main__":
    main()
