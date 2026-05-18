import argparse

import os
import sys

# Add the parent directory to sys.path to enable importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print(os.getcwd())

from src.visualization.plot_original_timeseries import time_series_analysis_plot

def main():
    parser = argparse.ArgumentParser(
        description="Plot original time series data for specified bacteria.",
        epilog="Example usage: python visualization.py --dataframe PATH/TO/COMPLETE_DF.csv --output PATH/TO/OUTPUT.png --bacteria PATH/TO/BACTERIA_MAPPING.pkl"
    )

    parser.add_argument("--dataframe", type=str, default="results/tables/complete_df.csv", help="Path to the complete dataframe CSV file.")
    parser.add_argument("--output", type=str, default="results/figures/original_timeseries.png", help="Path to save the output plot (PNG format).")
    parser.add_argument("--bacteria", type=str, default="results/intermediate/dic_TargTax.pkl", help="Path to the bacteria mapping pickle file.")

    args = parser.parse_args()

    time_series_analysis_plot(
        args.dataframe,
        args.output,
        args.bacteria
    )


if __name__ == "__main__":
    main()
