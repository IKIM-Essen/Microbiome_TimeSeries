import argparse

import os
import sys

# Add the parent directory to sys.path to enable importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print(os.getcwd())

from src.preprocessing.read import create_complete_df
from src.preprocessing.scaling import split_data


def main():

    parser = argparse.ArgumentParser(
        description="Create a complete dataframe with timeseries, metadata, and taxa data.",
        epilog="Example usage: python preprocessing.py --timeseries PATH/TO/TIMESERIES --metadata PATH/TO/METADATA --taxa PATH/TO/TAXA --output PATH/TO/OUTPUT"
    )

    parser.add_argument("--timeseries", type=str, default = "data/timeseries.tsv",help="Path to the timeseries data file (CSV format).")
    parser.add_argument("--metadata", type=str, default = "data/metadata.tsv", help="Path to the metadata file (CSV format).")
    parser.add_argument("--taxa", type=str, default = "data/taxa.tsv", help="Path to the taxa data file (CSV format).")
    parser.add_argument("--include_metadata", type = bool, default=False, help="Whether to include metadata in the output dataframe. Default is False.")
    parser.add_argument("--output", type=str, default="results/tables/complete_df.csv", help="Path to the output file (CSV format). If not provided, the output will be saved as 'results/tables/complete_df.csv'.")
    parser.add_argument("--split-data", action="store_true", help="Also scale and split the complete dataset after creation.")

    args = parser.parse_args()

    complete_df, metadata, number_taxa, dic_TargTax = create_complete_df(
        args.timeseries,
        args.metadata,
        args.taxa,
        args.include_metadata,
        args.output
    )

    if args.split_data:
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            complete_df,
            number_taxa
        )
        print("Data split completed:")
        print(f"X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"X_val={X_val.shape}, y_val={y_val.shape}")
        print(f"X_test={X_test.shape}, y_test={y_test.shape}")


if __name__ == "__main__":
    main()