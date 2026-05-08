import argparse

import os
import sys

# Add the parent directory to sys.path to enable importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print(os.getcwd())

from src.preprocessing.read import create_complete_df


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

    args = parser.parse_args()

    create_complete_df(
        args.timeseries,
        args.metadata,
        args.taxa,
        args.include_metadata,
        args.output
    )


if __name__ == "__main__":
    main()