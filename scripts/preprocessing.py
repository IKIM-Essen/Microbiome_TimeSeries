import argparse
import os
import pickle
import sys

import numpy as np

# Add the parent directory to sys.path to enable importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print(os.getcwd())

from src.preprocessing.read import create_complete_df
from src.preprocessing.split import split_data


def main():

    parser = argparse.ArgumentParser(
        description="Create a complete dataframe with timeseries, metadata, and taxa data.",
        epilog="Example usage: python scripts/preprocessing.py --timeseries PATH/TO/TIMESERIES --metadata PATH/TO/METADATA --taxa PATH/TO/TAXA --output PATH/TO/OUTPUT",
    )

    parser.add_argument(
        "--timeseries",
        type=str,
        default="data/timeseries.tsv",
        help="Path to the timeseries data file (CSV format).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata.tsv",
        help="Path to the metadata file (CSV format).",
    )
    parser.add_argument(
        "--taxa",
        type=str,
        default="data/taxa.tsv",
        help="Path to the taxa data file (CSV format).",
    )
    parser.add_argument(
        "--include-metadata",
        type=bool,
        default=False,
        help="Whether to include metadata in the output dataframe. Default is False.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/tables/complete_df.csv",
        help="Path to the output file (CSV format). If not provided, the output will be saved as 'results/tables/complete_df.csv'.",
    )
    parser.add_argument(
        "--mapping-output",
        type=str,
        default="results/intermediate/dic_TargTax.pkl",
        help="Path to save the target taxa mapping pickle file.",
    )
    parser.add_argument(
        "--split-data",
        action="store_true",
        default=True,
        help="Also scale and split the complete dataset after creation.",
    )
    parser.add_argument(
        "--splits-output",
        type=str,
        default="results/intermediate/splits.npz",
        help="Path to save the split numpy batches.",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="results/models/scaler.pkl",
        help="Path to save the scaler.",
    )
    parser.add_argument(
        "--splits-sizes",
        type=str,
        default="results/intermediate/split_sizes.pkl",
        help="Path to save the split sizes.",
    )

    args = parser.parse_args()

    complete_df, metadata, number_taxa, dic_TargTax = create_complete_df(
        args.timeseries, args.metadata, args.taxa, args.include_metadata, args.output
    )

    os.makedirs(os.path.dirname(args.mapping_output), exist_ok=True)
    with open(args.mapping_output, "wb") as mapping_file:
        pickle.dump(dic_TargTax, mapping_file)
    print(f"Saved target taxa mapping to {args.mapping_output}")

    if args.split_data:
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            complete_df,
            number_taxa,
            args.scaler_path,
            args.splits_sizes,
        )
        os.makedirs(os.path.dirname(args.splits_output), exist_ok=True)
        np.savez_compressed(
            args.splits_output,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        print("Data split completed:")
        print(f"X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"X_val={X_val.shape}, y_val={y_val.shape}")
        print(f"X_test={X_test.shape}, y_test={y_test.shape}")
        print(f"Saved split batches to {args.splits_output}")


if __name__ == "__main__":
    main()
