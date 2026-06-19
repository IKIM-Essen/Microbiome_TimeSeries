import argparse
import logging
import os
import pickle
import sys

import pandas as pd
import numpy as np

# Add the parent directory to sys.path to enable importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils.config import reshape
from src.visualization.plot_results import plot_taxa_dropdown

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOG_DIR = os.path.join(ROOT_DIR, "logs", "visualization")
os.makedirs(LOG_DIR, exist_ok=True)


def main():

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Plot selectable taxa time series with interval anomalies table.",
        epilog="Example usage: python scripts/plot_taxa_anomalies.py --complete results/tables/complete_df.csv --dic-taxa results/intermediate/dic_TargTax.pkl",
    )
    parser.add_argument(
        "--complete",
        type=str,
        default="results/tables/complete_df.csv",
        help="Path to complete dataframe CSV",
    )
    parser.add_argument(
        "--dic-taxa",
        type=str,
        default="results/intermediate/dic_TargTax.pkl",
        help="Pickle mapping Target->Taxa",
    )
    parser.add_argument(
        "--anomalies",
        type=str,
        default="results/tables/prediction_interval_anomalies.tsv",
        help="TSV of anomalies",
    )
    parser.add_argument(
        "--prediction-interval",
        type=str,
        default="results/tables/prediction_interval.tsv",
        help="TSV of prediction interval with columns species,timepoint,lower,upper,mean",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="results/intermediate/predictions.npz",
        help="NPZ file containing pred_train, pred_val, pred_test arrays",
    )
    parser.add_argument(
        "--split-sizes",
        type=str,
        default="results/intermediate/split_sizes.pkl",
        help="Optional pickle file containing train_size and val_size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures/plot_taxa_anomalies.html",
        help="Output HTML path",
    )
    args = parser.parse_args()
    plot_taxa_dropdown(
        args.complete,
        args.dic_taxa,
        args.anomalies,
        args.prediction_interval,
        args.predictions,
        args.output,
    )


if __name__ == "__main__":
    main()
