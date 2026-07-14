import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

# Add the parent directory to sys.path to enable importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.evaluation_metrics import combine_metrics

print(os.getcwd())


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Create an output for the evaluation metrics.",
        epilog="Example usage: python scripts/evaluation.py",
    )
    parser.add_argument(
        "--prediction-results",
        type=str,
        default="results/intermediate/predictions.npz",
        help="Path to the saved split numpy batches of predictions.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="results/intermediate/splits.npz",
        help="Path to the saved split numpy batches of original values.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/tables/evaluation_metrics.tsv",
        help="Path to the saved evaluation metrics.",
    )
    parser.add_argument(
        "--model-architecture",
        type=str,
        default=None,
        help="Model architecture used for prediction (tcn_lstm, lstm, attention). If omitted, reads config/profile.yaml",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=None,
        help="Path to the scaler file used for inverse scaling.",
    )

    args = parser.parse_args()
    logger.info("Starting evaluation with arguments: %s", args)

    logger.info("Loading prediction results from %s", args.prediction_results)
    predictions = np.load(args.prediction_results)
    if args.model_architecture != "attention":
        print("No attention")
        y_pred_train = predictions["pred_train"]
        y_pred_val = predictions["pred_val"]
        y_pred_test = predictions["pred_test"]

        y_pred_train = np.squeeze(y_pred_train, axis=1)
        y_pred_test = np.squeeze(y_pred_test, axis=1)

        actual = np.load(args.splits)
        y_train = actual["y_train"]
        y_val = actual["y_val"]
        y_test = actual["y_test"]
    
    elif args.model_architecture == "attention":
        print("attention")
        actual = np.load(args.splits)
        X_train = actual["X_bact_train"]
        X_test = actual["X_bact_test"]
        y_pred_train = predictions["pred_train"]
        y_pred_val = predictions["pred_val"]
        y_pred_test = predictions["pred_test"]

        y_train = actual["y_train"]
        y_val = actual["y_val"]
        y_test = actual["y_test"]

    evaluation_metrics = combine_metrics(
        X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, args.output, args.scaler_path, args.model_architecture
    )


if __name__ == "__main__":
    main()
