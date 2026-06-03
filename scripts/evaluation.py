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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Create an output for the evaluation metrics.",
        epilog="Example usage: python scripts/evaluation.py"
    )
    parser.add_argument("--prediction-results", type=str, default="results/intermediate/predictions.npz", help="Path to the saved split numpy batches of predictions.")
    parser.add_argument("--splits", type=str, default="results/intermediate/splits.npz", help="Path to the saved split numpy batches of original values.")
    parser.add_argument("--output", type = str, default="results/tables/evaluation_metrics.tsv", help="Path to the saved evaluation metrics.")
    
    args = parser.parse_args()
    logger.info("Starting evaluation with arguments: %s", args)

    logger.info("Loading prediction results from %s", args.prediction_results)
    predictions = np.load(args.prediction_results)
    y_pred_train = predictions["pred_train"]
    y_pred_val = predictions["pred_val"]
    y_pred_test = predictions["pred_test"]

    actual = np.load(args.splits)
    y_train = actual["y_train"]
    y_val = actual["y_val"]
    y_test = actual["y_test"]

    evaluation_metrics = combine_metrics(
        y_train,
        y_test,
        y_pred_train,
        y_pred_test,
        args.output
    )

if __name__ == "__main__":
    main()