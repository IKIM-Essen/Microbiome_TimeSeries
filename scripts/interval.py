import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

# Add the parent directory to sys.path to enable importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print(os.getcwd())

from src.evaluation.ensemble import predict_interval
from src.evaluation.outlier import find_interval_anomalies
from src.utils.config import extract_species, prediction_interval_to_df


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Create an output for the prediction interval and it's potential anomalies.",
        epilog="Example usage: python scripts/interval.py --num-models 50 --splits-input results/intermediate/splits.npz --scaler results/models/scaler.pkl --tcn-path results/models/tcn_model.h5 --lstm-path results/models/lstm_model.h5 --output results/tables/prediction_interval.tsv",
    )

    parser.add_argument(
        "--num-models",
        type=int,
        default=5,
        help="Number of models to train and include in the ensemble.",
    )
    parser.add_argument(
        "--splits-input",
        type=str,
        default="results/intermediate/splits.npz",
        help="Path to the saved split numpy batches.",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="results/models/scaler.pkl",
        help="Path to the saved scaler pickle file.",
    )
    parser.add_argument(
        "--tcn-path",
        type=str,
        default="results/models/tcn_model.h5",
        help="Path to the saved TCN model file.",
    )
    parser.add_argument(
        "--lstm-path",
        type=str,
        default="results/models/lstm_model.h5",
        help="Path to the saved LSTM model file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/tables/prediction_interval.tsv",
        help="Path to the saved prediction interval TSV file.",
    )
    parser.add_argument(
        "--complete-input",
        type=str,
        default="results/tables/complete_df.csv",
        help="Path to the complete dataframe CSV file used for violation detection.",
    )
    parser.add_argument(
        "--dic-taxa",
        type=str,
        default="results/intermediate/dic_TargTax.pkl",
        help="Path to the target-to-taxa mapping pickle file.",
    )
    parser.add_argument(
        "--anomalies-output",
        type=str,
        default="results/tables/prediction_interval_anomalies.tsv",
        help="Path to save detected interval anomalies as TSV.",
    )

    args = parser.parse_args()
    logger.info("Starting evaluation with arguments: %s", args)

    logger.info("Loading splits from %s", args.splits_input)
    splits = np.load(args.splits_input)
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    logger.info(
        "Successfully loaded splits. X_train shape: %s, y_train shape: %s",
        X_train.shape,
        y_train.shape,
    )

    df = pd.read_csv("data/taxa.tsv", header=None, index_col=None)

    species = extract_species(df)

    logger.info("Computing prediction interval with %s models", args.num_models)
    prediction_interval = predict_interval(
        args.num_models,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        args.scaler,
        species,
        args.tcn_path,
        args.lstm_path,
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert prediction interval into TSV-friendly dataframe
    prediction_interval_df = prediction_interval_to_df(prediction_interval, species)

    # Save prediction interval to TSV
    logger.info("Saving prediction interval TSV to %s", args.output)
    prediction_interval_df.to_csv(args.output, sep="\t", index=False)
    logger.info("Prediction interval successfully saved")

    # Run interval violation detection
    logger.info("Loading complete dataframe from %s", args.complete_input)
    complete_df = pd.read_csv(args.complete_input)
    logger.info("Loading taxa mapping from %s", args.dic_taxa)
    with open(args.dic_taxa, "rb") as mapping_file:
        dic_taxa = pickle.load(mapping_file)

    val_start = X_train.shape[0] + X_val.shape[0]
    logger.info("Detecting anomalies starting at row index %s", val_start)
    anomalies_df = find_interval_anomalies(
        complete=complete_df,
        prediction_intervals=prediction_interval,
        dic_taxa=dic_taxa,
        val_start=val_start,
        csv_path=None,
    )
    logger.info("Found %s anomalies", len(anomalies_df))

    anomalies_output_dir = os.path.dirname(args.anomalies_output)
    if anomalies_output_dir:
        os.makedirs(anomalies_output_dir, exist_ok=True)
    logger.info("Saving interval anomalies to %s", args.anomalies_output)
    anomalies_df.to_csv(args.anomalies_output, sep="\t", index=False)
    logger.info("Interval anomalies successfully saved")


if __name__ == "__main__":
    main()
