import argparse
import logging
import os
import sys

import numpy as np

# Add the parent directory to sys.path to enable importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print(os.getcwd())

from src.model_building.retraining_models import retrain_model
from src.visualization.plot_results import plot_taxa_dropdown

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Retraining already trained models on new data",
        epilog="Example usage: python scripts/retraining.py --splits-input results/intermediate/splits.npz --tcn-path results/models/tcn_model.h5 --lstm-path results/models/lstm_model.h5 --output results/intermediate/predictions.npz"
    )

    parser.add_argument("--timeseries", type=str, default="results/data/timeseries.tsv", help="Path to the timeseries of the bacterial genera.")
    parser.add_argument("--taxa", type=str, default="results/data/taxa.tsv", help="Path to the list of general taxa.")
    parser.add_argument("--metadata", type=str, default="results/data/metadata.tsv", help="Path to the metadata belonging to the timeseries.")
    parser.add_argument("--tcn-path", type=str, default="results/models/tcn_model.h5", help="Path to the saved TCN model file.")
    parser.add_argument("--lstm-path", type=str, default="results/models/lstm_model.h5", help="Path to the saved LSTM model file.")
    parser.add_argument("--scaler-path", type=str, default="results/models/scaler.pkl", help="Path to the saved scaler pickle file (unused by raw prediction).")
    parser.add_argument("--prediction-interval", type=str, default="results/tables/prediction_interval.tsv", help="Path to save the model's prediction interval.")
    parser.add_argument("--predictions", type=str, default=None, help="Optional NPZ file containing pred_train, pred_val, pred_test arrays")
    parser.add_argument("--anomalies-output", type=str, default="results/tables/prediction_interval_anomalies.tsv", help="Path to save the model's interval anomalies.")
    parser.add_argument("--plot-results", type=str, default="results/tables/plot_taxa_anomalies.html", help="Path to plot of results.")
    parser.add_argument("--train-percentage", type=float, default=0.5)
    parser.add_argument("--val-percentage", type=float, default=0.1)

    args = parser.parse_args()
    logger.info("Starting prediction with arguments: %s", args)

    logger.info("Loading splits from %s", args.splits_input)
    splits = np.load(args.splits_input)
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]
    logger.info("Successfully loaded splits. X_train shape: %s, X_val shape: %s, X_test shape: %s", X_train.shape, X_val.shape, X_test.shape)

    anomalies = retrain_model(
        args.timeseries,
        args.taxa,
        args.metadata,
        args.tcn_path,
        args.lstm_path,
        args.scaler_path,
        args.prediction_interval,
        args.predictions,
        args.anomalies_output,
        args.train_percentage,
        args.val_percentage
    )

    plot_taxa_dropdown(args.timeseries, "results/intermediate/dic_TargTax.pkl", anomalies, args.prediction_interval, args.predictions, args.plot_results)



if __name__ == "__main__":
    main()