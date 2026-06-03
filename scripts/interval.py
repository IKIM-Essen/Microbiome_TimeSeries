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
from src.utils.config import extract_species
#from src.evaluation.outlier import find_interval_violations

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Create an output for the prediction interval and it's potential violations.",
        epilog="Example usage: python scripts/interval.py --num-models 50 --splits-input results/intermediate/splits.npz --scaler results/models/scaler.pkl --tcn-path results/models/tcn_model.h5 --lstm-path results/models/lstm_model.h5 --output results/tables/prediction_interval.pkl"
    )

    parser.add_argument("--num-models", type=int, default = 5, help="Number of models to train and include in the ensemble.")
    parser.add_argument("--splits-input", type=str, default="results/intermediate/splits.npz", help="Path to the saved split numpy batches.")
    parser.add_argument("--scaler", type = str, default="results/models/scaler.pkl", help="Path to the saved scaler pickle file.")
    parser.add_argument("--tcn-path", type=str, default="results/models/tcn_model.h5", help="Path to the saved TCN model file.")
    parser.add_argument("--lstm-path", type=str, default="results/models/lstm_model.h5", help="Path to the saved LSTM model file.")
    parser.add_argument("--output", type = str, default="results/tables/prediction_interval.pkl", help="Path to the saved prediction interval file.")

    

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
    logger.info("Successfully loaded splits. X_train shape: %s, y_train shape: %s", X_train.shape, y_train.shape)

    df = pd.read_csv("data/taxa.tsv", header = None, index_col = None)
    
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
        args.lstm_path
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save prediction interval to output file
    logger.info("Saving prediction interval to %s", args.output)
    with open(args.output, 'wb') as f:
        pickle.dump(prediction_interval, f)
    logger.info("Prediction interval successfully saved")



if __name__ == "__main__":
    main()