import argparse
import logging
import os
import sys

import numpy as np

# Add the parent directory to sys.path to enable importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print(os.getcwd())

from src.model_building.predict import predict

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Compute raw model predictions and save them as a NumPy archive.",
        epilog="Example usage: python scripts/prediction.py --splits-input results/intermediate/splits.npz --tcn-path results/models/tcn_model.h5 --lstm-path results/models/lstm_model.h5 --output results/intermediate/predictions.npz"
    )

    parser.add_argument("--splits-input", type=str, default="results/intermediate/splits.npz", help="Path to the saved split numpy batches.")
    parser.add_argument("--tcn-path", type=str, default="results/models/tcn_model.h5", help="Path to the saved TCN model file.")
    parser.add_argument("--lstm-path", type=str, default="results/models/lstm_model.h5", help="Path to the saved LSTM model file.")
    parser.add_argument("--scaler-path", type=str, default="results/models/scaler.pkl", help="Path to the saved scaler pickle file (unused by raw prediction).")
    parser.add_argument("--output", type=str, default="results/intermediate/predictions.npz", help="Path to save the model prediction arrays.")

    args = parser.parse_args()
    logger.info("Starting prediction with arguments: %s", args)

    logger.info("Loading splits from %s", args.splits_input)
    splits = np.load(args.splits_input)
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]
    logger.info("Successfully loaded splits. X_train shape: %s, X_val shape: %s, X_test shape: %s", X_train.shape, X_val.shape, X_test.shape)

    pred_train, pred_val, pred_test = predict(
        X_train,
        X_val,
        X_test,
        args.tcn_path,
        args.lstm_path,
        args.scaler_path,
        args.output
    )
    
    logger.info("Prediction results shapes: pred_train=%s, pred_val=%s, pred_test=%s", pred_train.shape, pred_val.shape, pred_test.shape)
    print(f"Saved prediction arrays to {args.output}")



if __name__ == "__main__":
    main()