import argparse
import os
import sys

import numpy as np

# Add the parent directory to sys.path to enable importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print(os.getcwd())

from src.model_building.create_models import fit_model

def main():
    parser = argparse.ArgumentParser(
        description="Train model on previously preprocessed data",
        epilog="Example usage: python scripts/training.py "
    )

    parser.add_argument("--splits-input", type=str, default="results/intermediate/splits.npz", help="Path to the saved split numpy batches.")
    parser.add_argument("--tcn-path", type=str, default="results/models/tcn_model.h5", help="Path to save the trained TCN model.")
    parser.add_argument("--lstm-path", type=str, default="results/models/lstm_model.h5", help="Path to save the trained LSTM model.")

    args = parser.parse_args()

    splits = np.load(args.splits_input)
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]

    print(f"Loaded split batches from {args.splits_input}")
    print(f"X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"X_val={X_val.shape}, y_val={y_val.shape}")

    fit_model(
        X_train,
        y_train,
        X_val,
        y_val,
        X_train.shape[2],
        args.tcn_path,
        args.lstm_path,
    )


if __name__ == "__main__":
    main()