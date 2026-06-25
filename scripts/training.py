import argparse
import logging
import os
import sys

import numpy as np

# Add the parent directory to sys.path so imports from src work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print(os.getcwd())

from src.model_building.create_models import fit_model

# Setup logging for this script
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(ROOT_DIR, "..", "logs", "training")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "training.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Train model on previously preprocessed data",
            epilog="Example usage: python scripts/training.py ",
        )

        parser.add_argument(
            "--splits-input",
            type=str,
            default="results/intermediate/splits.npz",
            help="Path to the saved split numpy batches.",
        )
        parser.add_argument(
            "--path",
            type=str,
            default="results/models/",
            help="Path to save the trained model.",
        )
        parser.add_argument(
            "--model-architecture",
            type=str,
            default=None,
            help="Model architecture to train (tcn_lstm, lstm, attention). If omitted, will read from config/profile.yaml",
        )

        args = parser.parse_args()
        logger.info("Starting model training with splits from %s", args.splits_input)

        # Load preprocessed training and validation splits from the saved npz file
        splits = np.load(args.splits_input)
        X_train = splits["X_train"]
        y_train = splits["y_train"]
        X_val = splits["X_val"]
        y_val = splits["y_val"]

        print(f"Loaded split batches from {args.splits_input}")
        print(f"X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"X_val={X_val.shape}, y_val={y_val.shape}")
        logger.info("Loaded split batches from %s", args.splits_input)
        logger.info(
            "X_train=%s, y_train=%s, X_val=%s, y_val=%s",
            X_train.shape,
            y_train.shape,
            X_val.shape,
            y_val.shape,
        )

        # Train according to requested model architecture
        fit_model(
            X_train,
            y_train,
            X_val,
            y_val,
            X_train.shape[2],
            args.path,
            model_architecture=args.model_architecture,
        )
        logger.info("Model training completed successfully")
    except FileNotFoundError as e:
        logger.error("File not found error: %s", str(e), exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(
            "An unexpected error occurred during training: %s", str(e), exc_info=True
        )
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
