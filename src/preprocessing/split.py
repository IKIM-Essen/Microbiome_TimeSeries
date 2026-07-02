from itertools import count
import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

from src.preprocessing.scaling import scale_date

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
LOG_DIR = os.path.join(ROOT_DIR, "logs", "preprocessing")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "split.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)


def split_data(
    complete,
    num_taxa,
    scaler_path,
    train_percentage,
    val_percentage,
    split_sizes_path="results/intermediate/split_sizes.pkl",
):
    logger.info("Splitting data for %s taxa", num_taxa)
    scaled_data, scaler = scale_date(complete, scaler_path)
    count = sum(
        not col.startswith("Target") and not col.startswith("Time")
        for col in complete.columns
    )
    X = scaled_data.reshape(
        scaled_data.shape[0], scaled_data.shape[1]
    )  # Reshape for LSTM input
    y = scaled_data[:, 0 : num_taxa + count]
    # Split the data into training and testing sets
    train_size = int(len(X) * train_percentage)
    val_size = int(len(X) * (train_percentage + val_percentage))
    X_train, X_val, X_test = X[0:train_size], X[train_size:val_size], X[val_size:]
    y_train, y_val, y_test = y[0:train_size], y[train_size:val_size], y[val_size:]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    logger.info(
        "Split sizes X_train=%s y_train=%s X_val=%s y_val=%s X_test=%s y_test=%s",
        X_train.shape,
        y_train.shape,
        X_val.shape,
        y_val.shape,
        X_test.shape,
        y_test.shape,
    )

    # Save split sizes for later use
    split_sizes = {"train_size": train_size, "val_size": val_size}
    os.makedirs(os.path.dirname(split_sizes_path), exist_ok=True)
    dump(split_sizes, split_sizes_path)
    logger.info("Saved split sizes to %s", split_sizes_path)

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_without_scaling(
    scaled_data,
    num_taxa,
    train_percentage,
    val_percentage,
    split_sizes_path="results/intermediate/split_sizes.pkl",
):
    logger.info("Splitting data for %s taxa", num_taxa)

    count = sum(
        not col.startswith("Target") and not col.startswith("Time")
        for col in scaled_data.columns
    )
    X = scaled_data.reshape(
        scaled_data.shape[0], scaled_data.shape[1]
    )  # Reshape for LSTM input
    y = scaled_data[:, 0 : num_taxa + count]
    # Split the data into training and testing sets
    train_size = int(len(X) * train_percentage)
    val_size = int(len(X) * (train_percentage + val_percentage))
    X_train, X_val, X_test = X[0:train_size], X[train_size:val_size], X[val_size:]
    y_train, y_val, y_test = y[0:train_size], y[train_size:val_size], y[val_size:]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    logger.info(
        "Split sizes X_train=%s y_train=%s X_val=%s y_val=%s X_test=%s y_test=%s",
        X_train.shape,
        y_train.shape,
        X_val.shape,
        y_val.shape,
        X_test.shape,
        y_test.shape,
    )

    # Save split sizes for later use
    split_sizes = {"train_size": train_size, "val_size": val_size}
    os.makedirs(os.path.dirname(split_sizes_path), exist_ok=True)
    dump(split_sizes, split_sizes_path)
    logger.info("Saved split sizes to %s", split_sizes_path)

    return X_train, y_train, X_val, y_val, X_test, y_test

def split_indices(n_samples, train_percentage=0.7, val_percentage=0.1):
    train_end = int(n_samples * train_percentage)
    val_end = int(n_samples * (train_percentage + val_percentage))
    
    idx_train = np.arange(0, train_end)
    idx_val   = np.arange(train_end, val_end)
    idx_test  = np.arange(val_end, n_samples)
    
    return idx_train, idx_val, idx_test

def split_data_attention(complete, num_taxa, scaler_path, metadata_woT, train_percentage, val_percentage):
    logger.info("Splitting data for %s taxa", num_taxa)
    scaled_data, scaler = scale_date(complete, scaler_path)
    X_bact = scaled_data[:, num_taxa:(num_taxa + lag_targets)]
    X_bact = X_bact.reshape(X_bact.shape[0], 1, X_bact.shape[1])

    # Convert metadata to numpy and reshape for time dimension
    X_meta = np.array(metadata_woT, dtype=np.float32)
    X_meta = X_meta.reshape(X_meta.shape[0], 1, X_meta.shape[1])

    # Ensure bacterial data is numeric
    X_bact = scaled_data[:, num_taxa:(num_taxa + lag_targets)]
    X_bact = X_bact.reshape(X_bact.shape[0], 1, X_bact.shape[1])

    # Ensure target is numeric
    y = scaled_data[:, 0:num_taxa]

    # Generate indices
    idx_train, idx_val, idx_test = split_indices(len(X_bact), train_percentage, val_percentage)

    # Split bacterial data
    X_bact_train = X_bact[idx_train]
    X_bact_val   = X_bact[idx_val]
    X_bact_test  = X_bact[idx_test]

    # Split metadata
    X_meta_train = X_meta[idx_train]
    X_meta_val   = X_meta[idx_val]
    X_meta_test  = X_meta[idx_test]

    # Split targets
    y_train = y[idx_train]
    y_val   = y[idx_val]
    y_test  = y[idx_test]

    return X_bact_train, y_train, X_bact_val, y_val, X_bact_test, y_test, X_meta_train, X_meta_val, X_meta_test
