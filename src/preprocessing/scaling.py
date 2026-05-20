import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

from src.utils.config import extract_species

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
LOG_DIR = os.path.join(ROOT_DIR, "logs", "preprocessing")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "scaling.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(file_handler)


def scale_date(complete, scaler_path="results/models/scaler.pkl"):
    logger.info("Scaling complete dataframe with scaler path %s", scaler_path)
    complete = complete.dropna()
    # Normalize the data
    #scaler = load(open('Milwaukee/LSTM/scaler.pkl', 'rb'))
    scaler = MinMaxScaler(feature_range=(0, 1))
    complete_woT = complete.drop(['Time'], axis=1)
    scaled_data = scaler.fit_transform(complete_woT)
    dump(scaler, open(str(scaler_path), 'wb'))
    logger.info("Scaled data shape %s and saved scaler", scaled_data.shape)
    return scaled_data, scaler

def scale_data_with_scaler(complete, scaler_path="results/models/scaler.pkl", scaler_update="results/models/scaler_updated.pkl"):
    logger.info("Scaling complete dataframe with existing scaler %s", scaler_path)
    complete = complete.dropna()
    scaler = load(open(str(scaler_path), 'rb'))
    complete_woT = complete.drop(['Time'], axis=1)
    scaled_data = scaler.transform(complete_woT)
    dump(scaler, open(str(scaler_update), 'wb'))
    logger.info("Scaled data using existing scaler, saved updated scaler to %s", scaler_update)
    return scaled_data, scaler

def inverse_scale_data(scaled_data, scaler_path="results/models/scaler.pkl"):
    logger.info("Inversely scaling data with scaler %s", scaler_path)
    scaler = load(open(str(scaler_path), 'rb'))
    original_data = scaler.inverse_transform(scaled_data)
    logger.info("Inversely scaled data shape %s", original_data.shape)
    return original_data

def split_data(complete, num_taxa):
    logger.info("Splitting data for %s taxa", num_taxa)
    scaled_data, scaler = scale_date(complete)
    X = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1])  # Reshape for LSTM input
    y = scaled_data[:, 0:num_taxa]
    # Split the data into training and testing sets
    train_size = int(len(X) * 0.7)
    val_size = int(len(X)*0.8)
    X_train, X_val, X_test = X[0:train_size], X[train_size:val_size], X[val_size:]
    y_train, y_val, y_test = y[0:train_size], y[train_size:val_size], y[val_size:]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    logger.info("Split sizes X_train=%s y_train=%s X_val=%s y_val=%s X_test=%s y_test=%s",
                X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test
