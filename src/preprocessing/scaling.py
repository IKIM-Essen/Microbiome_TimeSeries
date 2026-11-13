import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

from src.utils.config import extract_species

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
LOG_DIR = os.path.join(ROOT_DIR, "logs", "preprocessing")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "scaling.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)


def scale_date(complete, scaler_path="results/models/scaler.pkl"):
    """Fit scaling and save it to the full filename given by scaler_path."""
    logger.info("Scaling complete dataframe with scaler path %s", scaler_path)
    complete = complete.dropna()
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    complete_woT = complete.drop(["Time"], axis=1)
    scaled_data = scaler.fit_transform(complete_woT)
    os.makedirs(os.path.dirname(os.path.abspath(scaler_path)), exist_ok=True)
    dump(scaler, scaler_path)
    logger.info("Scaled data shape %s and saved scaler", scaled_data.shape)
    return scaled_data, scaler


def scale_data_with_scaler(complete, scaler_path="results/models/scaler.pkl"):
    logger.info("Scaling complete dataframe with existing scaler %s", scaler_path)
    complete = complete.dropna()
    scaler = load(scaler_path)
    complete_woT = complete.drop(["Time"], axis=1)
    scaled_data = scaler.transform(complete_woT)
    return scaled_data


def inverse_scale_data(scaled_data, scaler_path):
    """Invert scaling for all features or the leading microbial target columns."""
    logger.info("Inversely scaling data with scaler %s", scaler_path)
    scaler = load(scaler_path)
    scaled_data = np.asarray(scaled_data)
    if scaled_data.ndim != 2:
        raise ValueError("Inverse scaling requires a two-dimensional array")
    feature_count = scaled_data.shape[1]
    if not 0 < feature_count <= scaler.n_features_in_:
        raise ValueError("Prediction columns do not match the fitted scaler")
    if feature_count < scaler.n_features_in_:
        # MinMaxScaler transforms columns independently. Padding unused input
        # features allows taxa-only predictions to use the original input scaler.
        padded = np.zeros((len(scaled_data), scaler.n_features_in_))
        padded[:, :feature_count] = scaled_data
        original_data = scaler.inverse_transform(padded)[:, :feature_count]
    else:
        original_data = scaler.inverse_transform(scaled_data)
    logger.info("Inversely scaled data shape %s", original_data.shape)
    return original_data


def inverse_scale_data_attention(scaled_data, scaler_path, num_taxa):
    return inverse_scale_data(scaled_data, scaler_path)[:, :num_taxa]
