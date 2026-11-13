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
    logger.info("Inversely scaling data with scaler %s", scaler_path)
    scaler = load(scaler_path)
    original_data = scaler.inverse_transform(scaled_data)
    logger.info("Inversely scaled data shape %s", original_data.shape)
    return original_data


def inverse_scale_data_attention(scaled_data, scaler_path, num_taxa):
    logger.info("Inversely scaling data with scaler %s", scaler_path)
    scaler = load(scaler_path)
    original_data = scaler.inverse_transform(scaled_data)[:, 0:num_taxa]
    logger.info("Inversely scaled data shape %s", original_data.shape)
    return original_data
