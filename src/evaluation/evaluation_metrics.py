import math
import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.config import reshape
from src.preprocessing.scaling import inverse_scale_data

# Configure module-level logger to write metric computations to a file under logs/predicting
LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logs", "predicting")
)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "evaluation_metrics.log")
logger = logging.getLogger(__name__)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)


def mae(actual, predicted):
    """Mean Absolute Error (MAE) computed on inverse-scaled data.

    Parameters
    - actual: array-like of ground truth values (scaled)
    - predicted: array-like of predicted values (scaled)

    Returns
    - float: MAE on original scale
    """
    # Inverse scale inputs back to the original space before computing metrics
    inverse = inverse_scale_data(predicted)
    actual_inv = inverse_scale_data(actual)
    value = mean_absolute_error(actual_inv, inverse)
    logger.info(
        "MAE computed: %f | actual_inv shape=%s | pred_inv shape=%s",
        value,
        getattr(actual_inv, "shape", None),
        getattr(inverse, "shape", None),
    )
    return value


def rmse(actual, predicted):
    """Root Mean Squared Error (RMSE) computed on inverse-scaled data.

    Parameters
    - actual: array-like of ground truth values (scaled)
    - predicted: array-like of predicted values (scaled)

    Returns
    - float: RMSE on original scale
    """
    inverse = inverse_scale_data(predicted)
    actual_inv = inverse_scale_data(actual)
    value = math.sqrt(mean_squared_error(actual_inv, inverse))
    logger.info("RMSE computed: %f", value)
    return value


def nrmse(actual, predicted):
    """Normalized RMSE (NRMSE) computed on inverse-scaled data.

    Normalization is by the standard deviation of the predicted (inverse-scaled) values.

    Parameters
    - actual: array-like of ground truth values (scaled)
    - predicted: array-like of predicted values (scaled)

    Returns
    - float: NRMSE on original scale
    """
    inverse = inverse_scale_data(predicted)
    actual_inv = inverse_scale_data(actual)
    rmse_val = math.sqrt(mean_squared_error(actual_inv, inverse))
    std = np.std(inverse)
    nrmse_val = rmse_val / std if std != 0 else float("inf")
    logger.info("NRMSE computed: %f | rmse=%f | std=%f", nrmse_val, rmse_val, std)
    return nrmse_val


def combine_metrics(y_train, y_test, predict_train, predict_test, output_path):
    """Compute common metrics for train and test sets and optionally save to TSV.

    Parameters
    - y_train, y_test: ground truth arrays (scaled)
    - predict_train, predict_test: predicted arrays (scaled)
    - output_path: optional path to save a TSV with the metrics

    Returns
    - dict: calculated metrics
    """
    metrics = {
        "MAE train": mae(y_train, predict_train),
        "RMSE train": rmse(y_train, predict_train),
        "NRMSE train": nrmse(y_train, predict_train),
        "MAE test": mae(y_test, predict_test),
        "RMSE test": rmse(y_test, predict_test),
        "NRMSE test": nrmse(y_test, predict_test),
    }
    logger.info("Combined metrics: %s", metrics)
    # Save to TSV if a path is provided
    if output_path is not None:
        df = pd.DataFrame([metrics])
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(
            "Saved metrics to TSV: %s | columns=%s", output_path, list(df.columns)
        )
    return metrics
