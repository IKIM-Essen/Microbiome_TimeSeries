import math
import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.config import reshape, reshape_attention, reshape_original_attention
from src.preprocessing.scaling import inverse_scale_data, inverse_scale_data_attention

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


def mae(actual, predicted, num_taxa, scaler_path):
    """Mean Absolute Error (MAE) computed on inverse-scaled data.

    Parameters
    - actual: array-like of ground truth values (scaled)
    - predicted: array-like of predicted values (scaled)
    - scaler_path: path to the scaler file used for inverse scaling

    Returns
    - float: MAE on original scale
    """
    # Inverse scale inputs back to the original space before computing metrics
    print("Scaling")
    inverse = inverse_scale_data_attention(predicted, scaler_path, num_taxa)
    actual_inv = inverse_scale_data_attention(actual, scaler_path, num_taxa)
    value = mean_absolute_error(actual_inv, inverse)
    logger.info(
        "MAE computed: %f | actual_inv shape=%s | pred_inv shape=%s",
        value,
        getattr(actual_inv, "shape", None),
        getattr(inverse, "shape", None),
    )
    return value


def rmse(actual, predicted, num_taxa, scaler_path):
    """Root Mean Squared Error (RMSE) computed on inverse-scaled data.

    Parameters
    - actual: array-like of ground truth values (scaled)
    - predicted: array-like of predicted values (scaled)

    Returns
    - float: RMSE on original scale
    """
    inverse = inverse_scale_data_attention(predicted, scaler_path, num_taxa)
    actual_inv = inverse_scale_data_attention(actual, scaler_path, num_taxa)
    value = math.sqrt(mean_squared_error(actual_inv, inverse))
    logger.info("RMSE computed: %f", value)
    return value


def nrmse(actual, predicted, num_taxa, scaler_path):
    """Normalized RMSE (NRMSE) computed on inverse-scaled data.

    Normalization is by the standard deviation of the predicted (inverse-scaled) values.

    Parameters
    - actual: array-like of ground truth values (scaled)
    - predicted: array-like of predicted values (scaled)

    Returns
    - float: NRMSE on original scale
    """
    inverse = inverse_scale_data_attention(predicted, scaler_path, num_taxa)
    actual_inv = inverse_scale_data_attention(actual, scaler_path, num_taxa)
    rmse_val = math.sqrt(mean_squared_error(actual_inv, inverse))
    std = np.std(inverse)
    nrmse_val = rmse_val / std if std != 0 else float("inf")
    logger.info("NRMSE computed: %f | rmse=%f | std=%f", nrmse_val, rmse_val, std)
    return nrmse_val


def combine_metrics(
    X_train,
    X_test,
    y_train,
    y_test,
    predict_train,
    predict_test,
    output_path,
    scaler_path,
    model_architecture=None,
):
    """Compute common metrics for train and test sets and optionally save to TSV.

    Parameters
    - y_train, y_test: ground truth arrays (scaled)
    - predict_train, predict_test: predicted arrays (scaled)
    - output_path: optional path to save a TSV with the metrics

    Returns
    - dict: calculated metrics
    """

    num_taxa = y_train.shape[1]

    if model_architecture == "attention" or model_architecture == "metadata_parallel":
        predict_train = reshape_attention(predict_train, X_train, y_train.shape[1])
        predict_test = reshape_attention(predict_test, X_test, y_test.shape[1])
        y_train = reshape_original_attention(y_train, X_train)
        y_test = reshape_original_attention(y_test, X_test)
    else:
        # For TCN/LSTM, predictions are shaped as (samples, horizon, num_targets)
        # and need to be reduced to the target dimension before inverse scaling.
        if predict_train.ndim == 3:
            predict_train = predict_train[:, -1, :]
        if predict_test.ndim == 3:
            predict_test = predict_test[:, -1, :]

    metrics = {
        "MAE train": mae(y_train, predict_train, num_taxa, scaler_path),
        "RMSE train": rmse(y_train, predict_train, num_taxa, scaler_path),
        "NRMSE train": nrmse(y_train, predict_train, num_taxa, scaler_path),
        "MAE test": mae(y_test, predict_test, num_taxa, scaler_path),
        "RMSE test": rmse(y_test, predict_test, num_taxa, scaler_path),
        "NRMSE test": nrmse(y_test, predict_test, num_taxa, scaler_path),
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
