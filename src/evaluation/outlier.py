import logging
import os
import pickle
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
LOG_DIR = os.path.join(ROOT_DIR, "logs", "predicting")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "outlier.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)


ANOMALY_COLUMNS = [
    "Time",
    "Target",
    "Taxa",
    "Actual",
    "Lower",
    "Upper",
    "Violation",
    "TestIndex",
]


def align_prediction_intervals(prediction_intervals, expected_length, skip_first=0):
    """Remove only surplus leading NaN padding, preserving actual test positions."""
    if expected_length < 0 or skip_first < 0:
        raise ValueError("Interval lengths and skip_first must be nonnegative")
    aligned = []
    for index, interval in enumerate(prediction_intervals):
        if len(interval) != 3:
            raise ValueError(
                "Each prediction interval must contain upper, lower, and mean"
            )
        arrays = [np.asarray(values, dtype=float) for values in interval]
        if any(values.ndim != 1 for values in arrays):
            raise ValueError("Prediction interval arrays must be one-dimensional")
        if len({len(values) for values in arrays}) != 1:
            raise ValueError("Upper, lower, and mean arrays must have equal lengths")
        arrays = [values[skip_first:] for values in arrays]
        excess = len(arrays[0]) - expected_length
        if excess < 0 or (
            excess > 0 and not all(np.isnan(values[:excess]).all() for values in arrays)
        ):
            raise ValueError(
                f"Interval {index + 1} does not match {expected_length} test observations; "
                "only surplus leading NaN padding can be removed"
            )
        aligned.append([values[excess:] for values in arrays])
    return aligned


def find_interval_anomalies(
    complete,
    prediction_intervals,
    dic_taxa,
    val_start,
    target_prefix="Target",
    date_column="Time",
    skip_first=0,
    csv_path=None,
):
    """
    Compare actual test values to prediction interval bounds.

    Parameters
    ----------
    complete : pd.DataFrame
        Full dataframe with Time and Target columns.
    prediction_intervals : list
        prediction_inter[i] == [upper_array, lower_array, mean_array].
    dic_taxa : dict
        Mapping of Target{i+1} to taxonomy name.
    val_start : int
        Row index where the test period begins (training plus validation rows).
    target_prefix : str
        Column name prefix, usually "Target".
    date_column : str
        Date column name in `complete`.
    skip_first : int
        Explicit legacy offset, normally zero. Surplus leading NaN padding is
        removed automatically; missing values within the test period are preserved.
    csv_path : str or None
        Optional path to save the violation results as CSV.
    """
    if not 0 <= val_start <= len(complete):
        raise ValueError("Test start must be within the complete dataframe")
    if len(prediction_intervals) != len(dic_taxa):
        raise ValueError("Prediction interval count must match the saved taxa mapping")
    test_df = complete.iloc[val_start:].reset_index(drop=True)
    times = test_df[date_column].tolist()

    anomalies = []
    aligned = align_prediction_intervals(prediction_intervals, len(test_df), skip_first)
    for target_idx, interval in enumerate(aligned):
        upper, lower, _ = interval

        target_name = f"{target_prefix}{target_idx+1}"
        actual_values = test_df[target_name].values

        for row_idx, (time, actual, u, l) in enumerate(
            zip(times, actual_values, upper, lower)
        ):
            if actual > u or actual < l:
                anomalies.append(
                    {
                        "Time": time,
                        "Target": target_name,
                        "Taxa": dic_taxa.get(target_name, ""),
                        "Actual": actual,
                        "Lower": l,
                        "Upper": u,
                        "Violation": "above" if actual > u else "below",
                        "TestIndex": row_idx,
                    }
                )

    anomalies_df = pd.DataFrame(anomalies, columns=ANOMALY_COLUMNS)
    if csv_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        anomalies_df.to_csv(csv_path, index=False)
    return anomalies_df
