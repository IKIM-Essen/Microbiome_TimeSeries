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


def find_interval_anomalies(
    complete,
    prediction_intervals,
    dic_taxa,
    val_start,
    target_prefix="Target",
    date_column="Time",
    skip_first=1,
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
        Row index where the test period begins (val_size).
    target_prefix : str
        Column name prefix, usually "Target".
    date_column : str
        Date column name in `complete`.
    skip_first : int
        Number of leading interval rows to skip if your interval arrays include an extra initial value.
    csv_path : str or None
        Optional path to save the violation results as CSV.
    """
    test_df = complete.iloc[val_start:].reset_index(drop=True)
    times = test_df[date_column].tolist()

    anomalies = []
    for target_idx, interval in enumerate(prediction_intervals):
        upper, lower, _ = interval
        if skip_first:
            upper = upper[skip_first:]
            lower = lower[skip_first:]

        target_name = f"{target_prefix}{target_idx+1}"
        actual_values = test_df[target_name].values

        if len(actual_values) != len(upper) or len(actual_values) != len(lower):
            raise ValueError(
                f"Length mismatch for {target_name}: "
                f"actual={len(actual_values)}, upper={len(upper)}, lower={len(lower)}"
            )

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

    anomalies_df = pd.DataFrame(anomalies)
    if csv_path is not None:
        anomalies_df.to_csv(csv_path, index=False)
    return anomalies_df
