import keras
import pickle
import logging
import os
import numpy as np

from src.preprocessing.read import create_complete_df
from src.preprocessing.split import split_without_scaling
from src.preprocessing.scaling import scale_data_with_scaler, inverse_scale_data
from src.utils.config import load_model_if_path, reshape, prediction_interval_to_df
from src.model_building.predict import predict_retrain
from src.evaluation.evaluation_metrics import combine_metrics
from src.evaluation.ensemble import predict_interval
from src.evaluation.outlier import find_interval_anomalies

# Setup logging for model-building and training operations
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
LOG_DIR = os.path.join(ROOT_DIR, "logs", "retraining")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "retraining.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)


def retrain_model(
    timeseries,
    taxalist,
    exo,
    tcn_path,
    lstm_path,
    scaler,
    output_path,
    predictions_out,
    anomalies_path,
    train_percentage,
    val_percentage,
):
    retraining = True
    print("Read new model data")
    new_timeseries, metadata_woT, num_taxa, new_taxa = create_complete_df(
        timeseries, taxalist, exo
    )
    new_timeseries.sort_values(by=["Time"], inplace=True, ignore_index=True)

    with open("results/intermediate/dic_TargTax.pkl", "wb") as mapping_file:
        pickle.dump(new_taxa, mapping_file)

    print("Scale new data")

    scaled_data = scale_data_with_scaler(new_timeseries, scaler)

    X_train, y_train, X_val, y_val, X_test, y_test = split_without_scaling(
        scaled_data,
        num_taxa,
        train_percentage,
        val_percentage,
        split_sizes_path="results/intermediate/split_sizes.pkl",
    )

    print("Fit Model")

    # --- Train TCN first ---

    tcn = load_model_if_path(tcn_path)
    lstm = load_model_if_path(lstm_path)

    # --- Compute residuals ---
    predictions_train, predictions_val, predictions_test, tcn, lstm = predict_retrain(
        X_train, y_train, X_val, y_val, X_test, tcn, lstm, output_path
    )

    if predictions_out:
        output_dir = os.path.dirname(predictions_out)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(
            predictions_out,
            pred_train=predictions_train,
            pred_val=predictions_val,
            pred_test=predictions_test,
        )
        logger.info("Saved predictions to %s", predictions_out)

    y_train_pred = reshape(predictions_train)
    y_test_pred = reshape(predictions_test)
    y_val_pred = reshape(predictions_val)

    metrics = combine_metrics(y_train, y_test, y_train_pred, y_test_pred, output_path)

    # prediction_inter = prediction_interval_two_stage.predict_interval(50, X_train, y_train, X_val, y_val, X_test, y_test, scaler, num_taxa)

    prediction_inter = predict_interval(
        50,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler,
        new_taxa,
        tcn_path,
        lstm_path,
        retraining,
    )

    anomalies_df = find_interval_anomalies(
        new_timeseries,
        prediction_inter,
        new_taxa,
        y_val.shape[0],
        target_prefix="Target",
        date_column="Time",
        skip_first=1,
        csv_path=anomalies_path,
    )

    prediction_interval_df = prediction_interval_to_df(prediction_inter, new_taxa)
    prediction_interval_df.to_csv(output_path, sep="\t", index=False)

    return anomalies_df
