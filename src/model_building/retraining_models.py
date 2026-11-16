import logging
import pickle
import shutil
from pathlib import Path

import numpy as np
from joblib import load

from src.preprocessing.read import create_complete_df
from src.preprocessing.split import split_without_scaling
from src.preprocessing.scaling import scale_data_with_scaler
from src.utils.config import load_model_if_path, prediction_interval_to_df
from src.model_building.predict import predict_retrain
from src.evaluation.evaluation_metrics import combine_metrics
from src.evaluation.ensemble import predict_retraining_interval
from src.evaluation.outlier import find_interval_anomalies
from src.visualization.plot_results import plot_taxa_dropdown

logger = logging.getLogger(__name__)


def _validate_source_model(model, input_shape, num_taxa):
    if (
        isinstance(model.input_shape, list)
        or tuple(model.input_shape[1:]) != input_shape
    ):
        raise ValueError(
            "Source model input shape does not match the prepared features"
        )
    if isinstance(model.output_shape, list) or tuple(model.output_shape[1:]) not in (
        (num_taxa,),
        (1, num_taxa),
    ):
        raise ValueError(
            "Source model must predict one step for the reference microbial taxa"
        )
    if getattr(model, "optimizer", None) is None:
        raise ValueError("Source model must include its saved training configuration")


def retrain_model(
    *,
    timeseries,
    taxa,
    source_mapping,
    scaler_path,
    output_dir,
    lstm_path,
    tcn_path=None,
    metadata=None,
    include_metadata=False,
    model_architecture="tcn_lstm",
    train_percentage=0.5,
    val_percentage=0.1,
    epochs=10,
    batch_size=32,
    number_models=5,
):
    """Retrain saved standard models using a fixed scaler and taxa/feature schema.

    All new artifacts go into an empty output directory. Source models, scaler,
    mapping, and input tables are never overwritten. Metadata-specific model
    architectures require a separate retraining implementation.
    """
    if model_architecture not in {"tcn_lstm", "lstm"}:
        raise ValueError("Retraining currently supports only tcn_lstm and lstm")
    if (
        not 0 < train_percentage < 1
        or not 0 < val_percentage < 1
        or train_percentage + val_percentage >= 1
    ):
        raise ValueError(
            "Training and validation fractions must be positive and sum to less than one"
        )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in (epochs, batch_size, number_models)
    ):
        raise ValueError(
            "Epochs, batch size, and ensemble size must be positive integers"
        )
    if model_architecture == "tcn_lstm" and tcn_path is None:
        raise ValueError("tcn_lstm retraining requires a source TCN model")
    if include_metadata and metadata is None:
        raise ValueError("A metadata file is required when metadata are enabled")
    source_tcn = tcn_path if model_architecture == "tcn_lstm" else None
    sources = [timeseries, taxa, source_mapping, scaler_path, lstm_path]
    sources += [source_tcn] if source_tcn is not None else []
    sources += [metadata] if include_metadata else []
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Retraining output directory must be new or empty")
    for source in sources:
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source file does not exist: {path}")
        if path.is_relative_to(output):
            raise ValueError(
                "Retraining outputs must be separate from all source files"
            )

    with open(source_mapping, "rb") as handle:
        original_mapping = pickle.load(handle)
    scaler = load(scaler_path)
    paths = {
        "complete_csv": output / "tables" / "complete_df.csv",
        "mapping": output / "intermediate" / "dic_TargTax.pkl",
        "splits": output / "intermediate" / "splits.npz",
        "split_sizes": output / "intermediate" / "split_sizes.pkl",
        "scaler": output / "models" / "scaler.pkl",
        "tcn": output / "models" / "tcn_model.h5",
        "lstm": output / "models" / "lstm_model.h5",
        "predictions": output / "intermediate" / "predictions.npz",
        "metrics": output / "tables" / "evaluation_metrics.tsv",
        "interval": output / "tables" / "prediction_interval.tsv",
        "anomalies": output / "tables" / "prediction_interval_anomalies.tsv",
        "plot": output / "figures" / "plot_taxa_anomalies.html",
    }
    complete, _, num_taxa, mapping = create_complete_df(
        timeseries, metadata, taxa, include_metadata, paths["complete_csv"]
    )
    if mapping != original_mapping:
        raise ValueError("Reference taxa and order must match the source taxa mapping")
    features = complete.drop(columns="Time")
    if hasattr(scaler, "feature_names_in_"):
        expected = list(scaler.feature_names_in_)
        if set(expected) != set(features.columns) or expected[:num_taxa] != list(
            mapping
        ):
            raise ValueError(
                "Metadata and target fields must match the source scaler schema"
            )
        complete = complete[["Time", *expected]]
        complete.to_csv(paths["complete_csv"])
    elif include_metadata:
        raise ValueError(
            "Source scaler has no feature names; metadata schema cannot be verified"
        )
    if features.shape[1] != scaler.n_features_in_:
        raise ValueError("Feature count does not match the source scaler")
    scaled = scale_data_with_scaler(complete, scaler_path)
    if len(scaled) != len(complete) or not np.isfinite(scaled).all():
        raise ValueError("Prepared retraining observations must be complete and finite")
    splits = split_without_scaling(
        scaled, num_taxa, train_percentage, val_percentage, paths["split_sizes"]
    )
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    if any(len(values) == 0 for values in (X_train, X_val, X_test)):
        raise ValueError(
            "Retraining requires nonempty training, validation, and test splits"
        )
    tcn = load_model_if_path(source_tcn)
    lstm = load_model_if_path(lstm_path)
    for model in (tcn, lstm):
        if model is not None:
            _validate_source_model(model, tuple(X_train.shape[1:]), num_taxa)

    # Preserve the source scaler bytes and map; new data are transformed, never fitted.
    paths["scaler"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scaler_path, paths["scaler"])
    with open(paths["mapping"], "wb") as handle:
        pickle.dump(mapping, handle)
    np.savez_compressed(
        paths["splits"],
        **dict(
            zip(("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"), splits)
        ),
    )
    logger.info("Retraining source models on %s", timeseries)
    pred_train, pred_val, pred_test, _, _ = predict_retrain(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        tcn,
        lstm,
        output_path=paths["predictions"],
        tcn_path=paths["tcn"] if tcn is not None else None,
        lstm_path=paths["lstm"],
        epochs=epochs,
        batch_size=batch_size,
    )
    combine_metrics(
        None,
        None,
        y_train,
        y_test,
        pred_train,
        pred_test,
        paths["metrics"],
        paths["scaler"],
        model_architecture,
    )
    logger.info(
        "Estimating intervals from %s independently retrained copies", number_models
    )
    intervals = predict_retraining_interval(
        number_models,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        paths["scaler"],
        source_tcn,
        lstm_path,
        epochs=epochs,
        batch_size=batch_size,
    )
    test_start = len(X_train) + len(X_val)
    prediction_interval_to_df(
        intervals, list(mapping.values()), timepoints=complete["Time"].iloc[test_start:]
    ).to_csv(paths["interval"], sep="\t", index=False)
    find_interval_anomalies(complete, intervals, mapping, test_start).to_csv(
        paths["anomalies"], sep="\t", index=False
    )
    plot_taxa_dropdown(
        paths["complete_csv"],
        paths["mapping"],
        paths["anomalies"],
        paths["interval"],
        paths["predictions"],
        paths["plot"],
        split_sizes_path=paths["split_sizes"],
        scaler_path=paths["scaler"],
    )
    if tcn is None:
        paths.pop("tcn")
    return paths
