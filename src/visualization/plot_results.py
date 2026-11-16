import ast
import logging
import os
import pickle
from numbers import Integral

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from joblib import load
from plotly.subplots import make_subplots

from src.evaluation.outlier import ANOMALY_COLUMNS, align_prediction_intervals
from src.preprocessing.scaling import inverse_scale_data

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
LOG_DIR = os.path.join(ROOT_DIR, "logs", "visualization")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "plot_visualize_results.log")
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)


def load_taxa_mapping(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def read_prediction_interval(tsv_path):
    frame = pd.read_csv(tsv_path, sep="\t", dtype={"Time": str})
    required = {"species", "lower", "upper", "mean"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Interval table requires columns: {sorted(required)}")
    if "Time" not in frame and "timepoint" not in frame:
        raise ValueError("Interval table requires Time or timepoint labels")

    def normalize_label(value):
        label = str(value)
        # Older interval files serialized singleton numpy labels as ['taxon'].
        if label.startswith("[") and label.endswith("]"):
            try:
                legacy = ast.literal_eval(label)
            except (ValueError, SyntaxError):
                return label
            if isinstance(legacy, list) and len(legacy) == 1:
                return str(legacy[0])
        return label

    frame["species"] = frame["species"].map(normalize_label)
    return frame


def read_anomalies(path):
    if path is None:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)
    try:
        frame = pd.read_csv(path, sep="\t", dtype={"Time": str})
    except pd.errors.EmptyDataError:
        logger.warning("Legacy empty anomaly file %s has no headers", path)
        return pd.DataFrame(columns=ANOMALY_COLUMNS)
    required = {"Time", "Target", "Actual", "Lower", "Upper", "Violation"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Anomaly table requires columns: {sorted(required)}")
    return frame


def flatten_prediction(pred):
    pred = np.asarray(pred)
    if pred.ndim == 3 and pred.shape[1] == 1:
        return pred[:, 0, :]
    if pred.ndim == 2:
        return pred
    raise ValueError(
        f"Expected single-step predictions shaped (samples, targets) or "
        f"(samples, 1, targets), got {pred.shape}"
    )


def load_predictions(predictions_npz, scaler_path):
    if scaler_path is None:
        raise ValueError(
            "A scaler file is required to plot predictions in abundance units"
        )
    with np.load(predictions_npz) as data:
        predictions = [
            flatten_prediction(data[key])
            for key in ("pred_train", "pred_val", "pred_test")
        ]
    if len({array.shape[1] for array in predictions}) != 1:
        raise ValueError("Prediction splits must have matching target columns")
    return tuple(
        inverse_scale_data(array, scaler_path) if len(array) else array
        for array in predictions
    )


def load_split_sizes(split_sizes_path):
    """Return training end and test start; legacy val_size is a cumulative boundary."""
    sizes = load(split_sizes_path)
    train_end, test_start = sizes.get("train_size"), sizes.get("val_size")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool)
        for value in (train_end, test_start)
    ):
        raise ValueError(
            "Split sizes require integer train_size and val_size boundaries"
        )
    return train_end, test_start


def parse_target_index(target_col):
    if target_col.startswith("Target"):
        index = int(target_col[len("Target") :]) - 1
        if index >= 0:
            return index
    raise ValueError(f"Unrecognized target column {target_col!r}")


def interval_values(frame, complete, test_start):
    """Align interval rows using explicit labels, or validated legacy split offsets."""
    if "Time" in frame:
        labels = frame["Time"]
        if labels.isna().any() or labels.duplicated().any():
            raise ValueError(
                "Interval Time labels must be nonempty and unique per taxon"
            )
        expected = (
            complete["Time"].iloc[test_start:] if test_start is not None else None
        )
        label_set = set(labels)
        if expected is not None and label_set != set(expected):
            raise ValueError("Interval dates do not match the saved test period")
        if not label_set.issubset(set(complete["Time"])):
            raise ValueError("Interval dates are absent from the prepared observations")
        times = [time for time in complete["Time"] if time in label_set]
        ordered = frame.set_index("Time").loc[times]
        return times, ordered["upper"], ordered["lower"], ordered["mean"]
    if test_start is None:
        raise ValueError(
            "Legacy intervals require predictions or a saved split-size file"
        )
    ordered = frame.assign(timepoint=pd.to_numeric(frame["timepoint"], errors="raise"))
    ordered = ordered.sort_values("timepoint")
    if ordered["timepoint"].tolist() != list(range(len(ordered))):
        raise ValueError(
            "Legacy interval timepoints must be consecutive starting at zero"
        )
    times = complete["Time"].iloc[test_start:].tolist()
    upper, lower, mean = align_prediction_intervals(
        [[ordered["upper"], ordered["lower"], ordered["mean"]]], len(times)
    )[0]
    return times, upper, lower, mean


def plot_taxa_dropdown(
    complete_csv,
    dic_taxa_path,
    anomalies_tsv,
    prediction_interval_tsv,
    predictions_npz,
    output_html="results/tables/plot_taxa_anomalies.html",
    split_sizes_path=None,
    scaler_path=None,
):
    complete = pd.read_csv(complete_csv, dtype={"Time": str})
    if "Time" not in complete or complete.empty:
        raise ValueError("Prepared observations require a nonempty Time column")
    if complete["Time"].isna().any() or complete["Time"].duplicated().any():
        raise ValueError("Prepared Time labels must be nonempty and unique")
    dic_taxa = load_taxa_mapping(dic_taxa_path)
    anomalies = read_anomalies(anomalies_tsv)
    if not set(anomalies["Target"]).issubset(set(dic_taxa)):
        raise ValueError("Anomaly targets are absent from the saved taxa mapping")
    if not set(anomalies["Time"]).issubset(set(complete["Time"])):
        raise ValueError("Anomaly dates are absent from the prepared observations")
    interval_df = (
        read_prediction_interval(prediction_interval_tsv)
        if prediction_interval_tsv is not None
        else None
    )
    if interval_df is not None and not set(interval_df["species"]).issubset(
        set(dic_taxa.values())
    ):
        raise ValueError("Interval taxa are absent from the saved taxa mapping")

    predictions = None
    train_end = test_start = None
    if predictions_npz is not None:
        predictions = load_predictions(predictions_npz, scaler_path)
        lengths = [len(array) for array in predictions]
        if sum(lengths) != len(complete):
            raise ValueError("Prediction rows do not match the prepared observations")
        train_end, test_start = lengths[0], lengths[0] + lengths[1]
    if split_sizes_path is not None:
        saved = load_split_sizes(split_sizes_path)
        if predictions is not None and saved != (train_end, test_start):
            raise ValueError(
                "Saved split boundaries do not match the prediction arrays"
            )
        train_end, test_start = saved
    if test_start is not None and not 0 <= train_end <= test_start <= len(complete):
        raise ValueError("Saved split boundaries are outside the prepared observations")
    if test_start is not None and not set(anomalies["Time"]).issubset(
        set(complete["Time"].iloc[test_start:])
    ):
        raise ValueError("Anomaly dates do not match the saved test period")

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.60, 0.40],
        horizontal_spacing=0.02,
        specs=[[{"type": "xy"}, {"type": "table"}]],
    )
    included_taxa = []
    trace_groups = []
    for target_col, taxa_name in dic_taxa.items():
        if target_col not in complete.columns:
            raise ValueError(
                f"Mapped target {target_col!r} is absent from the prepared table"
            )
        start = len(fig.data)
        visible = not included_taxa
        included_taxa.append(taxa_name)

        def add_scatter(**kwargs):
            fig.add_trace(go.Scatter(visible=visible, **kwargs), row=1, col=1)

        add_scatter(
            x=complete["Time"],
            y=complete[target_col],
            mode="lines+markers",
            name="Actual values",
            line=dict(width=2),
            hovertemplate="<b>Time:</b> %{x}<br><b>Abundance:</b> %{y:.4g}<extra></extra>",
        )
        target_anomalies = anomalies[anomalies["Target"] == target_col]
        add_scatter(
            x=target_anomalies["Time"],
            y=target_anomalies["Actual"],
            mode="markers",
            marker=dict(symbol="x", color="black", size=10),
            name="Anomalies",
            hovertemplate="<b>Time:</b> %{x}<br><b>Actual:</b> %{y:.4g}<extra></extra>",
        )
        if interval_df is not None:
            subset = interval_df[interval_df["species"] == taxa_name]
            if not subset.empty:
                times, upper, lower, mean = interval_values(
                    subset, complete, test_start
                )
                for values, name, color, dash in (
                    (mean, "Mean", "green", "dash"),
                    (upper, "Upper", "red", "dot"),
                    (lower, "Lower", "red", "dot"),
                ):
                    add_scatter(
                        x=times,
                        y=values,
                        mode="lines",
                        name=name,
                        line=dict(color=color, dash=dash),
                    )
                add_scatter(
                    x=times,
                    y=upper,
                    mode="lines",
                    showlegend=False,
                    line=dict(width=0),
                    hoverinfo="skip",
                )
                add_scatter(
                    x=times,
                    y=lower,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(255,0,0,0.1)",
                    showlegend=False,
                    line=dict(width=0),
                    hoverinfo="skip",
                )
        if predictions is not None:
            index = parse_target_index(target_col)
            if index >= predictions[0].shape[1]:
                raise ValueError(f"Prediction arrays have no column for {target_col}")
            for values, first, last, name, color in (
                (predictions[0], 0, train_end, "Train prediction", "orange"),
                (
                    predictions[1],
                    train_end,
                    test_start,
                    "Validation prediction",
                    "purple",
                ),
                (predictions[2], test_start, len(complete), "Test prediction", "blue"),
            ):
                add_scatter(
                    x=complete["Time"].iloc[first:last],
                    y=values[:, index],
                    mode="lines",
                    name=name,
                    line=dict(color=color, dash="dot"),
                )
        if target_anomalies.empty:
            table = go.Table(
                header=dict(values=["No anomalies for selected taxon"]),
                cells=dict(values=[[]]),
                visible=visible,
            )
        else:
            display_cols = ["Time", "Actual", "Lower", "Upper", "Violation"]
            table = go.Table(
                header=dict(values=display_cols),
                cells=dict(
                    values=[target_anomalies[col].tolist() for col in display_cols],
                    align="left",
                ),
                visible=visible,
            )
        fig.add_trace(table, row=1, col=2)
        trace_groups.append(range(start, len(fig.data)))
    if not included_taxa:
        raise ValueError("No matching taxa columns were found in the prepared table")

    buttons = []
    for taxon, group in zip(included_taxa, trace_groups):
        visibility = [index in group for index in range(len(fig.data))]
        buttons.append(
            dict(
                label=taxon,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Time series and anomalies for {taxon}"},
                ],
            )
        )
    fig.update_layout(
        title=f"Time series and anomalies for {included_taxa[0]}",
        xaxis_title="Time",
        yaxis_title="Abundance",
        template="plotly_white",
        margin=dict(l=60, r=20, t=100, b=120),
        legend=dict(orientation="h", y=-0.12, x=0.0, xanchor="left", yanchor="top"),
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
    pio.write_html(fig, output_html, auto_open=False)
    logger.info("Wrote interactive plot to %s", output_html)
    return fig
