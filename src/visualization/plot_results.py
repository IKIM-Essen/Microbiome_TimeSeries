import logging
import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from src.utils.config import get_num_taxa

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
LOG_DIR = os.path.join(ROOT_DIR, "logs", "visualization")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "plot_visualize_results.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(file_handler)

def load_taxa_mapping(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_prediction_interval(tsv_path):
    return pd.read_csv(tsv_path, sep="\t")


def flatten_prediction(pred):
    pred = np.asarray(pred)
    if pred.ndim == 3 and pred.shape[1] == 1:
        return pred.reshape(pred.shape[0], pred.shape[2])
    if pred.ndim == 2:
        return pred
    raise ValueError(f"Cannot flatten prediction array with shape {pred.shape}")


def load_predictions(predictions_npz):
    with np.load(predictions_npz) as data:
        pred_train = flatten_prediction(data["pred_train"])
        pred_val = flatten_prediction(data["pred_val"])
        pred_test = flatten_prediction(data["pred_test"])
    return pred_train, pred_val, pred_test


def load_split_sizes(split_sizes_path):
    with open(split_sizes_path, "rb") as f:
        split_sizes = pickle.load(f)
    return split_sizes.get("train_size"), split_sizes.get("val_size")


def parse_target_index(target_col):
    if target_col.startswith("Target"):
        return int(target_col.replace("Target", "")) - 1
    raise ValueError(f"Unrecognized target column '{target_col}'")


def plot_taxa_dropdown(complete_csv, dic_taxa_path, violations_tsv, prediction_interval_tsv, predictions_npz, output_html="results/tables/plot_taxa_violations.html"):
    logger.info("Loading complete dataframe from %s", complete_csv)
    complete = pd.read_csv(complete_csv, parse_dates=["Time"])
    dic_taxa = load_taxa_mapping(dic_taxa_path)

    # Use mapping values as taxa names and keys as target columns
    all_taxa = [(target, taxa_name) for target, taxa_name in dic_taxa.items()]

    logger.info("Loading violations from %s", violations_tsv)
    violations = pd.read_csv(violations_tsv, sep="\t")

    interval_df = None
    if prediction_interval_tsv and os.path.exists(prediction_interval_tsv):
        interval_df = read_prediction_interval(prediction_interval_tsv)
        if "species" not in interval_df.columns:
            raise ValueError("Prediction interval TSV must contain a 'species' column")

    include_interval = interval_df is not None
    train_size = None
    val_size = None
    pred_train = pred_val = pred_test = None

    #if split_sizes_path and os.path.exists(split_sizes_path):
    #    train_size, val_size = load_split_sizes(split_sizes_path)

    if predictions_npz and os.path.exists(predictions_npz):
        pred_train, pred_val, pred_test = load_predictions(predictions_npz)
        logger.info("Loaded prediction arrays from %s", predictions_npz)

    print("hi")
    print(pred_train.shape)
    print(pred_val.shape)

    train_size = pred_train.shape[0]
    val_size = pred_val.shape[0]

    prediction_traces = 2 if (train_size is not None and val_size is not None and pred_train is not None and pred_val is not None) else 0
    per_taxa_trace_count = 3 + prediction_traces + (5 if include_interval else 0)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.60, 0.40],
        horizontal_spacing=0.02,
        specs=[[{"type": "xy"}, {"type": "table"}]],
    )

    included_taxa = []
    for index, (target_col, taxa_name) in enumerate(all_taxa):
        if target_col not in complete.columns:
            logger.warning("Target column %s missing from complete dataframe, skipping", target_col)
            continue

        included_taxa.append((target_col, taxa_name))
        visible = len(included_taxa) == 1

        fig.add_trace(
            go.Scatter(
                x=complete["Time"],
                y=complete[target_col],
                mode="lines+markers",
                name="Actual values",
                line=dict(width=2),
                hovertemplate="<b>Taxa:</b> " + taxa_name + "<br><b>Date:</b> %{x|%Y-%m-%d}<br><b>Value:</b> %{y:.2f}<extra></extra>",
                visible=visible,
            ),
            row=1, col=1,
        )

        target_violations = violations[violations["Target"] == target_col]
        v_times = None
        if not target_violations.empty and "Time" in target_violations.columns:
            try:
                v_times = pd.to_datetime(target_violations["Time"])
            except Exception:
                v_times = target_violations["Time"]

        fig.add_trace(
            go.Scatter(
                x=v_times if v_times is not None else target_violations.index,
                y=target_violations["Actual"] if not target_violations.empty else [],
                mode="markers",
                marker=dict(symbol="x", color="black", size=10),
                name="Violations",
                hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> %{y:.2f}<extra></extra>",
                visible=visible,
            ),
            row=1, col=1,
        )
        #print(interval_df)
        if include_interval:
            pi_slice = interval_df[interval_df["species"] == taxa_name].sort_values("timepoint")

            interval_df["species"] = (
                interval_df["species"]
                .astype(str)
                .str.strip("[]")
                .str.strip("'")
            )

            #if taxa_name in interval_df["species"].values:
            #    print(f"Found: {taxa_name}")
            #print(taxa_name)
            #print("1")
            if not pi_slice.empty:
                #print("2")
                times = complete["Time"].values
                n_pi = len(pi_slice)
                times_pi = times[-n_pi:] if n_pi <= len(times) else times
                mean = pi_slice["mean"].values
                lower = pi_slice["lower"].values
                upper = pi_slice["upper"].values

                fig.add_trace(
                    go.Scatter(
                        x=times_pi,
                        y=mean,
                        mode="lines",
                        line=dict(color="green", dash="dash"),
                        name="Mean",
                        showlegend=True,
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=times_pi,
                        y=upper,
                        mode="lines",
                        line=dict(color="red", dash="dot"),
                        name="Upper",
                        showlegend=True,
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=times_pi,
                        y=lower,
                        mode="lines",
                        line=dict(color="red", dash="dot"),
                        name="Lower",
                        showlegend=True,
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=times_pi,
                        y=upper,
                        fill=None,
                        showlegend=False,
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=times_pi,
                        y=lower,
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)',
                        showlegend=False,
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )
            else:
                #print("3")
                for _ in range(5):
                    fig.add_trace(go.Scatter(x=[], y=[], visible=False), row=1, col=1)
        print
        if train_size is not None and pred_train is not None and pred_val is not None:
            #print("5")
            idx = parse_target_index(target_col)
            #print(idx)
            #print(pred_train)
            fig.add_trace(
                go.Scatter(
                    x=complete["Time"].iloc[:train_size],
                    y=pred_train[:, idx],
                    mode="lines",
                    line=dict(color="orange", dash="dot"),
                    name="Train prediction",
                    visible=visible,
                ),
                row=1,
                col=1,
            )
            #print(pred_val)
            #print(pred_val[:, idx])
            fig.add_trace(
                go.Scatter(
                    x=complete["Time"].iloc[train_size:(val_size+train_size)],
                    y=pred_val[:, idx],
                    mode="lines",
                    line=dict(color="purple", dash="dot"),
                    name="Validation prediction",
                    visible=visible,
                ),
                row=1,
                col=1,
            )

        if target_violations.empty:
            fig.add_trace(
                go.Table(
                    header=dict(values=["No violations for selected taxa"]),
                    cells=dict(values=[[]]),
                    visible=visible,
                ),
                row=1,
                col=2,
            )
        else:
            display_cols = ["Time", "Actual", "Lower", "Upper", "Violation"]
            table_df = target_violations[display_cols].copy()
            table_df["Time"] = table_df["Time"].astype(str)
            cells = [table_df[col].tolist() for col in table_df.columns]
            fig.add_trace(
                go.Table(
                    header=dict(values=list(table_df.columns)),
                    cells=dict(values=cells, align="left"),
                    visible=visible,
                ),
                row=1,
                col=2,
            )

    if not included_taxa:
        raise ValueError("No matching taxa columns were found in the complete dataframe.")

    buttons = []
    for idx, (_, taxa_name) in enumerate(included_taxa):
        base = idx * per_taxa_trace_count
        button_visibility = [False] * len(fig.data)
        for trace_i in range(per_taxa_trace_count):
            if base + trace_i < len(button_visibility):
                button_visibility[base + trace_i] = True
        buttons.append(
            dict(
                label=taxa_name,
                method="update",
                args=[{"visible": button_visibility}, {"title": f"Time series and interval violations for {taxa_name}"}],
            )
        )

    fig.update_layout(
        title=f"Time series and interval violations for {included_taxa[0][1]}",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        margin=dict(l=60, r=20, t=100, b=120),
        legend=dict(
            orientation="h",
            y=-0.12,
            x=0.0,
            xanchor="left",
            yanchor="top",
        ),
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

    out_dir = os.path.dirname(output_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    logger.info("Writing interactive plot to %s", output_html)
    pio.write_html(fig, output_html, auto_open=False)
    logger.info("Finished writing interactive plot")
    return fig