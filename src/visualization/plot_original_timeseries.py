import logging
import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
LOG_DIR = os.path.join(ROOT_DIR, "logs", "visualization")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "plot_original_timeseries.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(file_handler)


def _load_bacteria_mapping(bacteria):
    """Resolve the bacteria mapping into column names and plot labels.

    Parameters
    ----------
    bacteria : str | dict | list
        If str, path to a pickle file containing a mapping dict.
        If dict, mapping of dataframe column names to legend labels.
        If list, a list of dataframe column names.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing the column names to plot and the labels for each line.
    """
    if isinstance(bacteria, str):
        with open(bacteria, "rb") as file:
            bacteria = pickle.load(file)
    if isinstance(bacteria, dict):
        columns = list(bacteria.keys())
        labels = list(bacteria.values())
    elif isinstance(bacteria, list):
        columns = bacteria
        labels = bacteria
    else:
        raise ValueError("bacteria must be a mapping path, dict, or list of column names")
    return columns, labels


def time_series_analysis_plot(dataframe_complete_path, filename, bacteria):
    """Plot time series data for selected bacterial targets.

    Parameters
    ----------
    dataframe_complete_path : str
        Path to the CSV file containing the complete dataframe.
    filename : str
        Path where the output image file will be saved.
    bacteria : str | dict | list
        If a string, the path to a pickle file containing a
        mapping dict from dataframe columns to labels.
        If a dict, keys are dataframe columns and values are legend labels.
        If a list, it should contain dataframe column names.
    """
    logger.info("Starting time series analysis plot: dataframe=%s, output=%s", dataframe_complete_path, filename)

    # Load the complete data frame from disk and resolve the bacteria mapping.
    dataframe_complete = pd.read_csv(dataframe_complete_path, parse_dates=["Time"])
    logger.info("Loaded complete dataframe with shape %s", dataframe_complete.shape)
    plot_columns, plot_labels = _load_bacteria_mapping(bacteria)
    logger.info("Resolved bacteria mapping for %s columns", len(plot_columns))

    # Create a deterministic color palette for Plotly traces.
    np.random.seed(100)
    hues = np.linspace(0, 360, len(plot_columns), endpoint=False)
    colors = []
    for idx, _ in enumerate(plot_columns):
        # Alternate colors by taking one hue from the start and one from the mid-point
        # to keep adjacent traces visually distinct.
        if idx % 2 == 0:
            hue = hues[idx // 2]
        else:
            hue = hues[(idx // 2 + len(hues) // 2) % len(hues)]
        colors.append(f"hsl({hue}, 70%, 50%)")

    # Build the Plotly figure with one trace per selected taxa.
    fig = go.Figure()
    for column, label, color in zip(plot_columns, plot_labels, colors):
        if column in dataframe_complete.columns:
            fig.add_trace(
                go.Scatter(
                    x=dataframe_complete["Time"],
                    y=dataframe_complete[column],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color),
                    hovertemplate=(
                        "<b>Bacteria:</b> %{fullData.name}<br>"
                        "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                        "<b>Value:</b> %{y:.2f}<extra></extra>"
                    ),
                )
            )
        else:
            logger.error("Column '%s' not found in dataframe_complete columns %s", column, list(dataframe_complete.columns))
            raise KeyError(f"Column '{column}' not found in dataframe_complete")

    # Configure the interactive layout and legend.
    fig.update_layout(
        title="Display of microbial genera in samples over time",
        xaxis_title="Date",
        yaxis_title="Number of reads found",
        xaxis=dict(type="date", tickformat="%Y-%m-%d"),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            itemsizing="trace"
        ),
        margin=dict(l=80, r=320, t=120, b=80),
        hovermode="closest",
        template="plotly_white"
    )

    # Save the interactive plot to an HTML file.
    logger.info("Writing interactive plot to %s", filename)
    pio.write_html(fig, filename, auto_open=False)
    logger.info("Finished writing interactive plot")
    return fig