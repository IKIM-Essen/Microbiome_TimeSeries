import logging
import os

import numpy as np
import pandas as pd

from src.utils.config import load_config

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
LOG_DIR = os.path.join(ROOT_DIR, "logs", "preprocessing")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "read.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

CONFIG_PATH = "config/encodings.yaml"


def read_data(path):
    """
    Read a taxa table into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the taxa file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing taxa information.
    """

    logger.info("Reading metadata from %s", path)
    # Read the taxa file:
    # - header=None -> file does not contain column names
    # - index_col=0 -> use the first column as row indices
    metadata = pd.read_csv(path, header=0, sep="\t", index_col=0)
    logger.info("Read metadata shape %s", metadata.shape)
    return metadata


def read_taxa(path):
    """
    Read a taxa table into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the taxa file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing taxa information.
    """

    logger.info("Reading taxa reference from %s", path)
    # Read the taxa file:
    # - header=None -> file does not contain column names
    # - index_col=0 -> use the first column as row indices
    taxa = pd.read_csv(path, header=None, index_col=0)
    logger.info("Read taxa shape %s", taxa.shape)
    return taxa


def get_timeframe(df):
    """
    Extract all column names from a DataFrame as a list.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    list
        List containing the column names.
    """

    # Convert DataFrame column names into a Python list
    timelist = df.columns.values.tolist()
    return timelist


def preprocess_metadata(metadata):
    """
    Preprocess metadata for downstream machine learning analysis.

    The preprocessing includes:
    - replacing missing values with 0
    - encoding categorical season values numerically
    - encoding month names numerically
    - removing unnecessary columns

    Parameters
    ----------
    metadata : pandas.DataFrame
        Input metadata table.

    Returns
    -------
    pandas.DataFrame
        Preprocessed metadata table.
    """

    config = load_config(CONFIG_PATH)

    season_map = config["season_map"]
    month_map = config["month_map"]
    cols_to_drop = config["columns_to_drop"]

    logger.info(
        "Preprocessing metadata: season_map keys=%s, month_map keys=%s, drop_cols=%s",
        list(season_map.keys()),
        list(month_map.keys()),
        cols_to_drop,
    )

    # metadata_woT = metadata.drop(['Time'], axis=1)
    metadata_woT = metadata.fillna(0)

    # Encode season and month values if they exist in the metadata
    if "season" in metadata.index:
        metadata_woT["season_ord"] = metadata_woT["season"].str.lower().map(season_map)

    if "month" in metadata.index:
        metadata_woT["month_ord"] = metadata_woT["month"].str.lower().map(month_map)

    # Remove unnecessary columns if they exist in the metadata
    metadata_woT = metadata_woT.drop(
        columns=[c for c in cols_to_drop if c in metadata_woT.columns]
    )
    logger.info("Metadata preprocessed shape %s", metadata_woT.shape)

    return metadata_woT


def read_timeseries(path_taxa, path_data):
    """
    Read and preprocess microbiome time-series abundance data.

    The function:
    - loads taxa and abundance tables
    - aggregates duplicate taxa entries
    - removes unassigned taxa
    - standardizes taxonomy labels
    - aligns abundance data to a reference taxa table
    - replaces missing values with 0

    Parameters
    ----------
    path_taxa : str
        Path to the taxa reference file.

    path_data : str
        Path to the abundance/time-series data file.

    Returns
    -------
    tuple
        index_list : list
            List of taxa names after preprocessing.

        result : pandas.DataFrame
            Preprocessed abundance table aligned to the taxa reference.
    """
    taxa_df = read_taxa(path_taxa)
    data_df = read_data(path_data)
    logger.info("Reading time-series data from %s", path_data)
    data_df_reduced = data_df.groupby(data_df.index).sum()
    logger.info("Combined duplicate rows, shape now %s", data_df_reduced.shape)
    # Remove unassigned taxa and species level taxonomy labels
    # for i in data_df_reduced.index:
    #    if "Unassigned" not in i:
    #        data_df_reduced.index = data_df_reduced.index.str.split("; s__").str[0]
    #    if "Unassigned" in i:
    #        data_df_reduced.drop(index=i, inplace=True)
    data_df_reduced.index = data_df_reduced.index.str.split("; s__").str[0]
    data_df_reduced = data_df_reduced[~data_df_reduced.index.str.contains("Unassigned")]
    logger.info("Filtered unassigned taxa, shape now %s", data_df_reduced.shape)
    # Combine duplicate genera entries by summing their abundances
    data_df_reduced = data_df_reduced.groupby(data_df_reduced.index).sum()
    index_list = data_df_reduced.index.tolist()
    result = data_df_reduced.reindex(taxa_df.index)
    result.fillna(0, inplace=True)
    logger.info("Reindexed taxa to reference list, final shape %s", result.shape)
    return index_list, result


def create_complete_df(
    path_prob, path_exo, path_tax, include_metadata=False, output="results/output.csv"
):
    """
    Create a combined dataframe containing microbiome abundance data
    and optionally associated metadata.

    Parameters
    ----------
    path_prob : str
        Path to the microbiome abundance/time-series data file.

    path_exo : str
        Path to the metadata file.

    path_tax : str
        Path to the taxa reference file.

    include_metadata : bool, optional
        If True, metadata features are added to the final dataframe.
        Default is False.

    output : str, optional
        Path where the final dataframe should be saved as CSV.
        Default is "results/output.csv".

    Returns
    -------
    tuple
        complete : pandas.DataFrame
            Combined dataframe containing taxa abundances and optionally metadata.

        metadata_woT : pandas.DataFrame
            Preprocessed metadata dataframe.

        number_taxa : int
            Number of taxa/features included.

        dic_TargTax : dict
            Dictionary mapping generated target column names
            (e.g. "Target1") to taxa names.
    """
    logger.info(
        "Creating complete dataframe from prob=%s exo=%s tax=%s include_metadata=%s output=%s",
        path_prob,
        path_exo,
        path_tax,
        include_metadata,
        output,
    )
    index_list, tax = read_timeseries(path_tax, path_prob)
    metadata = read_data(path_exo)
    metadata_trans = metadata.T
    time = get_timeframe(tax)
    # Initialize the complete dataframe with the "Time" column
    complete = pd.DataFrame({"Time": time})
    # metadata_trans = pd.DataFrame({"Time": time})
    metadata_trans["Time"] = time
    number_taxa = len(tax.values)
    metadata_woT = preprocess_metadata(metadata_trans)
    dic_TargTax = {}
    for i in range(len(tax.values)):
        complete[f"Target{i+1}"] = tax.iloc[i].values
        # print(tax.index[i])
        dic_TargTax[f"Target{i+1}"] = tax.index[i]
        # print(i)
    # Add metadata features to the complete dataframe if include_metadata is True
    if include_metadata:
        for col in metadata_woT.columns:
            complete[col] = metadata_woT[col].reindex(complete["Time"]).values
    # Sort the complete dataframe by the "Time" column and reset the index
    complete.sort_values(by=["Time"], inplace=True, ignore_index=True)
    complete.fillna(0, axis=1, inplace=True)
    complete.to_csv(output)
    print(metadata_woT)
    metadata_woT.sort_values(by=["Time"], inplace=True, ignore_index=True)
    return complete, metadata_woT, number_taxa, dic_TargTax
