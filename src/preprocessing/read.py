import csv
import logging
import os
import re

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

CONFIG_PATH = os.path.join(ROOT_DIR, "config", "encodings.yaml")


def read_data(path):
    """Read a TSV with features in rows and unique sample identifiers in columns."""
    # Check the raw header before pandas can rename duplicate column labels.
    with open(path, newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"), [])
    samples = pd.Index(header[1:])
    if samples.empty or any(not sample.strip() for sample in samples):
        raise ValueError(f"Sample identifiers must be nonempty in {path}")
    if not samples.is_unique:
        duplicates = samples[samples.duplicated()].tolist()
        raise ValueError(f"Duplicate sample identifiers in {path}: {duplicates}")

    logger.info("Reading sample table from %s", path)
    data = pd.read_csv(path, header=0, sep="\t", index_col=0)
    logger.info("Read sample table shape %s", data.shape)
    return data


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
    """Encode and validate metadata with samples in rows and fields in columns."""
    config = load_config(CONFIG_PATH)
    season_map = config["season_map"]
    month_map = config["month_map"]
    cols_to_drop = config["columns_to_drop"]
    metadata_woT = metadata.copy()

    for field, mapping in (("season", season_map), ("month", month_map)):
        if field not in metadata_woT.columns:
            continue
        encoded_field = f"{field}_ord"
        if encoded_field in metadata_woT.columns:
            raise ValueError(
                f"Metadata field {encoded_field!r} is generated from {field!r}"
            )
        values = metadata_woT[field]
        names = values.astype("string").str.strip().str.lower()
        codes = pd.to_numeric(values, errors="coerce")
        encoded = names.map(mapping).fillna(codes.where(codes.isin(mapping.values())))
        unknown = values.notna() & encoded.isna()
        if unknown.any():
            invalid = values[unknown].unique().tolist()
            raise ValueError(f"Unknown values for metadata field {field!r}: {invalid}")
        metadata_woT[encoded_field] = encoded
        metadata_woT = metadata_woT.drop(columns=[field])

    metadata_woT = metadata_woT.drop(columns=cols_to_drop, errors="ignore")
    if metadata_woT.shape[1] == 0:
        raise ValueError("Metadata must contain at least one usable numeric field")
    for field in metadata_woT.columns:
        try:
            values = pd.to_numeric(metadata_woT[field], errors="raise").fillna(0)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Metadata field {field!r} must contain numeric values"
            ) from error
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Metadata field {field!r} must contain finite numeric values"
            )
        metadata_woT[field] = values.astype(float)
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
        Path to the metadata file. Only read when include_metadata is True.

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
            Preprocessed metadata aligned to complete; contains only Time when disabled.

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
    if not isinstance(include_metadata, (bool, np.bool_)):
        raise TypeError("include_metadata must be a Boolean")
    index_list, tax = read_timeseries(path_tax, path_prob)
    time = get_timeframe(tax)
    complete = pd.DataFrame({"Time": time})
    number_taxa = len(tax.values)
    dic_TargTax = {}
    for i in range(number_taxa):
        complete[f"Target{i+1}"] = tax.iloc[i].values
        dic_TargTax[f"Target{i+1}"] = tax.index[i]

    metadata_woT = pd.DataFrame({"Time": time})
    if include_metadata:
        if not path_exo:
            raise ValueError(
                "A metadata file is required when include_metadata is True"
            )
        metadata = read_data(path_exo)
        if metadata.index.hasnans or any(
            not str(field).strip() for field in metadata.index
        ):
            raise ValueError("Metadata field names must be nonempty")
        if not metadata.index.is_unique:
            raise ValueError("Duplicate metadata field names are not allowed")
        reserved = [
            field
            for field in metadata.index
            if field == "Time" or re.fullmatch(r"Target\d+", str(field))
        ]
        if reserved:
            raise ValueError(f"Metadata uses reserved field names: {reserved}")
        missing = tax.columns.difference(metadata.columns).tolist()
        extra = metadata.columns.difference(tax.columns).tolist()
        if missing or extra:
            raise ValueError(
                f"Metadata sample identifiers do not match the time series: "
                f"missing={missing}, extra={extra}"
            )
        # Reindex by identifiers, never by the position of metadata columns.
        features = preprocess_metadata(metadata.T.reindex(time))
        metadata_woT = features.reset_index(drop=True)
        metadata_woT.insert(0, "Time", time)
        for field in features.columns:
            complete[field] = features[field].to_numpy()

    complete.sort_values(by=["Time"], inplace=True, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    complete.to_csv(output)
    metadata_woT.sort_values(by=["Time"], inplace=True, ignore_index=True)
    return complete, metadata_woT, number_taxa, dic_TargTax
