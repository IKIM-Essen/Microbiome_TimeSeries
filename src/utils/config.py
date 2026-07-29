import yaml
import logging
import os
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.utils import register_keras_serializable


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config


def extract_species(taxa):
    species = list(taxa.values)
    return species


def reshape(predictions):
    reshaped = predictions.reshape(predictions.shape[0], predictions.shape[2])
    return reshaped


def reshape_attention(predictions, original, num_taxa):
    array = []
    for i in range(predictions.shape[1]):
        liste = predictions[:, i].reshape(-1, 1)
        array.append(liste)
    predictions_reshaped = np.concatenate((array), axis=1)
    predictions_reshaped = predictions.reshape(predictions.shape[0], num_taxa)
    predictions_concat = np.concatenate(
        [predictions_reshaped, original.reshape(original.shape[0], original.shape[2])],
        axis=1,
    )
    return predictions_concat


def reshape_original_attention(y, X):
    y = np.concatenate([y, X.reshape(X.shape[0], X.shape[2])], axis=1)
    return y


def get_num_taxa(taxa):
    num_taxa = len(taxa.values)
    return num_taxa


def load_model_if_path(model_or_path):
    if isinstance(model_or_path, (str, Path)):
        return keras.models.load_model(model_or_path)
    return model_or_path


def prediction_interval_to_df(prediction_interval, species):
    records = []
    for species_idx, species_name in enumerate(species):
        upper, lower, mean = prediction_interval[species_idx]
        if not (len(upper) == len(lower) == len(mean)):
            raise ValueError(
                "Prediction interval arrays must be the same length for each species"
            )
        for time_idx in range(len(mean)):
            records.append(
                {
                    "species": species_name,
                    "timepoint": time_idx,
                    "lower": lower[time_idx],
                    "upper": upper[time_idx],
                    "mean": mean[time_idx],
                }
            )
    return pd.DataFrame.from_records(records)


def load_profile(path="config/profile.yaml"):
    """Load a pipeline profile YAML and return as a dict.

    Defaults to `config/profile.yaml` inside the repository. This is a thin
    wrapper around `load_config` to make intent clearer in the codebase.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profile file not found: {path}")
    return load_config(path)


def validate_profile(profile, required_keys=None):
    """Basic validation for a loaded profile.

    - `required_keys` can be a list of top-level keys that must exist (e.g.
      ['data','model','output']). Raises `ValueError` if a required key is
      missing. Returns True on success.
    """
    if required_keys is None:
        required_keys = ["data", "output"]

    missing = [k for k in required_keys if k not in profile]
    if missing:
        raise ValueError(f"Profile missing required keys: {missing}")

    # Optionally add simple structural checks
    if not isinstance(profile.get("data", {}), dict):
        raise ValueError("Profile 'data' section must be a mapping of paths")

    return True


@register_keras_serializable()
def zero_aware_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Smooth weighting: 1 near zero, gradually decreasing
    weight = tf.exp(-tf.abs(y_true) / 0.05)

    zero_penalty = tf.reduce_mean(weight * tf.square(y_pred))

    return mse + 0.1 * zero_penalty
