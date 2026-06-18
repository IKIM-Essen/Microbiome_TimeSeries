import yaml
import logging
import os
import keras
import pandas as pd
from pathlib import Path


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
            raise ValueError("Prediction interval arrays must be the same length for each species")
        for time_idx in range(len(mean)):
            records.append({
                "species": species_name,
                "timepoint": time_idx,
                "lower": lower[time_idx],
                "upper": upper[time_idx],
                "mean": mean[time_idx],
            })
    return pd.DataFrame.from_records(records)