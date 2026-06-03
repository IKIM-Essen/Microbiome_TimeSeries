import yaml
import logging
import os


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