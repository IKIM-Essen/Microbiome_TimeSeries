# Machine learning prediction on microbial time series

This repository contains scripts for creating and analyzing machine learning models for the prediction of microbial time series data. The goal is
to predict changes in microbial communities and forecast trends.

# Testing phase

This is the testing ground for the updated model and workflow architecture. If you are testing this, please follow these instructions:

First you have to clone this repository to your machine. The code has been currently only tested on a linux system, so this would definitely be recommended at this point.
You can either clone the repo with:

    git clone https://github.com/IKIM-Essen/Microbiome_TimeSeries.git

Or you can directly download the repo as zip file form GitHub and then unpack it.

Eitherway, when you installed the code, please change into the main project directory "Microbiome_TimeSeries".

Then create a conda environment from the yaml file environment.yaml with:

    conda create --file environment.yml -n YOUR_ENV_NAME

After that, activate it with:

    conda activate YOUR_ENV_NAME

Before you can test the whole thing, you need provide some data. There are currently three files that need to be provided. The main file is the timeseries.tsv. This holds the bacterial time series data from your analysis in the following format:

| taxonomy                  | timepoint1      | timepoint2      | timepoint3      |
|---------------------------|-----------------------------------------------------|
| genus name in Silva format| abundance1      | abundance2      | abundance3      |

Then there is the metadata.tsv which has the same format but holds all kinds of metadata or additional information to the data.

The third file is the taxa.tsv file. Here all the bacterial taxa on genus level should be listed. This is important, if you want to train and retrain the model on datasets potentially holding different bacteria, as the model needs to be trained on a fixed set of bacteria, even if they are only present in one dataset you want to analyze and not all of them.

To run the preprocessing of the data which creates a table that will be used as input please use the following command:

    python scripts/preprocessing.py --timeseries path/to/file --metadata /path/to/file --taxa path/to/file --include_metadata True --output path/to/output

If you do not set any of these flags, the code will use the default settings. The output should be a table located in results/tables/ .


## Features and Capabilities

The repository contains scripts for the creation, training and testing of different model architectures. Included are VARMA, Random Forest, LSTM and GRU
models. All models are capable of predicting one step into the future based on three previous time steps. LSTMs have been built to create a prediction
interval as base for outlier detection.
Calculated evaluation parameters include: MSE, RMSE, NRMSE
Model feature importance analysis has been done with SHAP.

## Datasets

The data used as input is 16S rRNA gene abundance data in form of a tsv file created from a BIOM file. The environmental model can process metadata as
well provided in a separate file.
Models have been trained and tested on published time series data from the following references: \
Caporaso et al., Genome Biology, 2011 \
David et al., Genome Biology, 2014 \
Kodera et al., Environ Microbiome, 2023 \
LaMartina et al., Microbiome, 2021

## Installation

To run the code a conda environment was used. The following packages and dependencies are necessary to run the code:
Pandas: 1.5.0
Tensorflow: 2.7.0
Scikit-Learn: 1.1.2
matplotlib: 3.4.3
SHAP: 0.41.0
keras: 2.7.0
statsmodel: 0.14.0

## Usage and Instructions

When all necessary packages are installed and the conda environment is activated, the code can be run. Please be careful, as this is a work in progress,
there are still some hardcoded paths. Please change them to a path existing on your machine. This will be changed in the future, please bear with me. The
code has been built and tested on a Linux machine.
