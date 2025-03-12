# Machine learning prediction on microbial time series

This repository contains scripts for creating and analyzing machine learning models for the prediction of microbial time series data. The goal is
to predict changes in microbial communities and forecast trends.

## Features and Capabilities

The repository contains scripts for the creation, training and testing of different model architectures. Included are VARMA, Random Forest, LSTM and GRU models. All models are capable of predicting one step into the future based on three previous time steps. LSTMs have been built to create a prediction interval as base for outlier detection. 
Calculated evaluation parameters include: MSE, RMSE, NRMSE
Model feature importance analysis has been done with SHAP.

## Datasets

The data used as input is 16S rRNA gene abundance data in form of a tsv file created from a BIOM file. The environmental model can process metadata as well provided in a separate file.
Models have been trained and tested on published time series data from the following references:
Caporaso et al., Genome Biology, 2011
David et al., Genome Biology, 2014
Kodera et al., Environ Microbiome, 2023
LaMartina et al., Microbiome, 2021

## Installation

## Usage and Instructions