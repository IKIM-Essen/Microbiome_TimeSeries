from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Embedding, GRU
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    PolynomialFeatures,
)
from numpy import asarray
from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from pickle import load


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(predictions, Xtrain, species, nrmse):
    # make predictions
    yhat = predictions
    # print(yhat)
    yhat_species = []
    y = 0
    # print(len(yhat[1]))
    while y < len(yhat[1]):
        lst2 = [item[y] for item in yhat]
        yhat_species.append(lst2)
        y += 1
    # print(yhat_species)
    # yhat_list.append(yhat_species)
    species_list = [[] for _ in range(species)]
    # print(species_list)
    # print(yhat_list[1])
    # print(len(species_list))
    # species_list = np.empty(len(species)-1, dtype = np.object)
    # for list_small in yhat_species:
    #    i = 0
    #    while i < species:
    #        species_list[i].append(list_small[i])
    #        i += 1
    # species_list = asarray(species_list)
    # print(species_list)
    # print(len(species_list))
    # species_list_new = []
    # for sp_list in yhat_species:
    #    list_swapp = np.swapaxes(sp_list, 0,1)
    #    species_list_new.append(list_swapp)
    # print(len(species_list_new))
    # calculate 95% gaussian prediction interval
    # needs to be done for every time step and every bacterial genera
    list_yhat = []
    # print(len(yhat_species[1]))
    # print(Xtrain.shape[1])
    for list_num in yhat_species:
        list_lower = [np.nan] * Xtrain.shape[1]
        list_upper = [np.nan] * Xtrain.shape[1]
        list_mean = [np.nan] * Xtrain.shape[1]
        list_list = []
        for value in list_num:
            lower, upper = value - (value * 1.96 * nrmse), value + (
                value * 1.96 * nrmse
            )
            if lower < 0:
                lower = 0
            list_upper.append(upper)
            list_lower.append(lower)
        # print(list_upper)
        list_list.append(list_upper)
        list_list.append(list_lower)
        list_yhat.append(list_list)
    # print(list_yhat)
    # print(len(list_yhat[1]))

    return list_yhat
