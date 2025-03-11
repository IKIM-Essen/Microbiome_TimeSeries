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


def fit_model(X_train, y_train, X_val, y_val, n_features):
    # define neural network model
    n_steps = 3
    # n_features = X_train.shape[2]
    # print(X_train.shape)
    # print(y_train.shape)
    model = Sequential()
    model.add(
        LSTM(64, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2]))
    )  # , return_sequences = True
    model.add(Dropout(0.2))
    model.add(Dense(n_features, name="target"))
    # compile the model and specify loss and optimizer
    model.compile(optimizer="adam", loss="mae")
    # fit the model on the training dataset
    es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=10)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=400,
        verbose=0,
        batch_size=10,
        callbacks=[es],
    )
    return model


def fit_ensemble(
    n_members, X_train, X_val, X_test, y_train, y_val, scaler, num_features
):
    ensemble = []
    for i in range(n_members):
        # define and fit the model on the training set
        # model = keras.models.load_model("1LayerLSTM.h5")
        # print(X_train)
        # print(y_train)
        model = fit_model(X_train, y_train, X_val, y_val, num_features)
        predictions = model.predict(X_test)
        # print(predictions)
        # evaluate model on the test set
        # yhat = scaler.inverse_transform(model.predict(X_val, verbose=0))
        # print('>%d, MAE: %.3f' % (i+1, mae))
        # store the model
        ensemble.append(model)
    return ensemble


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X_train, Xtest, Ytest, scaler, species):
    # make predictions
    yhat_list = []
    for model in ensemble:
        predictions = model.predict(Xtest, verbose=0)
        # print(Xtest)
        # print(predictions[:,1])
        array = []
        for i in range(predictions.shape[1]):
            liste = predictions[:, i].reshape(-1, 1)
            array.append(liste)
        predictions_reshaped = np.concatenate((array), axis=1)
        # print(predictions_reshaped.shape)
        predictions_reshaped = np.concatenate(
            (predictions_reshaped, Xtest.reshape(Xtest.shape[0], Xtest.shape[2])),
            axis=1,
        )
        predictions_reshaped = scaler.inverse_transform(predictions_reshaped)[
            :, 0:species
        ]
        # print(predictions_reshaped[:,1])
        yhat = predictions_reshaped[:species]
        # print(yhat)
        yhat_species = []
        y = 0
        # print(len(yhat[1]))
        while y < len(yhat[1]):
            lst2 = [item[y] for item in yhat]
            yhat_species.append(lst2)
            y += 1
        yhat_list.append(yhat_species)
    species_list = [[] for _ in range(species)]
    # print(yhat_list[1])
    # print(len(species_list))
    # species_list = np.empty(len(species)-1, dtype = np.object)
    for list_small in yhat_list:
        i = 0
        while i < species:
            species_list[i].append(list_small[i])
            i += 1
    species_list = asarray(species_list)
    # print(len(species_list))
    species_list_new = []
    for sp_list in species_list:
        list_swapp = np.swapaxes(sp_list, 0, 1)
        species_list_new.append(list_swapp)
    # print(len(species_list_new))
    # calculate 95% gaussian prediction interval
    # needs to be done for every time step and every bacterial genera
    list_yhat = []
    for list_num in species_list_new:
        list_lower = [np.nan] * X_train.shape[1]
        list_upper = [np.nan] * X_train.shape[1]
        list_mean = [np.nan] * X_train.shape[1]
        list_list = []
        for liste in list_num:
            interval = 1.96 * liste.std()
            lower, upper = liste.mean() - interval, liste.mean() + interval
            mean = liste.mean()
            if lower < 0:
                lower = 0
            list_upper.append(upper)
            list_lower.append(lower)
            list_mean.append(mean)
        list_list.append(list_upper)
        list_list.append(list_lower)
        list_list.append(list_mean)
        list_yhat.append(list_list)
    # print(list_yhat[1])
    # Calculate the prediction interval also on the predictin errors
    list_error = []
    for list_num in species_list_new:
        list_lower = [np.nan] * X_train.shape[1]
        list_upper = [np.nan] * X_train.shape[1]
        list_mean = [np.nan] * X_train.shape[1]
        list_list = []
        i = 0
        comp_error_list = []
        while i < len(list_num):
            y_actual = Ytest[i]
            error_list = []
            for item in list_num[i]:
                residual = abs(item - y_actual)
                error_list.append(residual)
            error_liste = np.array(error_list)
            interval = 1.96 * error_liste.std()
            lower, upper = error_liste.mean() - interval, error_liste.mean() + interval
            # mean = liste.mean()
            if lower < 0:
                lower = 0
            list_upper.append(upper)
            list_lower.append(lower)
            i += 1
        list_list.append(list_upper)
        list_list.append(list_lower)
        list_error.append(list_list)

    return list_yhat  # , list_error
