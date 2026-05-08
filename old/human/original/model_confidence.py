from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Embedding
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

plotpath = "../allGutFemale/"

# Load old model to get original predictions
# model_loaded = load_model('1LayerLSTm.h5')
# predictvalY = model_loaded.predict(Xval)
# predictTestX = model_loaded.predict(Xtest)
# predictTest = scaler.inverse_transform(predictTestX)
# testY = scaler.inverse_transform(Ytest)
# print(predictTest)


def fit_model(Xtrain, Ytrain, Xval, Yval):
    # define neural network model
    n_steps = 3
    n_features = Xtrain.shape[2]
    model = Sequential()
    model.add(LSTM(2048, activation="relu", input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(n_features))
    # compile the model and specify loss and optimizer
    model.compile(optimizer="adam", loss="mae", metrics=["accuracy", "mse"])
    # fit the model on the training dataset
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50)
    model.fit(
        Xtrain,
        Ytrain,
        validation_data=(Xval, Yval),
        epochs=400,
        verbose=0,
        batch_size=5,
        callbacks=[es],
    )
    return model


# fit an ensemble of models
def fit_ensemble(n_members, Xtrain, Xval, Ytrain, Yval, scaler):
    ensemble = []
    for i in range(n_members):
        # define and fit the model on the training set
        model = fit_model(Xtrain, Ytrain, Xval, Yval)
        # evaluate model on the test set
        yhat = scaler.inverse_transform(model.predict(Xval, verbose=0))
        # print('>%d, MAE: %.3f' % (i+1, mae))
        # store the model
        ensemble.append(model)
    return ensemble


# fit an ensemble of models and save the model for later use
def fit_save_ensemble(n_members, Xtrain, Xval, Ytrain, Yval, scaler):
    ensemble = []
    for i in range(n_members):
        # define and fit the model on the training set
        model = fit_model(Xtrain, Ytrain, Xval, Yval)
        # evaluate model on the test set
        yhat = scaler.inverse_transform(model.predict(Xval, verbose=0))
        # print('>%d, MAE: %.3f' % (i+1, mae))
        # store the model
        model.save(plotpath + "predictionInterval/1LayerLSTM" + str(i) + ".h5")
        ensemble.append(model)
    return ensemble


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, Xtest, Ytest, scaler, species, n_steps):
    # make predictions
    yhat_list = []
    for model in ensemble:
        yhat = scaler.inverse_transform(model.predict(Xtest, verbose=0))
        yhat_species = []
        y = 0
        # print(len(yhat[1]))
        while y < len(yhat[1]):
            lst2 = [item[y] for item in yhat]
            yhat_species.append(lst2)
            y += 1
        yhat_list.append(yhat_species)
    species_list = [[] for _ in range(len(species))]
    # species_list = np.empty(len(species)-1, dtype = np.object)
    for list_small in yhat_list:
        i = 0
        while i < len(species):
            species_list[i].append(list_small[i])
            i += 1
    species_list = asarray(species_list)
    species_list_new = []
    for sp_list in species_list:
        list_swapp = np.swapaxes(sp_list, 0, 1)
        species_list_new.append(list_swapp)
    # calculate 95% gaussian prediction interval
    # needs to be done for every time step and every bacterial genera
    list_yhat = []
    for list_num in species_list_new:
        list_lower = [np.nan] * n_steps
        list_upper = [np.nan] * n_steps
        list_mean = [np.nan] * n_steps
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
    # Calculate the prediction interval also on the predictin errors
    list_error = []
    for list_num in species_list_new:
        list_lower = [np.nan] * n_steps
        list_upper = [np.nan] * n_steps
        list_mean = [np.nan] * n_steps
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

    return list_yhat, list_error


def retrain_confidence_model(ensemble_path, trainX, trainY, valX, valY, savepath):
    ensemble = []
    for model_path in os.listdir(ensemble_path):
        if "h5" in model_path:
            path = ensemble_path + "/" + model_path
            model_loaded = keras.models.load_model(path)
            es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50)
            model_loaded.fit(
                trainX,
                trainY,
                validation_data=(valX, valY),
                epochs=400,
                verbose=0,
                batch_size=5,
                callbacks=[es],
            )
            model_loaded.save(savepath + "ensemblemodels/" + model_path)
            ensemble.append(model_loaded)
    return ensemble


# make predictions with the ensemble_path that has been previously loaded and calculate a prediction interval
def predict_load_with_pi(ensemble, Xtest, scaler, species, n_steps):
    ensemble = []
    for model_path in ensemble:
        if "h5" in model_path:
            path = ensemble_path + "/" + model_path
            model_loaded = keras.models.load_model(path)
            ensemble.append(model_loaded)
    # make predictions
    yhat_list = []
    for model in ensemble:
        yhat = scaler.inverse_transform(model.predict(Xtest, verbose=0))
        yhat_species = []
        y = 0
        # print(len(yhat[1]))
        while y < len(yhat[1]):
            lst2 = [item[y] for item in yhat]
            yhat_species.append(lst2)
            y += 1
        yhat_list.append(yhat_species)
    species_list = [[] for _ in range(len(species))]
    # species_list = np.empty(len(species)-1, dtype = np.object)
    for list_small in yhat_list:
        i = 0
        while i < len(species):
            species_list[i].append(list_small[i])
            i += 1
    species_list = asarray(species_list)
    species_list_new = []
    for sp_list in species_list:
        list_swapp = np.swapaxes(sp_list, 0, 1)
        species_list_new.append(list_swapp)
    # calculate 95% gaussian prediction interval
    # needs to be done for every time step and every bacterial genera
    list_yhat = []
    for list_num in species_list_new:
        list_lower = [np.nan] * n_steps
        list_upper = [np.nan] * n_steps
        list_list = []
        for liste in list_num:
            interval = 1.96 * liste.std()
            lower, upper = liste.mean() - interval, liste.mean() + interval
            list_upper.append(upper)
            list_lower.append(lower)
        list_list.append(list_upper)
        list_list.append(list_lower)
        list_yhat.append(list_list)
    return list_yhat
