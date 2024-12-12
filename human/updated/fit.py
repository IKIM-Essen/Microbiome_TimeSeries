from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Embedding
from keras.losses import MeanSquaredError 
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, PolynomialFeatures
from numpy import asarray
from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras


def fit_model(Xtrain, Ytrain, Xval, Yval):
    # define neural network model
    n_steps = 3
    n_features = Xtrain.shape[2]
    model = Sequential()
    model.add(LSTM(2048, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(n_features))
    # compile the model and specify loss and optimizer
    model.compile(optimizer='adam', loss='mae', metrics = ["accuracy", "mse"])
    # fit the model on the training dataset
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    model.fit(Xtrain, Ytrain,  validation_data=(Xval,Yval), epochs=400, verbose=0, batch_size=5, callbacks = [es])
    return model

def fit_ensemble(n_members, Xtrain, Xval, Ytrain, Yval, scaler):
    ensemble = []
    for i in range(n_members):
        # define and fit the model on the training set
        model = fit_model(Xtrain, Ytrain, Xval, Yval)
        # evaluate model on the test set
        yhat = scaler.inverse_transform(model.predict(Xval, verbose=0))
        #print('>%d, MAE: %.3f' % (i+1, mae))
        # store the model
        ensemble.append(model)
    return ensemble

def fit_save_ensemble(n_members, Xtrain, Xval, Ytrain, Yval, scaler):
    ensemble = []
    for i in range(n_members):
        # define and fit the model on the training set
        model = fit_model(Xtrain, Ytrain, Xval, Yval)
        # evaluate model on the test set
        yhat = scaler.inverse_transform(model.predict(Xval, verbose=0))
        #print('>%d, MAE: %.3f' % (i+1, mae))
        # store the model
        model.save(plotpath+"predictionInterval/1LayerLSTM"+str(i)+".h5")
        ensemble.append(model)
    return ensemble