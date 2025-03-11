from numpy import array
from numpy import hstack
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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    get_scorer_names,
)
from matplotlib import pyplot
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.inspection import permutation_importance
from mlxtend.evaluate import feature_importance_permutation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.compat.v1.keras.backend import get_session
from datetime import datetime, timedelta, date

import numpy as np
import datetime as dt
import tensorflow as tf
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shap
import math


def create_model(optimizer, loss, n_steps, n_features, Xtrain, Ytrain, Xval, Yval):
    """ """
    model = Sequential()
    model.add(LSTM(2048, activation="relu", input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(n_features))
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy", "mse"])
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=200)
    history = model.fit(
        Xtrain,
        Ytrain,
        validation_data=(Xval, Yval),
        epochs=400,
        verbose=0,
        batch_size=5,
        callbacks=[es],
    )
    return model, history


def retrain_confidence_model(ensemble_path, trainX, trainY, valX, valY, savepath):
    """ """
    ensemble = []
    for model_path in os.listdir(ensemble_path):
        if "h5" in model_path:
            path = ensemble_path + "/" + model_path
            model_loaded = keras.models.load_model(path)
            model_loaded.fit(
                trainX,
                trainY,
                validation_data=(valX, valY),
                epochs=400,
                verbose=0,
                batch_size=5,
            )
            model_loaded.save(savepath + "ensemblemodels/" + model_path)
            ensemble.append(model_loaded)
    return ensemble
