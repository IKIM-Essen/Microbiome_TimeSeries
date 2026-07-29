import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    get_scorer_names,
)
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Embedding
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import math
from pickle import dump, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from pickle import load

from src.preprocessing.scaling import inverse_scale_data, inverse_scale_data_attention
from src.utils.config import reshape, reshape_attention, reshape_original_attention

from src.model_building.create_models import (
    build_tcn,
    build_lstm,
    fit_model,
    ensemble_predict,
    fit_model_retraining,
)


def predict_interval(
    number_models,
    Xtrain,
    Ytrain,
    Xval,
    Yval,
    Xtest,
    X_meta_train,
    X_meta_val,
    X_meta_test,
    Ytest,
    scaler_path,
    species,
    tcn_path,
):
    # make predictions

    ensemble = []
    ensemble_2 = []
    yhat_list = []
    for i in range(number_models):
        # define and fit the model on the training set
        tcn_model, lstm_model, meta_model = fit_model(
            Xtrain,
            Ytrain,
            Xval,
            Yval,
            Xtrain.shape[2],
            tcn_path,
            save_model=False,
            X_meta_train=X_meta_train,
            X_meta_val=X_meta_val,
        )

        def ensemble_predict(X):
            y_tcn = tcn_model.predict(X)
            y_lstm_residual = lstm_model.predict(X)
            return y_tcn + y_lstm_residual

        # Make predictions on the test set
        predictions_val = ensemble_predict(Xval)

        y_val_tcn = reshape_attention(predictions_val, Xval, Yval.shape[1])

        y_val_tcn = inverse_scale_data_attention(y_val_tcn, scaler_path, Yval.shape[1])

        meta_val = meta_model.predict(X_meta_val)

        meta_val_reshape = reshape_attention(meta_val, Xval, Yval.shape[1])

        meta_val_res = inverse_scale_data_attention(
            meta_val_reshape, scaler_path, Yval.shape[1]
        )

        y_val_tcn = (y_val_tcn + meta_val_res) / 2

        yhat = y_val_tcn[: Yval.shape[1]]
        ensemble.append(yhat)
        yhat_species = []
        y = 0
        while y < len(yhat[1]):
            lst2 = [item[y] for item in yhat]
            yhat_species.append(lst2)
            y += 1
        yhat_list.append(yhat_species)

    stacked = np.stack(ensemble, axis=0)
    # Reshape to (3*14, 1544)
    reshaped = stacked.reshape(-1, Ytrain.shape[1])  # shape: (42, 1544)

    # Variance over timepoints and repetitions
    feature_variance = np.var(reshaped, axis=0, ddof=1)  # shape: (1544,)

    mean_prediction = np.mean(stacked, axis=0)
    residuals = mean_prediction - Yval
    # -- In case of standard deviation use ---
    mse = np.mean(residuals**2, axis=0)
    var_residual = mse
    print(residuals.shape)

    # -- In case of non-distributional use --
    # For asymmetric intervals compute quantiles of residuals:
    alpha = 0.05
    q_low = np.quantile(residuals, alpha / 2, axis=0)  # e.g. 0.025 quantile
    q_high = np.quantile(residuals, 1 - alpha / 2, axis=0)

    # Widths relative to mean
    #lower_width = np.abs(q_low)
    #upper_width = np.abs(q_high)

    min_width = 0.02
    
    lower_width = np.maximum(np.abs(q_low), min_width)
    upper_width = np.maximum(np.abs(q_high), min_width)

    print(q_low)
    print(q_high)
    """ Test for normality in case of normality assumption
    # 1. Shapiro-Wilk test
    feature_results = {}

    for feature_idx in range(residuals.shape[1]):
        vals = residuals[:, feature_idx]  # 14 values for this feature
        stat, p = shapiro(vals)
        
        feature_results[feature_idx] = {
            "stat": stat,
            "p_value": p,
            "normal": p > 0.05  # True if we fail to reject normality
        }

    # Summarize results
    num_normal = sum(v["normal"] for v in feature_results.values())
    num_non_normal = len(feature_results) - num_normal

    print(f"Out of {residuals.shape[1]} features:")
    print(f" - {num_normal} look Gaussian (p > 0.05)")
    print(f" - {num_non_normal} do NOT look Gaussian (p <= 0.05)")
    
    #print(var_residual.shape)
    z = norm.ppf(0.975)

    total_variance = feature_variance + var_residual
    std_total = np.sqrt(total_variance)
    """

    print("ensemble model shape")
    for i in range(number_models):
        predictions_test = ensemble_predict(Xtest)

        meta_test = meta_model.predict(X_meta_test)

        meta_test_reshape = reshape_attention(meta_test, Xtest, Ytest.shape[1])
        meta_test_reshape = inverse_scale_data_attention(
            meta_test_reshape, scaler_path, Ytest.shape[1]
        )

        """
        array = []
        for i in range(predictions_test.shape[1]):
            liste = predictions_test[:, i].reshape(-1, 1)
            array.append(liste)
        #print(len(array))
        predictions_reshaped = np.concatenate((array),axis=1)
        """

        y_test_tcn = predictions_test.reshape(
            predictions_test.shape[0], predictions_test.shape[2]
        )

        y_test_tcn = reshape_attention(y_test_tcn, Xtest, Ytest.shape[1])
        predictions_reshaped = inverse_scale_data_attention(
            y_test_tcn, scaler_path, Yval.shape[1]
        )

        predictions_reshaped = (predictions_reshaped - meta_test_reshape) / 2
        print(len(species))
        yhat = predictions_reshaped[: len(species)]
        ensemble_2.append(yhat)
    stacked_2 = np.stack(ensemble_2, axis=0)
    mean_prediction_2 = np.mean(stacked_2, axis=0)
    mean_prediction_2 = np.swapaxes(mean_prediction_2, 0, 1)
    list_yhat = []
    i = 0
    while i < len(species):
        list_error = []
        list_lower = [np.nan] * Xtrain.shape[1]
        # print(len(list_lower)) =27
        list_upper = [np.nan] * Xtrain.shape[1]
        list_mean = [np.nan] * Xtrain.shape[1]
        y = 0
        while y < mean_prediction_2.shape[1]:
            # print(mean_prediction_2.shape[1])
            # lower = mean_prediction_2[i][y] - std_total[i] * z
            # upper = mean_prediction_2[i][y] + std_total[i] * z
            lower = mean_prediction_2[i][y] - lower_width[i]
            upper = mean_prediction_2[i][y] + upper_width[i]
            mean = mean_prediction_2[i][y]
            upper, lower, mean = [max(0, x) for x in (upper, lower, mean)]
            list_lower.append(lower)
            list_upper.append(upper)
            list_mean.append(mean)
            y += 1
        # Find first non-nan index
        # first_valid = np.argmax(~np.isnan(list_upper))
        # Slice from there
        # list_upper = list_upper[first_valid:]
        list_error.append(list_upper)

        # first_valid = np.argmax(~np.isnan(list_lower))
        # Slice from there
        # list_lower = list_lower[first_valid:]
        list_error.append(list_lower)

        # first_valid = np.argmax(~np.isnan(list_mean))
        # Slice from there
        # list_mean = list_mean[first_valid:]
        list_error.append(list_mean)
        list_yhat.append(list_error)
        i += 1
    return list_yhat
