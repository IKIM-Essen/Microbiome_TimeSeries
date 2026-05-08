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
import csv


def merge_dataframes(path_base, sample_path):
    first_df = pd.read_csv(path_base, header=None, index_col=0)
    data_df = pd.read_csv(sample_path, header=0, sep="\t", index_col="taxonomy")
    data_df_reduced = data_df.groupby(data_df.index).sum()
    for i in data_df_reduced.index:
        if "Unassigned" not in i:
            data_df_reduced.index = data_df_reduced.index.str.split("; s__").str[0]
        if "Unassigned" in i:
            data_df_reduced.drop(index=i, inplace=True)
    data_df_reduced = data_df_reduced.groupby(data_df_reduced.index).sum()
    index_list = data_df_reduced.index.tolist()
    # print(index_list)
    result = pd.concat([first_df, data_df_reduced], axis=1)
    result.fillna(0, inplace=True)
    # print(result)
    # result.to_csv("complete_df.tsv", sep = "\t")
    return index_list, result


def merge_sameday(df):
    grouped_df = df.groupby(lambda x: x.split(".")[0], axis=1).sum()

    print(grouped_df)
    return grouped_df


def read_csv(df):
    # df = pd.read_csv(filepath, delimiter = "\t", header = 0, index_col = "taxonomy")
    # print(df)
    # df.drop("#OTU ID", axis = 1, inplace = True)
    df_reduced = df.groupby(df.index).sum()
    # print(df_reduced)
    # for i in df_reduced.index:
    #    if "Unassigned" not in i:
    #        df_reduced.index = df_reduced.index.str.split("; s__").str[0]
    df_reduced_second = df_reduced.groupby(df_reduced.index).sum()
    for i in df_reduced_second.index:
        if "Chloroplast" in i:
            df_reduced_second.drop(index=i, inplace=True)
    species = df_reduced_second.index.unique()
    # print(len(df_reduced_second.index))
    # getting the unique indices
    # print(len(species))
    # Stripping the name of the lake from the index
    # Needs to be done in to steps, because otherwise the "1" at the beginning of the date qould be eliminated as well, because of the lstrip() function
    # Transforming the dataframe, so that the index is now the date
    df_trans = df_reduced_second.T
    # print(df_trans)
    # Transforming the index object to a datetime object and sorting the dataframe by date
    df_trans.index = pd.to_datetime(df_trans.index, yearfirst=True)
    df_trans.sort_index(axis=0, inplace=True)
    # print(df_trans)
    # correlation_rawdata = df_trans.corr()
    # correlation_rawdata.to_csv("rawdatacorrealtion.csv")
    return df_trans, species, df_reduced_second


def create_datetime_outof_int(dataframe):
    columns = list(dataframe.columns)
    test_list = [int(i) for i in columns]
    # print(type(test_list[1]))
    startdate = date(2011, 1, 1)
    datetime_indices = [startdate + timedelta(days=value) for value in test_list]
    # new_col_list.append("taxonomy")
    dataframe.columns = datetime_indices
    return dataframe


def interpolate_samples(dataframe, order):
    # asfreq creates an NaN value for all new samples
    upsampled = dataframe.resample("D", convention="start").asfreq()
    # print("upsamples:", upsampled)
    # NaN values get interpolated
    interpolated = upsampled.interpolate(method="spline", order=order)
    # print("interpolated:", interpolated)
    resampled = interpolated.resample("1D", convention="start").asfreq()
    # print(resampled)
    return resampled


# Model-Method for sklearn cross-val
def create_model(optimizer, loss, n_steps, n_features, Xtrain, Ytrain, Xval, Yval):
    model = Sequential()
    model.add(LSTM(2048, activation="relu", input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(n_features))
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy", "mse"])
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50)
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


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps=3):
    X, y = [], []
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    return X, y


def get_metrics(trainY, predictTrain, valY, predictVal, testY, predictTest, path):
    mae_train = mean_absolute_error(trainY, predictTrain)
    mse_train = mean_squared_error(trainY, predictTrain)
    mae_val = mean_absolute_error(valY, predictVal)
    mse_val = mean_squared_error(valY, predictVal)
    mae_test = mean_absolute_error(testY, predictTest)
    mse_test = mean_squared_error(testY, predictTest)
    r2 = r2_score(trainY, predictTrain)
    r2_test = r2_score(testY, predictTest)
    rmse_train = math.sqrt(mean_squared_error(trainY, predictTrain))
    rmse_test = math.sqrt(mean_squared_error(testY, predictTest))
    nrmse_train = rmse_train / np.std(predictTrain)
    nrmse_test = rmse_test / np.std(predictTest)
    eval_dict = {
        "mae_train": mae_train,
        "mae_val": mae_val,
        "mae_test": mae_test,
        "mse_train": mse_train,
        "mse_val": mse_val,
        "mse_test": mse_test,
        "r2_train": r2,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "nrmse_train": nrmse_train,
        "nrmse_test": nrmse_test,
    }
    with open(path, "w") as f:
        w = csv.DictWriter(f, eval_dict.keys())
        w.writeheader()
        w.writerow(eval_dict)
    return eval_dict


def model_sklearn(
    Xtrain, Ytrain, Xval, Yval, scaler, optimizer, loss, n_features, n_steps
):
    """
    Creating a sklearn model out of a keras model
    """
    model_sklearn = KerasRegressor(
        build_fn=create_model(
            optimizer, loss, n_steps, n_features, Xtrain, Ytrain, Xval, Yval
        ),
        epochs=40,
        batch_size=50,
        verbose=2,
    )
    tscv = TimeSeriesSplit(n_splits=5)
    score = cross_val_score(
        model_sklearn, Xtrain, Ytrain, cv=tscv, scoring="neg_mean_absolute_error"
    )
    print(get_scorer_names())
    print("Loss: {0:.3f} (+/- {1:.3f})".format(score.mean(), score.std()))
    model_sklearn.fit(Xtrain, Ytrain)
    sklearn_trainpred = model_sklearn.predict(Xtrain)
    print(sklearn_trainpred)
    skl_predictTrain = scaler.inverse_transform(sklearn_trainpred)
    skl_trainY = scaler.inverse_transform(Ytrain)
    sklearn_mae_train = mean_absolute_error(skl_trainY, skl_predictTrain)
    print("Sklearn MAE: ", sklearn_mae_train)

    model_fit = KerasRegressor(build_fn=create_model, verbose=2)
    batch_size = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    loss = ["mse", "binary_cross_entropy", "mae"]
    param_grid = dict(batch_size=batch_size)
    grid = GridSearchCV(estimator=model_fit, param_grid=param_grid)
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)
    grid_result = grid.fit(
        Xtrain, Ytrain, validation_data=(Xval, Yval), epochs=400, callbacks=[es]
    )
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    fit_trainpred = grid_result.predict(Xtrain)
    print(fit_trainpred)
    fit_predictTrain = scaler.inverse_transform(fit_trainpred)
    fit_trainY = scaler.inverse_transform(Ytrain)
    fit_mae_train = mean_absolute_error(fit_trainY, fit_predictTrain)
    print("Fit MAE: ", fit_mae_train)
    return model_sklearn


def unpad(array, species, num_pad):
    nill = num_pad - len(species)
    x = 0
    outerarray = []
    while x < len(array):
        print(array[x])
        smaller = array[x][nill:]
        print(smaller)
        outerarray.append(smaller)
        x += 1
    new_array = np.array(outerarray)
    print(type(new_array))
    print(new_array.shape)
    return new_array


def eliminate_unnecessary(folder_path, bacteria):
    i = 0
    for f in os.listdir(folder_path):
        if ".png" in f:
            # print(f.rsplit("-")[0])
            # if f.split("_p")[0] not in list_bacteria:
            if bacteria.count(f.rsplit("-", 1)[0]) == 0:
                os.remove(folder_path + f)
                i += 1
    print("Removed " + str(i) + " plots from folder.")
