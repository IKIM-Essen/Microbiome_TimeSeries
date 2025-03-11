from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Embedding
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping

# from keras.preprocessing import sequences
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
from pickle import dump

import numpy as np
import pandas as pd
import matplotlib as mpl
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import shap

tf.compat.v1.disable_v2_behavior()

import model_confidence
import ml_plots
import shap_plots
import general_functions
import outlier

plotpath = "../allGutFemale/"

bacteria, df = general_functions.merge_dataframes(
    "bacteria_list_final4.tsv", "../table.from_biom_w_taxonomy-featcount_femalegut.txt"
)
df_trans, species, df_reduced_second = general_functions.read_csv(df)

ml_plots.time_series_analysis_plot(
    species, df_trans, plotpath + "plot_original.png", bacteria
)

# interpolate_df = general_functions.interpolate_samples(df_trans,1)
ml_plots.time_series_analysis_plot(
    species, df_trans, plotpath + "plot_interpolated.png", bacteria
)
df_family, families = ml_plots.plot_scaled_family(df_reduced_second)
# ml_plots.time_series_analysis_plot(families, df_family, "plot_scaled.png", bacteria)


# Getting for all timesteps a list holding all values for each species for that time step
# Appending those lists to another list
all_values_list = []
runvar = 0
while runvar < len(df_trans):
    list = np.array(df_trans.iloc[runvar].to_numpy())
    all_values_list.append(list)
    runvar = runvar + 1
# Changing the list of lists into a numpy.array, the input for the LSTM
all_values_list = np.array(all_values_list)
# Changing the type of the list
all_values_list = all_values_list.astype("float32")
# Applying a scaler to the array values, so the model works bette
# Needs to be changed back when analysing the models results'
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(all_values_list)
all_values_list = scaler.transform(all_values_list)
dump(scaler, open(plotpath + "scaler.pkl", "wb"))
print(all_values_list.shape)
# choose a number of time steps
n_steps = 3

# train-val-split
train_size = int(len(all_values_list) * 0.8)
# print(train_size)
train, test = (
    all_values_list[0:train_size, :],
    all_values_list[train_size : len(all_values_list), :],
)
train_final_size = int(len(train) * 0.8)
# Split the training and val set again into the batches used as input and output
train_final = train[0:train_final_size, :]
val = train[train_final_size : len(train), :]
Xtrain, Ytrain = general_functions.split_sequences(train_final, n_steps)
Xval, Yval = general_functions.split_sequences(val, n_steps)
Xtest, Ytest = general_functions.split_sequences(test, n_steps)

n_features = Xtrain.shape[2]
# Creating a model with two hidden layers with 100 LSTM blocks each
# model = Sequential()
# model.add(LSTM(2048, activation='relu', input_shape=(n_steps, n_features)))
# model.add(Dropout(0.2))
# model.add(LSTM(2048, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(n_features))
# model.compile(optimizer='adam', loss='mae', metrics = ["accuracy", "mse"])
# Fitting the model to the training data
# print(Xtrain.shape)
# print(Ytrain.shape)
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
# history=model.fit(Xtrain, Ytrain,  validation_data=(Xval,Yval), epochs=400, verbose=0, batch_size=5, callbacks = [es])
# print(history.history)
model, history = general_functions.create_model(
    "adam", "mae", n_steps, n_features, Xtrain, Ytrain, Xval, Yval
)
print(history.history)
# Save model
# model.save(plotpath+"1LayerLSTM.h5")
# Evaluation of model by plotting the loss function of the mean absolute error and the accuracy
train_score = model.evaluate(Xtrain, Ytrain, verbose=0)
val_score = model.evaluate(Xval, Yval, verbose=0)
ml_plots.plot_loss(history)
# Predicting outputs of train and val set and undoing the scaling for analysis
predictTrainY = model.predict(Xtrain)
print(Xtrain)
print(predictTrainY.shape)
# print(predictX.shape)

predictvalY = model.predict(Xval)
predictTestX = model.predict(Xtest)

# numpad = 50
# new_predictTrainY = general_functions.unpad(predictTrainY, species, numpad)
# new_trainY = general_functions.unpad(Ytrain, species, numpad)
# new_predictvalY = general_functions.unpad(predictvalY, species, numpad)
# new_valY = general_functions.unpad(Yval, species, numpad)
# new_predictTestX = general_functions.unpad(predictTestX, species, numpad)
# new_testY = general_functions.unpad(Ytest, species, numpad)


predictX = scaler.inverse_transform(predictTrainY)
trainY = scaler.inverse_transform(Ytrain)
# predictvalY = model.predict(Xval)
predictval = scaler.inverse_transform(predictvalY)
valY = scaler.inverse_transform(Yval)
# predictTestX = model.predict(Xtest)
predictTest = scaler.inverse_transform(predictTestX)
testY = scaler.inverse_transform(Ytest)

df_pred = pd.DataFrame(predictTest)
df_pred.to_csv(plotpath + "predictionTest.csv")


def get_only_predicted_bacteria(completebact, bacteria, Yretrans, predictedY):
    whole_bact = pd.read_csv(completebact, header=None)
    bacteria_list = whole_bact.values.tolist()
    bacteriaSeries = pd.Series(bacteria_list)
    # print(type(bacteria_list))
    # print(len(bacteria_list))
    # print(type(Yretrans))
    # print(len(Yretrans))
    # print(type(predictedY))
    # print(len(predictedY))
    Yretrans_flipped = np.transpose(Yretrans, (1, 0))
    predictedY_flipped = np.transpose(predictedY, (1, 0))

    df_normal = pd.DataFrame(
        Yretrans_flipped, columns=[f"col_{i}" for i in range(Yretrans_flipped.shape[1])]
    )
    df_normal.set_index(bacteriaSeries, inplace=True)
    df_predicted = pd.DataFrame(
        predictedY_flipped,
        columns=[f"col_{i}" for i in range(predictedY_flipped.shape[1])],
    )
    df_predicted.set_index(bacteriaSeries, inplace=True)

    df_normal = df_normal[~df_normal.index.isin(bacteria)]
    df_predicted = df_predicted[~df_predicted.index.isin(bacteria)]

    present_orig = df_normal.values
    present_predicted = df_predicted.values
    return present_orig, present_predicted


# print(predictX.shape)


"""
Creating a sklearn model out of a keras model
"""
# model_sklearn = model_sklearn(Xtrain, Ytrain, Xval, Yval, scaler, "adam", "mae", n_features, n_steps)

"""
Look for feature importance with shap
"""
shap_plots.shap_featureimportance_plots(model, Xtrain, Xtest, species)

# eval_dict = general_functions.get_metrics(trainY, predictX, valY, predictval, testY, predictTest)

# Load functions for confidence interval
n_members = 15
ensemble = model_confidence.fit_save_ensemble(
    n_members, Xtrain, Xval, Ytrain, Yval, scaler
)
# make predictions with prediction interval
yhat_list, error_hat = model_confidence.predict_with_pi(
    ensemble, Xtest, scaler.inverse_transform(Ytest), scaler, species, n_steps
)

present_orig, present_predicted = get_only_predicted_bacteria(
    "bacteria_list_final4.tsv", bacteria, testY, predictTest
)
print(type(present_orig))
array = []
b = 0
while b < len(yhat_list):
    array.append(yhat_list[b][2])
    b += 1
print(len(array))
array_list = np.array(array)
print(len(array_list))
print(array_list.shape)
print(present_predicted.shape)
print(array_list[1])
array_list_small = []
for liste in array_list:
    print(liste)
    liste = liste[~np.isnan(liste)]
    print(liste)
    array_list_small.append(liste)


df_median_pred = pd.DataFrame(array_list_small)
df_median_pred.to_csv(plotpath + "predictionTestMedian.csv")


def get_metrics(Y, predicted):
    mae_test = mean_absolute_error(Y, predicted)
    mse_test = mean_squared_error(Y, predicted)
    rmse_test = math.sqrt(mean_squared_error(Y, predicted))
    r2_test = r2_score(Y, predicted)
    nrmse_test = rmse_test / np.std(predicted)
    eval_dict = {
        "mae": mae_test,
        "mse": mse_test,
        "rmse": rmse_test,
        "r2": r2_test,
        "nrmse": nrmse_test,
    }
    return eval_dict


eval_dict = get_metrics(present_orig, array_list_small)
outlier = outlier.outlier_detection(yhat_list, test, species)

ml_plots.confidence_plot(
    plotpath + "confidence_plots/", species, predictTest, yhat_list, n_steps
)

ml_plots.prediction_plot(
    plotpath + "prediction_plots/",
    df_trans,
    predictX,
    predictval,
    predictTest,
    species,
    yhat_list,
    error_hat,
    n_steps,
    eval_dict,
    outlier,
)
ml_plots.plot_residuals(plotpath + "residual_plots/", trainY, predictX, species)

general_functions.eliminate_unnecessary(plotpath + "confidence_plots/", bacteria)
general_functions.eliminate_unnecessary(plotpath + "prediction_plots/", bacteria)
general_functions.eliminate_unnecessary(plotpath + "residual_plots/", bacteria)
general_functions.get_metrics(
    trainY,
    predictX,
    valY,
    predictval,
    present_orig,
    array_list_small,
    plotpath + "errormetrics.csv",
)
