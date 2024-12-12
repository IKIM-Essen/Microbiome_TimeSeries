from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Embedding
from keras.losses import MeanSquaredError 
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, get_scorer_names
from matplotlib import pyplot
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.inspection import permutation_importance
from mlxtend.evaluate import feature_importance_permutation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.compat.v1.keras.backend import get_session
from datetime import datetime, timedelta, date

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor

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
    first_df = pd.read_csv(path_base, header = None, index_col=0)
    data_df = pd.read_csv(sample_path, header = 0, sep = "\t", index_col = "taxonomy")
    data_df_reduced = data_df.groupby(data_df.index).sum()
    for i in data_df_reduced.index:
        if "Unassigned" not in i:
            data_df_reduced.index = data_df_reduced.index.str.split("; s__").str[0]
        if "Unassigned" in i:
            data_df_reduced.drop(index=i, inplace=True)
    data_df_reduced = data_df_reduced.groupby(data_df_reduced.index).sum()
    index_list = data_df_reduced.index.tolist()
    #print(index_list)
    result = pd.concat([first_df, data_df_reduced], axis = 1)
    result.fillna(0, inplace = True)
    #print(result)
    #result.to_csv("complete_df.tsv", sep = "\t")
    return index_list, result

def read_reduce_dataframe(path):
    data_df = pd.read_csv(path, header = 0, sep = "\t", index_col = "taxonomy")
    data_df_reduced = data_df.groupby(data_df.index).sum()
    for i in data_df_reduced.index:
        if "Unassigned" not in i:
            data_df_reduced.index = data_df_reduced.index.str.split("; s__").str[0]
        if "Unassigned" in i:
            data_df_reduced.drop(index=i, inplace=True)
    data_df_reduced = data_df_reduced.groupby(data_df_reduced.index).sum()
    index_list = data_df_reduced.index.tolist()
    #print("reduced")
    return index_list,data_df_reduced


def read_exogenous(path):
    exogenous = pd.read_csv(path, header = None, sep = "\t")
    exogenous.drop(0, axis = 1, inplace = True)
    #print(exogenous)
    return exogenous

def get_timeframe(df):
    timelist = df.columns.values.tolist()
    return timelist

def create_complete_df(path_proband,path_tax,path_exo):
    index_list,tax = merge_dataframes(path_tax, path_proband)
    #index_list,tax = read_reduce_dataframe(path_proband)
    #exo = read_exogenous(path_exo)
    time = get_timeframe(tax)
    complete = pd.DataFrame({"Time": time})
    number_taxa = len(tax.values)
    for i in range(len(tax.values)):
        complete[f"Target{i+1}"] = tax.iloc[i].values
        #print(i)
    #complete["Exo"] = exo.iloc[0]
    return complete,number_taxa

complete, num_taxa = create_complete_df("/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/female_bacteroides.txt","/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/bacteria_list_final4.tsv","/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/female_shannon.txt")

#print(complete)
lag_targets =0
for y in range(num_taxa):
    for i in range(1, 4):
        complete[f'Target{y+1}_Lag_{i}'] = complete[f'Target{y+1}'].shift(i)
        lag_targets += 1
complete = complete.dropna()


scaler = MinMaxScaler(feature_range=(0, 1))
complete_woT = complete.drop(['Time'], axis=1)
print(complete_woT.shape)
scaled_data = scaler.fit_transform(complete_woT)


# Prepare data for LSTM
X_LSTM = np.array(scaled_data[:,num_taxa:(num_taxa+lag_targets)])
#print(X.shape)
X_LSTM = X_LSTM.reshape(X_LSTM.shape[0], 1, X_LSTM.shape[1])  # Reshape for LSTM input
y_LSTM = scaled_data[:, 0:num_taxa]

train_size = int(len(X_LSTM) * 0.8)
X_train_L, X_test_L = X_LSTM[0:train_size], X_LSTM[train_size:]
y_train_L, y_test_L = y_LSTM[0:train_size], y_LSTM[train_size:]

# Build LSTM model with an exogenous variable for two targets
LSTM_m = Sequential()
LSTM_m.add(LSTM(50, input_shape=(X_train_L.shape[1], X_train_L.shape[2])))
LSTM_m.add(Dense(num_taxa, name='target'))  # Output layer for Target_1
#model.add(Dense(1, name='target_2'))  # Output layer for Target_2
LSTM_m.compile(loss='mean_absolute_error', optimizer='adam')

# Train the model
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
LSTM_m.fit(X_train_L, y_train_L, epochs=400, batch_size=1, verbose=2, callbacks = [es])

# Train Random Forest model
X_rf = scaled_data[:,num_taxa:(num_taxa+lag_targets)]
y_rf = scaled_data[:, 0:num_taxa]

X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(X_rf, y_rf, test_size=0.2, shuffle = False)

RF = RandomForestRegressor(n_estimators=100, criterion = "absolute_error")
RF.fit(X_train_RF, y_train_RF)

predictions_LSTM = LSTM_m.predict(X_test_L)
predictions_RF = RF.predict(X_test_RF)

ensemble_predictions = (predictions_LSTM + predictions_RF) / 2

#lstm_predictions = scaler.inverse_transform(np.concatenate((lstm_predictions, X_lstm[:, 0, 1:]), axis=1))[:, 0]
lstm_predictions_reshaped = predictions_LSTM
#print(predictions_reshaped.shape)
#print(predictions.shape)
array = []
for i in range(predictions_LSTM.shape[1]):
    liste = predictions_LSTM[:, i].reshape(-1, 1)
    array.append(liste)
lstm_predictions_reshaped = np.concatenate((array),axis=1)
#print(predictions_reshaped.shape)
#print(X_test.shape)
lstm_predictions_reshaped = np.concatenate((lstm_predictions_reshaped,X_test_L.reshape(X_test_L.shape[0],X_test_L.shape[2])), axis=1)
#print(predictions_reshaped)
lstm_predictions_reshaped = scaler.inverse_transform(lstm_predictions_reshaped)[:, 0:num_taxa]

rf_predictions = scaler.inverse_transform(np.concatenate((predictions_RF, X_test_RF), axis=1))[:, 0:num_taxa]

ensemble_predictions = scaler.inverse_transform(np.concatenate((ensemble_predictions, X_test_RF), axis=1))[:, 0:num_taxa]

y_actual = scaler.inverse_transform(np.concatenate((y_test_L, X_test_L.reshape(X_test_L.shape[0],X_test_L.shape[2])), axis=1))[:, 0:num_taxa]


train_predictions_LSTM = LSTM_m.predict(X_train_L)
train_predictions_RF = RF.predict(X_train_RF)

ensemble_predictions_train = (train_predictions_LSTM + train_predictions_RF) / 2

ensemble_predictions_train = scaler.inverse_transform(np.concatenate((ensemble_predictions_train, X_train_RF), axis=1))[:, 0:num_taxa]
y_train_actual = scaler.inverse_transform(np.concatenate((y_train_L, X_train_L.reshape(X_train_L.shape[0],X_train_L.shape[2])), axis=1))[:, 0:num_taxa]

# Evaluate models
mae_lstm = mean_absolute_error(y_actual, lstm_predictions_reshaped)
mae_rf = mean_absolute_error(y_actual, rf_predictions)
mae_ensemble = mean_absolute_error(y_actual, ensemble_predictions)
rmse = math.sqrt(mean_squared_error(y_actual, ensemble_predictions))
nrmse = rmse/np.std(ensemble_predictions)

mae_train = mean_absolute_error(y_train_actual, ensemble_predictions_train)
rmse_train = math.sqrt(mean_squared_error(y_train_actual, ensemble_predictions_train))
nrmse_train = rmse_train/np.std(ensemble_predictions_train)

print(f'Mean Absolute Error - LSTM: {mae_lstm}')
print(f'Mean Absolute Error - Random Forest: {mae_rf}')
print(f'Mean Absolute Error - Ensemble train: {mae_train}')
print(f'Mean Absolute Error - Ensemble test: {mae_ensemble}')
print(f'RMSE - Ensemble train: {rmse_train}')
print(f'RMSE - Ensemble test: {rmse}')
print(f'NRMSE - Ensemble train: {nrmse_train}')
print(f'NRMSE - Ensemble test: {nrmse}')

