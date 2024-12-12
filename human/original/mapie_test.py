from keras.models import Sequential
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, PolynomialFeatures
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, get_scorer_names
from matplotlib.offsetbox import AnchoredText
from pickle import load

import tensorflow as tf
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import keras
import math
import os

import general_functions
import ml_plots
import model_confidence
import shap_plots

from mapie.metrics import (regression_coverage_score,
                           regression_mean_width_score)
from mapie.subsample import BlockBootstrap
from mapie.regression import MapieTimeSeriesRegressor

plotpath="../allGutFemale/"

bacteria, df = general_functions.merge_dataframes("bacteria_list_final4.tsv", "/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/allDonorA/table.from_biom_w_taxonomy-featcount_DonorA_small.txt")
print("Bacteria: "+ str(len(bacteria)))
print(len(bacteria))
print(df.shape) 
merged = general_functions.merge_sameday(df)
print(merged.shape) 
df_new=general_functions.create_datetime_outof_int(merged)
print(df_new.shape) 
df_trans, species, df_reduced_second = general_functions.read_csv(df_new)
print(df_trans.shape) 
df_trans.to_csv(plotpath+"dftrans.csv")
interpolate_df = general_functions.interpolate_samples(df_trans,1)
ml_plots.time_series_analysis_plot(species, interpolate_df, plotpath+"plot_interpolated.png", bacteria)

all_values_list = []
runvar = 0
while runvar < len(interpolate_df):
    list = np.array(interpolate_df.iloc[runvar].to_numpy())
    all_values_list.append(list)
    runvar = runvar +1
# Changing the list of lists into a numpy.array, the input for the LSTM
all_values_list = np.array(all_values_list)
# Changing the type of the list
all_values_list = all_values_list.astype('float32')
# Applying a scaler to the array values, so the model works bette
# Needs to be changed back when analysing the models results'
#scaler = MinMaxScaler(feature_range=(0, 1))
#all_values_list = scaler.fit_transform(all_values_list) 
scaler = load(open('/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/allGutFemale/scaler.pkl', 'rb'))
all_values_list = scaler.fit_transform(all_values_list) 
train_size = int(len(all_values_list) * 0.8)
train, test = all_values_list[0:train_size,:], all_values_list[train_size:len(all_values_list),:]
train_final_size = int(len(train) * 0.8)
train_final = train[0:train_final_size,:]
val = train[train_final_size:len(train),:]
print(train_final.shape)
print(val)
print("Creating model!")
n_steps = 3
trainX,trainY = general_functions.split_sequences(train_final, n_steps)
valX,valY = general_functions.split_sequences(val, n_steps)
X,Y = general_functions.split_sequences(test, n_steps)

model = keras.models.load_model("/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/multitraining/DonorB/1LayerLSTM_retrained.h5")
history = model.fit(trainX, trainY, validation_data=(valX,valY), epochs=400, verbose=0, batch_size=5)
print(history.history.keys())
ml_plots.plot_loss(history)


for time in trainX:
    

one_trainX = trainX[1][1]
print(one_trainX)
print(one_trainX.shape)
one_trainY = trainY[1][1]



alpha = 0.05
gap = 1
cv_mapiets = BlockBootstrap(
    n_resamplings=10, n_blocks=10, overlapping=False, random_state=59
)
mapie_enbpi = MapieTimeSeriesRegressor(
    model, method="enbpi", cv=cv_mapiets, agg_function="mean", n_jobs=-1
)
mapie_enbpi = mapie_enbpi.fit(one_trainX, one_trainY)
y_pred_npfit, y_pis_npfit = mapie_enbpi.predict(
    X, alpha=alpha, ensemble=True, optimize_beta=True
)
coverage_npfit = regression_coverage_score(
    Y, y_pis_npfit[:, 0, 0], y_pis_npfit[:, 1, 0]
)
width_npfit = regression_mean_width_score(
    y_pis_npfit[:, 0, 0], y_pis_npfit[:, 1, 0]
)
y_preds = [y_pred_npfit]
y_pis = [y_pis_npfit]
coverages = [coverage_npfit]
widths = [width_npfit]

fig, axs = plt.subplots(
    nrows=1, ncols=1, figsize=(14, 8), sharey="row", sharex="col"
)
ax.set_ylabel("Hourly demand (GW)")
ax.plot(
    trainY[int(-len(Y)/2):],
    lw=2,
    label="Training data", c="C0"
)
ax.plot(Y, lw=2, label="Test data", c="C1")
ax.plot(
    Y.index, y_preds[i], lw=2, c="C2", label="Predictions"
)
ax.fill_between(
    Y.index,
    y_pis[i][:, 0, 0],
    y_pis[i][:, 1, 0],
    color="C2",
    alpha=0.2,
    label="Prediction intervals",
)
title = f"EnbPI, {w} update of residuals. "
title += f"Coverage:{coverages[i]:.3f} and Width:{widths[i]:.3f}"
ax.set_title(title)
ax.legend()
fig.tight_layout()
plt.show()
figure.savefig("mapie_test_fig.png", bbox_extra_artists=(lgd,), bbox_inches='tight')