# from keras.models import Sequential
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    PolynomialFeatures,
)
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    get_scorer_names,
)
from matplotlib.offsetbox import AnchoredText
from pickle import load
from keras.callbacks import EarlyStopping

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
import outlier

plotpath = "../allGutFemale/"


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


def plot_data(
    totalValues, trained, predicted, species, n_steps, eval_dictionary, y_hat
):
    # Iterating through train and val results to create lists that can be plotted
    # with the original values to visualise the model
    b = 0
    predictTrain = []
    while b < len(species):
        lst2 = [item[b] for item in trained]
        empty = [np.nan] * n_steps
        empty.extend(lst2)
        # print(lst2)
        predictTrain.append(empty)
        b += 1
    i = 0
    predictListsX = []
    while i < len(species):
        lst2 = [item[i] for item in predicted]
        empty = [np.nan] * (len(predictTrain[1]) + n_steps)
        empty.extend(lst2)
        # print(lst2)
        predictListsX.append(empty)
        i += 1
    x_confidence = np.arange(len(predictTrain[1]), len(predictListsX[1]))
    y = 0
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    while y < len(species):
        # fig,ax = plt.subplots(figsize=(12,5), dpi=100)
        ax.set(
            title="Moving Pictures",
            xlabel="Date timepoints",
            ylabel="Number of sequences found",
        )
        ax.plot(totalValues[species[y]].values, label="actual")
        ax.plot(predictTrain[y], label="train")
        ax.plot(predictListsX[y], label="test")
        ax.plot(x_confidence, y_hat[y][0], label="upper")
        ax.plot(x_confidence, y_hat[y][1], label="lower")
        ax.fill_between(x_confidence, y_hat[y][0], y_hat[y][1], alpha=0.2)
        ax.legend(loc="upper left")
        mae_text = (
            "MAE : "
            + str(round(eval_dictionary["mae"], 2))
            + "\n"
            + "MSE: "
            + str((round(eval_dictionary["mse"], 2)))
            + "\n"
            + "R2: "
            + str((round(eval_dictionary["r2"], 2)))
            + "\n"
            + "RMSE test-set: "
            + str(round(eval_dictionary["rmse"], 2))
            + "\n"
            + "NRMSE: "
            + str(round(eval_dictionary["nrmse"], 2))
        )
        anchored_text = AnchoredText(
            mae_text,
            loc="upper left",
            frameon=False,
            bbox_to_anchor=(1.0, 1.0),
            bbox_transform=ax.transAxes,
            prop=dict(fontsize="small"),
        )
        ax.add_artist(anchored_text)
        plt.savefig(plotpath + "prediction_plots/" + species[y] + "-predictLSTMplt.png")
        plt.cla()
        y += 1


def eliminate_unnecessary(folder_path, bacteria):
    i = 0
    for f in os.listdir(folder_path):
        if ".png" in f:
            # print(f.rsplit("-")[0])
            # if f.split("_p")[0] not in list_bacteria:
            if bacteria.count(f.rsplit("-")[0]) == 0:
                os.remove(folder_path + f)
                i += 1
    print("Removed " + str(i) + " plots from folder.")


bacteria, df = general_functions.merge_dataframes(
    "bacteria_list_final4.tsv",
    "/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/allDonorA/table.from_biom_w_taxonomy-featcount_DonorA_small.txt",
)
print("Bacteria: " + str(len(bacteria)))
print(len(bacteria))
print(df.shape)
merged = general_functions.merge_sameday(df)
print(merged.shape)
df_new = general_functions.create_datetime_outof_int(merged)
print(df_new.shape)
df_trans, species, df_reduced_second = general_functions.read_csv(df_new)
print(df_trans.shape)
df_trans.to_csv(plotpath + "dftrans.csv")
interpolate_df = general_functions.interpolate_samples(df_trans, 1)
ml_plots.time_series_analysis_plot(
    species, interpolate_df, plotpath + "plot_interpolated.png", bacteria
)

all_values_list = []
runvar = 0
while runvar < len(interpolate_df):
    list = np.array(interpolate_df.iloc[runvar].to_numpy())
    all_values_list.append(list)
    runvar = runvar + 1
# Changing the list of lists into a numpy.array, the input for the LSTM
all_values_list = np.array(all_values_list)
# Changing the type of the list
all_values_list = all_values_list.astype("float32")
# Applying a scaler to the array values, so the model works bette
# Needs to be changed back when analysing the models results'
# scaler = MinMaxScaler(feature_range=(0, 1))
# all_values_list = scaler.fit_transform(all_values_list)
scaler = load(
    open(
        "/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/allGutFemale/scaler.pkl",
        "rb",
    )
)
all_values_list = scaler.fit_transform(all_values_list)
train_size = int(len(all_values_list) * 0.8)
train, test = (
    all_values_list[0:train_size, :],
    all_values_list[train_size : len(all_values_list), :],
)
train_final_size = int(len(train) * 0.8)
train_final = train[0:train_final_size, :]
val = train[train_final_size : len(train), :]
print(train_final.shape)
print(val)
print("Creating model!")
n_steps = 3
trainX, trainY = general_functions.split_sequences(train_final, n_steps)
valX, valY = general_functions.split_sequences(val, n_steps)
X, Y = general_functions.split_sequences(test, n_steps)

model = keras.models.load_model(
    "/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/multitraining/DonorB/1LayerLSTM_retrained.h5"
)
es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50)
history = model.fit(
    trainX,
    trainY,
    validation_data=(valX, valY),
    epochs=400,
    verbose=0,
    batch_size=5,
    callbacks=[es],
)
print(history.history.keys())
ml_plots.plot_loss(history)

model.save(plotpath + "1LayerLSTM_retrained.h5")

predictTrainY = model.predict(trainX)
predictValY = model.predict(valX)
predictY = model.predict(X)

TrainRetrans = scaler.inverse_transform(trainY)
trainYRetrans = scaler.inverse_transform(predictTrainY)

ValYRetrans = scaler.inverse_transform(valY)
predictValYRetrans = scaler.inverse_transform(predictValY)

Yretrans = scaler.inverse_transform(Y)
predictedY = scaler.inverse_transform(predictY)
df_pred = pd.DataFrame(predictedY)
df_pred.to_csv(plotpath + "predictionTest.csv")


ensemble = model_confidence.retrain_confidence_model(
    "/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/multitraining/DonorB/ensemblemodels",
    trainX,
    trainY,
    valX,
    valY,
    plotpath,
)
y_hat, error_hat = model_confidence.predict_with_pi(
    ensemble, X, scaler.inverse_transform(Y), scaler, species, n_steps
)
# ml_plots.prediction_plot(interpolate_df, predictX, predictval, predictTest, species, yhat_list, n_steps, eval_dict)


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


present_orig, present_predicted = get_only_predicted_bacteria(
    "bacteria_list_final4.tsv", bacteria, Yretrans, predictedY
)
print(type(present_orig))
array = []
b = 0
while b < len(y_hat):
    array.append(y_hat[b][2])
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
eval_dict = get_metrics(present_orig, array_list_small)
outlier = outlier.outlier_detection(y_hat, test, species)
ml_plots.prediction_plot(
    plotpath + "prediction_plots/",
    interpolate_df,
    trainYRetrans,
    predictValYRetrans,
    predictedY,
    species,
    y_hat,
    error_hat,
    n_steps,
    eval_dict,
    outlier,
)
general_functions.eliminate_unnecessary(plotpath + "prediction_plots/", bacteria)
shap_plots.shap_featureimportance_plots(model, trainX, X, species)
general_functions.get_metrics(
    TrainRetrans,
    trainYRetrans,
    ValYRetrans,
    predictValYRetrans,
    present_orig,
    array_list_small,
    plotpath + "errormetrics.csv",
)
