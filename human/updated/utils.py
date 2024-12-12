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
import outlier

def create_datetime_outof_int(dataframe):
    columns = list(dataframe.columns)
    test_list = [int(i) for i in columns]
    #print(type(test_list[1]))
    startdate = date(2011, 1, 1)
    datetime_indices = [startdate + timedelta(days=value) for value in test_list]
    #new_col_list.append("taxonomy")
    dataframe.columns=datetime_indices
    return dataframe


def interpolate_samples(dataframe, order):
    # asfreq creates an NaN value for all new samples
    upsampled = dataframe.resample("D",convention='start').asfreq()
    #print("upsamples:", upsampled)
    # NaN values get interpolated
    interpolated = upsampled.interpolate(method="spline", order=order)
    #print("interpolated:", interpolated)
    resampled = interpolated.resample("1D",convention='start').asfreq()
    #print(resampled)
    return resampled


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps = 3):
	X, y = [], []
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	X = np.array(X)
	y = np.array(y)
	return X, y


def get_TrainValTest(all_values_list, n_steps):
    train_size = int(len(all_values_list) * 0.8)
    #print(train_size)
    train, test = all_values_list[0:train_size,:], all_values_list[train_size:len(all_values_list),:]
    train_final_size = int(len(train) * 0.8)
    # Split the training and val set again into the batches used as input and output
    train_final = train[0:train_final_size,:]
    val = train[train_final_size:len(train),:]
    Xtrain,Ytrain = general_functions.split_sequences(train_final, n_steps)
    Xval,Yval = general_functions.split_sequences(val, n_steps)
    Xtest,Ytest = general_functions.split_sequences(test, n_steps)
    return Xtrain,Ytrain,Xval,Yval,Xtest,Ytest


def eliminate_unnecessary(folder_path, bacteria):
    i = 0
    for f in os.listdir(folder_path):
        if ".png" in f:
            #print(f.rsplit("-")[0])
            #if f.split("_p")[0] not in list_bacteria:
            if bacteria.count(f.rsplit("-",1)[0]) == 0:
                os.remove(folder_path + f)
                i += 1
    print("Removed " + str(i) + " plots from folder.")


def get_only_predicted_bacteria(completebact, bacteria, Yretrans, predictedY):
    whole_bact = pd.read_csv(completebact, header = None)
    bacteria_list = whole_bact.values.tolist()
    bacteriaSeries = pd.Series(bacteria_list)
    #print(type(bacteria_list))
    #print(len(bacteria_list))
    #print(type(Yretrans))
    #print(len(Yretrans))
    #print(type(predictedY))
    #print(len(predictedY))
    Yretrans_flipped = np.transpose(Yretrans, (1, 0))
    predictedY_flipped = np.transpose(predictedY, (1, 0))

    df_normal = pd.DataFrame(Yretrans_flipped, columns=[f'col_{i}' for i in range(Yretrans_flipped.shape[1])])
    df_normal.set_index(bacteriaSeries, inplace = True)
    df_predicted = pd.DataFrame(predictedY_flipped, columns=[f'col_{i}' for i in range(predictedY_flipped.shape[1])])
    df_predicted.set_index(bacteriaSeries, inplace = True)

    df_normal = df_normal[~df_normal.index.isin(bacteria)]
    df_predicted = df_predicted[~df_predicted.index.isin(bacteria)]

    present_orig=df_normal.values
    present_predicted = df_predicted.values
    return present_orig, present_predicted

def scale_data(interpolate_df):
    # Getting for all timesteps a list holding all values for each species for that time step
    # Appending those lists to another list
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
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_values_list)
    all_values_list = scaler.transform(all_values_list) 
    dump(scaler, open(plotpath+'scaler.pkl', 'wb'))
    print(all_values_list.shape)
    return all_values_list, scaler


def scale_data(interpolate_df,scaler_path):
    # Getting for all timesteps a list holding all values for each species for that time step
    # Appending those lists to another list
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
    scaler = load(open(scaler, 'rb'))
    scaler.fit(all_values_list)
    all_values_list = scaler.transform(all_values_list) 
    #print(all_values_list.shape)
    return all_values_list, scaler

def get_median_prediction(present_orig,present_predicted):
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
    df_median_pred.to_csv(plotpath+"predictionTestMedian.csv")
    return df_median_pred