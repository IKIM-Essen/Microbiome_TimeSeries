from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Embedding
from keras.losses import MeanSquaredError 
from keras.callbacks import EarlyStopping
#from keras.preprocessing import sequences
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

import general_functions
import utils

def merge_dataframes(path_base, sample_path):
    """
    """
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


def merge_sameday(df):
    grouped_df = df.groupby(lambda x: x.split(".")[0], axis=1).sum()

    print(grouped_df)
    return grouped_df

def read_csv(df):
    #df = pd.read_csv(filepath, delimiter = "\t", header = 0, index_col = "taxonomy")
    #print(df)
    #df.drop("#OTU ID", axis = 1, inplace = True)
    df_reduced = df.groupby(df.index).sum()
    #print(df_reduced)
    #for i in df_reduced.index:
    #    if "Unassigned" not in i:
    #        df_reduced.index = df_reduced.index.str.split("; s__").str[0]
    df_reduced_second = df_reduced.groupby(df_reduced.index).sum()
    for i in df_reduced_second.index:
        if "Chloroplast" in i:
            df_reduced_second.drop(index=i, inplace=True)
    species = df_reduced_second.index.unique()
    #print(len(df_reduced_second.index))
    # getting the unique indices
    #print(len(species))
    # Stripping the name of the lake from the index
    # Needs to be done in to steps, because otherwise the "1" at the beginning of the date qould be eliminated as well, because of the lstrip() function  
    # Transforming the dataframe, so that the index is now the date
    df_trans = df_reduced_second.T
    #print(df_trans)
    # Transforming the index object to a datetime object and sorting the dataframe by date
    df_trans.index = pd.to_datetime(df_trans.index, yearfirst = True)
    df_trans.sort_index(axis=0, inplace=True)
    #print(df_trans)
    #correlation_rawdata = df_trans.corr()
    #correlation_rawdata.to_csv("rawdatacorrealtion.csv")
    return df_trans, species, df_reduced_second

def create_datetime(df):
    merged = merge_sameday(df)
    df_merged = utils.create_datetime_outof_int(merged)
    return df_merged