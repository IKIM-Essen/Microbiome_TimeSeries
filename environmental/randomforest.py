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
    exogenous = pd.read_csv(path, header = 0, sep = "\t", index_col = "Index")
    #exogenous.drop(0, axis = 1, inplace = True)
    #print(exogenous)
    return exogenous

def get_timeframe(df):
    timelist = df.columns.values.tolist()
    return timelist

def create_complete_df(path_proband,path_tax,path_exo):
    #index_list,tax = merge_dataframes(path_tax, path_proband)
    index_list,tax = read_reduce_dataframe(path_proband)
    #exo = read_exogenous(path_exo)
    #exo_trans = exo.T
    time = get_timeframe(tax)
    complete = pd.DataFrame({"Time": time})
    number_taxa = len(tax.values)
    dic_TargTax = {}
    for i in range(len(tax.values)):
        complete[f"Target{i+1}"] = tax.iloc[i].values
        dic_TargTax[f"Target{i+1}"] = tax.index[i]
        #print(i)
    #exo_trans.drop("Index", axis = 0, inplace = True)
    #print(exo_trans)
    #print(complete)
    #complete["temp"] = exo_trans[["average_temp"]].values
    #complete["precipitation"] = exo_trans[["rainfall_literpersquaremeter"]].values
    #complete["flow"] = exo_trans[["Flow_MGD"]].values
    #complete["ammonia"] = exo_trans[["Ammonia_mgL"]].values
    #complete["bod5"] = exo_trans[["BOD5_mgL"]].values
    #complete["phosphorus"] = exo_trans[["Phosphorus_mgL"]].values
    #complete["tss"] = exo_trans[["TSS_mgL"]].values
    return complete,number_taxa, dic_TargTax

#reduced = read_reduce_dataframe("/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/female_bacteroides.txt")
#print(reduced)
#exo = read_exogenous("/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/female_shannon.txt")
#time = get_timeframe(reduced)
complete, num_taxa, dic_Taxa = create_complete_df("Milwaukee/timeseries_JonesIsland.tsv","/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/bacteria_list_final4.tsv","Milwaukee/metadata_JonesIsland.tsv")
complete.sort_values(by=["Time"], inplace = True, ignore_index = True)
complete.fillna(0, axis = 1, inplace = True)
#print(complete)

for y in range(num_taxa):
    for i in range(1, 4):
        complete[f'Target{y+1}_Lag_{i}'] = complete[f'Target{y+1}'].shift(i)

#for i in range(1,4):
#    complete[f'Temp_Lag_{i}'] = complete['temp'].shift(i)
#    complete[f'Precip_Lag{i}'] = complete["precipitation"].shift(i)
#    complete[f'flow_Lag{i}'] = complete["flow"].shift(i)
#    complete[f'ammonia_Lag{i}'] = complete["ammonia"].shift(i)
#    complete[f'bod5_Lag{i}'] = complete["bod5"].shift(i)
#    complete[f'phosphorus_Lag{i}'] = complete["phosphorus"].shift(i)
#    complete[f'tss_Lag{i}'] = complete["tss"].shift(i)

complete = complete.dropna()
#print(complete)
X = complete.drop(['Time'], axis=1)
y = pd.DataFrame()
for i in range(num_taxa):
    X = X.drop([f"Target{i+1}"], axis = 1)
    y[f'Target{i+1}'] = complete[[f'Target{i+1}']]
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

print("Created dataframe, building model")
model = RandomForestRegressor(n_estimators=300, criterion = "absolute_error")
model.fit(X_train, y_train)
print("Model trained")

# Make predictions using the model
predictions = model.predict(X_test)
dictio = model.get_params
#print(dictio)

train_predictions = model.predict(X_train)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = math.sqrt(mean_squared_error(y_test, predictions))
nrmse = rmse/np.std(predictions)
real_list = y_test.values.tolist()
array = np.array(real_list)
r2 = r2_score(array,predictions)
#print(array.shape)
#print(predictions.shape)
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Sqaured Error: {rmse}')
print(f'Normalized Root Mean Sqaured Error: {nrmse}')
#print(f'R2: {r2}')
#for i in range(len(array)):
#    print(r2_score(array[i],predictions[i]))

mae_train = mean_absolute_error(y_train, train_predictions)
rmse_train = math.sqrt(mean_squared_error(y_train, train_predictions))
r2_train = r2_score(y_train, train_predictions)
nrmse_train = rmse_train/np.std(train_predictions)
print(mae_train)
print(rmse_train)
print(nrmse_train)
#print(r2_train)
#print(complete.iloc[129])
#print(complete.iloc[X_test.index]['Time'])

print("Creating ensemble")
n_predictions = 2
predictions_list = []

for _ in range(n_predictions):
    model = RandomForestRegressor(n_estimators=300, criterion = "absolute_error")
    model.fit(X_train, y_train)
    # Make predictions on the bootstrap sample
    predictions_list.append(model.predict(X_test))
    print("Model created")

predictions_list.append(predictions)

lower_bounds = np.percentile(predictions_list, 2.5, axis=0)
upper_bounds = np.percentile(predictions_list, 97.5, axis=0)

#print(lower_bounds)
#print(upper_bounds)






"""
male_complete, num_taxa_male = create_complete_df("/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/allGutMale/table.from_biom_w_taxonomy-featcount_mgut_smaller.txt","/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/bacteria_list_final4.tsv","/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/female_shannon.txt")

for y in range(num_taxa_male):
    for i in range(1, 4):
        male_complete[f'Target{y+1}_Lag_{i}'] = male_complete[f'Target{y+1}'].shift(i)

male_complete = male_complete.dropna()


#print(complete.shape)
#print(male_complete)

# Function to retrain a Random Forest model
def retrain_random_forest(original_model, new_data, num_taxa):
    # Assume 'Time' and 'Value' are the column names
    X_train = original_model.feature_importances_.reshape(1, -1)
    y_train = original_model.predict(X_train)
    #print(X_train)
    # Append new data to the original training set
    new_data_wo = new_data.drop(['Time'], axis = 1)
    b = pd.DataFrame()
    for y in range(num_taxa):
        new_data_wo = new_data_wo.drop([f'Target{y+1}'], axis = 1)
        b[f'Target{y+1}'] = new_data[f'Target{y+1}']
    new_X_train = np.concatenate([X_train, new_data_wo], axis=0)
    new_y_train = np.concatenate([y_train, b], axis=0)

    # Retrain the Random Forest model on the updated training set
    new_model = RandomForestRegressor(n_estimators=100, random_state=42)
    new_model.fit(new_X_train, new_y_train)

    return new_model

retrained_model = retrain_random_forest(model,male_complete, num_taxa_male)

X_male = male_complete.drop(['Time'], axis=1)
y_male = pd.DataFrame()
for i in range(num_taxa_male):
    X_male = X_male.drop([f"Target{i+1}"], axis = 1)
    y_male[f'Target{i+1}'] = male_complete[[f'Target{i+1}']]
#print(y)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_male, y_male, test_size=0.2, shuffle = False)

updated_predictions = retrained_model.predict(X_test_m)

updated_mae = mean_absolute_error(y_test_m, updated_predictions)
print(f'Updated Mean Absolute Error: {updated_mae}')
"""

#print(complete.loc[X_test.index[0]:])
#print(X_test.index[0])
#print(complete.index)
#print(y_test.index)
# Visualize predictions
#print(predictions)
#print(X_test.shape)
for i in range(num_taxa):
    plt.plot(complete['Time'], complete[f'Target{i+1}'], label='True Values')
    plt.plot(complete.loc[:int(X_test.index[0])-1]["Time"], train_predictions[:, i], label = 'Predictions train')
    plt.plot(complete.loc[X_test.index[0]:]["Time"], predictions[:, i], label='Predictions test')
    plt.plot(complete.loc[X_test.index[0]:]["Time"], lower_bounds[:, i], label='Lower')
    plt.plot(complete.loc[X_test.index[0]:]["Time"], upper_bounds[:, i], label='Upper')
    plt.fill_between(complete.loc[X_test.index[0]:]["Time"], upper_bounds[:, i], lower_bounds[:, i], alpha=0.2)
    plt.title('RandomForest '+dic_Taxa[f"Target{i+1}"], fontsize = 8)
    plt.xlabel('Time')
    plt.ylabel(f'Target{i+1}')
    plt.legend()
    plt.show()
    plt.savefig(f"Milwaukee/randomforest/plot{i+1}_RF_larger_woEx_JonesIsland.png")
    plt.cla()




"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Generate a hypothetical time series dataset
np.random.seed(42)
time_series = np.cumsum(np.random.normal(size=100))  # Cumulative sum of random normal values

# Create a DataFrame
df = pd.DataFrame({'Time': range(1, 101), 'Value': time_series})

# Feature engineering: Creating lag features
for i in range(1, 4):
    df[f'Value_Lag_{i}'] = df['Value'].shift(i)

# Drop rows with NaN values resulting from lag features
df = df.dropna()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Time', 'Value'], axis=1), df['Value'], test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Example: Create lag features for new data
new_data = pd.DataFrame({'Time': range(101, 111)})  # New time steps
for i in range(1, 4):
    new_data[f'Value_Lag_{i}'] = df['Value'].shift(i)  # Use the same lag features as in the training data

# Drop rows with NaN values resulting from lag features
new_data = new_data.dropna()

# Make predictions using the pre-trained Random Forest model
predictions = rf_model.predict(new_data.drop('Time', axis=1))

# Visualize predictions
plt.plot(df['Time'], df['Value'], label='Training Data')
plt.plot(new_data['Time'], predictions, label='Predictions')
plt.title('Time Series Prediction with Pre-trained Random Forest')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

"""
"""
new_data,male_taxa = create_complete_df("test_forecasting.txt","/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/bacteria_list_final4.tsv","/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/female_shannon.txt")
for y in range(male_taxa):
    for i in range(1, 4):
        new_data[f'Target{y+1}_Lag_{i}'] = complete[f'Target{y+1}'].shift(-i)
new_data = new_data.dropna()
X_new = new_data.drop(['Time'], axis=1)
y_new = pd.DataFrame()
for i in range(male_taxa):
    X_new = X_new.drop([f"Target{i+1}"], axis = 1)
    y_new[f'Target{i+1}'] = new_data[[f'Target{i+1}']]

print(X_new.shape)
print(y_new.shape)
new_predictions = model.predict(X_new)

print(new_predictions.shape)
print(new_predictions)

mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')
"""
