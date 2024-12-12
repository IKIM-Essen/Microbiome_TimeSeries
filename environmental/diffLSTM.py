import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, get_scorer_names
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

from mapie.metrics import (regression_coverage_score,
                           regression_mean_width_score)#coverage_width_based
from mapie.subsample import BlockBootstrap
from mapie.regression import MapieTimeSeriesRegressor

import keras
import ensemble
import shap
import shap_plots
import errorInterval

def func(evt):
    if legend.contains(evt):
        bbox = legend.get_bbox_to_anchor()
        bbox = Bbox.from_bounds(bbox.x0, bbox.y0+d[evt.button], bbox.width, bbox.height)
        tr = legend.axes.transAxes.inverted()
        legend.set_bbox_to_anchor(bbox.transformed(tr))
        fig.canvas.draw_idle()

def time_series_analysis_plot(species_series, dataframe_complete, filename, bacteria):
    # Creating colors for the plot
    np.random.seed(100)
    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(bacteria), replace=True)
    # Creating a plot displaying the OTU's found
    figure, ax = plt.subplots(figsize=(30,10))
    figure.legend(loc=2, prop={'size': 6})
    plt.rcParams["figure.figsize"] = (20,30)
    figure.legend(bbox_to_anchor=(1.1, 1.05))
    for i, y in enumerate(bacteria):
        if i > 0:
            ax.plot(dataframe_complete.index, dataframe_complete[y], color = mycolors[i], label=y)
    d = {"down" : 30, "up" : -30}
    handles, labels = ax.get_legend_handles_labels()
    lgd = figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.9, 0, 0.07, 0.9))  
	#loc='upper center', bbox_to_anchor=(0.5,-0.1)
    plt.gca().set(ylabel='Number of species found', xlabel='Date')
    plt.yticks(fontsize=12, alpha=.7)
    if len(dataframe_complete.index) < 50:
        plt.xticks(fontsize=10, rotation=45)
    if len(dataframe_complete.index) > 50:
        plt.xticks(fontsize=8, rotation=45)
    figure.canvas.mpl_connect("scroll_event", func)
    plt.title("Display of microbial species in patient dependent on time", fontsize=20)
    plt.show()
    figure.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_loss(history):
    figure, axs = plt.subplots(2, figsize=(20,10))
    axs[0].set_title('Loss Function')
    axs[1].set_title('Accuracy')
    axs[0].plot(history.history['loss'], label='train')
    #axs[0].plot(history.history['val_loss'], label='val')
    handles, labels = axs[0].get_legend_handles_labels()
    lgd = figure.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.5,1))
    axs[0].legend(bbox_to_anchor=(1.1, 1.05))
    axs[0].set(xlabel='Epochs', ylabel='Loss')
    axs[1].plot(history.history["acc"])
    #axs[1].plot(history.history["val_acc"])
    axs[1].legend(bbox_to_anchor=(1.1, 1.05))
    axs[1].set(xlabel='Epochs', ylabel='Accuracy')
    figure.savefig("Milwaukee/LSTM_new/loss_withMetadata.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()

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
    result = data_df_reduced.reindex(first_df.index)
    #print(result.shape)
    #print(index_list)
    #result = pd.concat([first_df, data_df_reduced], axis = 1)
    result.fillna(0, inplace = True)
    #print(result)
    #result.to_csv("complete_df.tsv", sep = "\t")
    return index_list, result

# Generate sample multivariate time series data with an exogenous variable for two targets
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

def create_complete_df(path_prob,path_tax,path_exo):
    index_list,tax = merge_dataframes(path_tax, path_prob)
    #index_list,tax = read_reduce_dataframe(path_prob)
    exo = read_exogenous(path_exo)
    exo_trans = exo.T
    time = get_timeframe(tax)
    complete = pd.DataFrame({"Time": time})
    number_taxa = len(tax.values)
    dic_TargTax = {}
    for i in range(len(tax.values)):
        complete[f"Target{i+1}"] = tax.iloc[i].values
        #print(tax.index[i])
        dic_TargTax[f"Target{i+1}"] = tax.index[i]
        #print(i)
    #print(complete)
    #print(exo_trans[["avg_temp"]].values)
    complete.sort_values(by=["Time"], inplace = True, ignore_index = True)
    #print(complete["Time"])
    #print(exo_trans.loc[complete["Time"],"Air_temp_F"])
    #complete["temp"] = exo_trans.loc[complete["Time"],"Air_temp_F"]
    complete["temp"]=exo_trans["average_temp"].reindex(complete["Time"]).values
    #print(complete["temp"])
    #complete["temp"] = exo_trans[["Air_temp_F"]].values
    #complete["precipitation"] = exo_trans.loc[complete["Time"],"Precip_48hrs_in"]
    #complete["precipitation"]=exo_trans["rainfall_literpersquaremeter"].reindex(complete["Time"]).values
    #print(exo_trans)
    #complete["precipitation"] = exo_trans[["Precip_48hrs_in"]].values
    #complete["flow"] = exo_trans[["Flow_MGD"]].values
    #complete["ammonia"] = exo_trans[["Ammonia_mgL"]].values
    #complete["bod5"] = exo_trans[["BOD5_mgL"]].values
    #complete["phosphorus"] = exo_trans[["Phosphorus_mgL"]].values
    #complete["tss"] = exo_trans[["TSS_mgL"]].values
    return complete, number_taxa, dic_TargTax

complete, num_taxa, dic_taxa = create_complete_df("/local/work/16S/snakemake_qiime/16S/MachineLearning/Wastewater/Dinslaken/timeseries.tsv", "/local/work/16S/snakemake_qiime/16S/MachineLearning/Wastewater/Dinslaken/index_only.tsv", "/local/work/16S/snakemake_qiime/16S/MachineLearning/Wastewater/Dinslaken/metadata.tsv")
#print(dic_taxa)

species = list(dic_taxa.values())
print(len(species))

#time_series_analysis_plot(species, complete, "plot_original.png", dic_taxa)

complete.sort_values(by=["Time"], inplace = True, ignore_index = True)
complete.fillna(0, axis = 1, inplace = True)
# Feature engineering: Creating lag features
lag_targets =0
for y in range(num_taxa):
    for i in range(1, 4):
        complete[f'Target{y+1}_Lag_{i}'] = complete[f'Target{y+1}'].shift(i)
        lag_targets += 1

for i in range(1,4):
    complete[f'Temp_Lag_{i}'] = complete['temp'].shift(i)
    lag_targets += 1
    #complete[f'Precip_Lag{i}'] = complete["precipitation"].shift(i)
    #lag_targets += 1
    #complete[f'flow_Lag{i}'] = complete["flow"].shift(i)
    #lag_targets += 1
    #complete[f'ammonia_Lag{i}'] = complete["ammonia"].shift(i)
    #lag_targets += 1
    #complete[f'bod5_Lag{i}'] = complete["bod5"].shift(i)
    #lag_targets += 1
    #complete[f'phosphorus_Lag{i}'] = complete["phosphorus"].shift(i)
    #lag_targets += 1
    #complete[f'tss_Lag{i}'] = complete["tss"].shift(i)
    #lag_targets += 1

#print(lag_targets)
# Drop rows with NaN values resulting from lag features
complete = complete.dropna()

#print(complete)
# Normalize the data
#scaler = scaler = load(open('Milwaukee/LSTM/scaler.pkl', 'rb'))
scaler = MinMaxScaler(feature_range=(0, 1))
complete_woT = complete.drop(['Time'], axis=1)
#scaler.fit(complete_woT)
scaled_data = scaler.fit_transform(complete_woT)
#dump(scaler, open('Milwaukee/LSTM_new/scaler.pkl', 'wb'))



#print(complete_woT.shape)

# Prepare data for LSTM
# careful, the plus two is becaused of the two exogenous variables, delete if necessary!
X = np.array(scaled_data[:,num_taxa:(num_taxa+lag_targets+1)])
#print(X.shape)
X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape for LSTM input
y = scaled_data[:, 0:num_taxa]
#print(y)
# Split the data into training and testing sets
train_size = int(len(X) * 0.7)
val_size = int(len(X)*0.8)
X_train, X_val, X_test = X[0:train_size], X[train_size:val_size], X[val_size:]
y_train, y_val, y_test = y[0:train_size], y[train_size:val_size], y[val_size:]

print("Read data, creating model")
#print(X_train.shape)
#print(y_train.shape)

# Build LSTM model with an exogenous variable for two targets
model = Sequential()
model.add(LSTM(2048, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))#, return_sequences = True
model.add(Dropout(0.2))
#model.add(LSTM(512, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(Dropout(0.2))
model.add(Dense(num_taxa, name='target'))  # Output layer for Targets
model.compile(loss='mean_absolute_error', optimizer='adam', metrics = ["accuracy"])
#model = keras.models.load_model("/local/work/16S/snakemake_qiime/16S/MachineLearning/Wastewater/Milwaukee/1LayerLSTM.h5")
print(X_train.shape)
print(len(species))
# Train the model
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
#history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=4, batch_size=5, verbose=0, callbacks = [es])
history = model.fit(X_train, y_train, epochs=400, batch_size=10, verbose=0)
model.summary()
#model.save("Milwaukee/LSTM_new/1LayerLSTM_JI.h5")
#print(X_test)
# Make predictions on the test set
predictions = model.predict(X_test)
predictions_train = model.predict(X_train)
#plot_loss(history)


#print(predictions[:,1])
# Inverse transform the predictions and actual values to the original scale
#print(predictions[1])
predictions_reshaped = predictions

#shap_plots.shap_featureimportance_plots(model, X_train, y_train, X_test, dic_taxa, complete_woT.columns[num_taxa:(num_taxa+lag_targets+2)])

ensemble_list = ensemble.fit_ensemble(4,X_train,X_val,X_test,y_train,y_val,scaler,num_taxa)
print("Ensemble generated!")
prediction_inter = ensemble.predict_with_pi(ensemble_list, X_train, X_test, y_test, scaler, num_taxa)
print("Errors calculated!")

print("Rescaling data")
array = []
for i in range(predictions.shape[1]):
    liste = predictions[:, i].reshape(-1, 1)
    array.append(liste)
predictions_reshaped = np.concatenate((array),axis=1)
print(predictions_reshaped.shape)
print(X_test.shape)
#predictions_reshaped = np.concatenate((predictions_reshaped,X_val.reshape(X_val.shape[0],X_val.shape[2])), axis=1)
predictions_reshaped = np.concatenate((predictions_reshaped,X_test.reshape(X_test.shape[0],X_test.shape[2])), axis=1)
#print(predictions_reshaped.shape)
predictions_reshaped = scaler.inverse_transform(predictions_reshaped)[:, 0:num_taxa]
y_test = scaler.inverse_transform(np.concatenate([y_test, X_test.reshape(X_test.shape[0],X_test.shape[2])], axis=1))[:, 0:num_taxa]

val_predictions = model.predict(X_val)

val_pred_reshaped = val_predictions
array_val = []
for i in range(val_predictions.shape[1]):
    liste = val_predictions[:,i].reshape(-1,1)
    array_val.append(liste)
val_pred_reshaped = np.concatenate((array_val),axis = 1)
val_pred_reshaped = np.concatenate((val_pred_reshaped,X_val.reshape(X_val.shape[0], X_val.shape[2])), axis = 1)
val_pred_reshaped = scaler.inverse_transform(val_pred_reshaped)[:, 0:num_taxa]


train_pred_reshaped = predictions_train
array_train = []
for i in range(predictions_train.shape[1]):
    liste = predictions_train[:, i].reshape(-1, 1)
    array_train.append(liste)
train_pred_reshaped = np.concatenate((array_train),axis=1)
#train_pred_reshaped = np.concatenate((train_pred_reshaped,X_val.reshape(X_val.shape[0],X_val.shape[2])), axis=1)
train_pred_reshaped = np.concatenate((train_pred_reshaped,X_train.reshape(X_train.shape[0],X_train.shape[2])), axis=1)
train_pred_reshaped = scaler.inverse_transform(train_pred_reshaped)[:, 0:num_taxa]

y_train = scaler.inverse_transform(np.concatenate([y_train, X_train.reshape(X_train.shape[0],X_train.shape[2])], axis=1))[:, 0:num_taxa]

#print(X_train.shape)
#print(y_train.shape)

#print(predictions_reshaped.shape)
#print(y_test.shape)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions_reshaped)
rmse = math.sqrt(mean_squared_error(y_test, predictions_reshaped))
nrmse = rmse/np.std(predictions_reshaped)
mae_train = mean_absolute_error(y_train, train_pred_reshaped)
rmse_train = math.sqrt(mean_squared_error(y_train, train_pred_reshaped))
nrmse_train = rmse_train/np.std(train_pred_reshaped)

print(complete.shape)

print(f'Mean Absolute Error: {mae_train}')
print(f'Root Mean Squared Error: {rmse_train}')
print(f'Normalized Root Mean Squared Error: {nrmse_train}')

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Normalized Root Mean Squared Error: {nrmse}')

#error = errorInterval.predict_with_pi(predictions_reshaped, X_train, num_taxa, nrmse)

"""
alpha = 0.05
gap = 1
cv_mapiets = BlockBootstrap(
    n_resamplings=10, n_blocks=10, overlapping=False, random_state=59
)

mapie_enbpi = MapieTimeSeriesRegressor(
    model, method="enbpi", cv=cv_mapiets, agg_function="mean", n_jobs=-1
)

mapie_aci = MapieTimeSeriesRegressor(
    model, method="aci", cv=cv_mapiets, agg_function="mean", n_jobs=-1
)
y_flattend = y_train.flatten()
print(y_flattend.shape)
X_flattend = X_train.flatten()
print(X_flattend.shape)
mapie_aci = mapie_aci.fit(X_flattend, y_flattend)

#mapie_enbpi = mapie_enbpi.fit(X_train, y_train)

y_pred_enbpi_npfit, y_pis_enbpi_npfit = mapie_enbpi.predict(
    X_test, alpha=alpha, ensemble=True, optimize_beta=True,
    allow_infinite_bounds=True
)
y_pis_enbpi_npfit = np.clip(y_pis_enbpi_npfit, 1, 10)
coverage_enbpi_npfit = regression_coverage_score(
    y_test, y_pis_enbpi_npfit[:, 0, 0], y_pis_enbpi_npfit[:, 1, 0]
)
width_enbpi_npfit = regression_mean_width_score(
    y_pis_enbpi_npfit[:, 0, 0], y_pis_enbpi_npfit[:, 1, 0]
)

# = coverage_width_based(
#    y_test, y_pis_enbpi_npfit[:, 0, 0],
#    y_pis_enbpi_npfit[:, 1, 0],
#    eta=10,
#    alpha=0.05
#)

y_enbpi_preds = [y_pred_enbpi_npfit, y_pred_enbpi_pfit]
y_enbpi_pis = [y_pis_enbpi_npfit, y_pis_enbpi_pfit]
coverages_enbpi = [coverage_enbpi_npfit, coverage_enbpi_pfit]
widths_enbpi = [width_enbpi_npfit, width_enbpi_pfit]
"""

#print("Plotting results")
#print(prediction_inter[1])
#print(complete['Target1'])

for i in range(num_taxa):
    # Visualize predictions for Targets
    plt.plot(complete['Time'], complete[f'Target{i+1}'], label='True Values')
    # plt.plot(complete.iloc[val_size:]['Time'], predictions_reshaped[:, i], label='Predictions test')
    plt.plot(complete.iloc[:train_size]['Time'], train_pred_reshaped[:, i], label='Training')
    plt.plot(complete.iloc[train_size:val_size]['Time'], val_pred_reshaped[:,i], label = 'Validation')
    plt.plot(complete.iloc[val_size:]['Time'],prediction_inter[i][0][1:len(prediction_inter[i][0])], label = "upper")
    plt.plot(complete.iloc[val_size:]['Time'],prediction_inter[i][1][1:len(prediction_inter[i][1])], label = "lower")
    plt.plot(complete.iloc[val_size:]['Time'],prediction_inter[i][2][1:len(prediction_inter[i][2])], label = "mean")
    #plt.plot(complete.iloc[val_size:]['Time'], predictions_reshaped[:,i], label = 'Test')
    plt.fill_between(complete.iloc[val_size:]['Time'],prediction_inter[i][0][1:len(prediction_inter[i][0])], prediction_inter[i][1][1:len(prediction_inter[i][1])], alpha=0.2)
    # plt.plot(complete['Time'], complete['temp'], label='Predictions train')
    plt.title('Multivariate Time Series '+dic_taxa[f"Target{i+1}"], fontsize = 6)
    plt.xlabel('Time')
    plt.ylabel(f'Target{i+1}')
    plt.legend()
    plt.savefig(f"Dinslaken/temponly/plot{i+1}_LSTM_test.png")
    plt.cla()

#model = keras.models.load_model("/local/work/16S/snakemake_qiime/16S/MachineLearning/Wastewater/1LayerLSTM_retrained.h5")
#scaler = load(open('/local/work/16S/snakemake_qiime/16S/MachineLearning/Wastewater/Milwaukee/LSTM_new/scaler.pkl', 'rb'))


def retrain_model(timeseries, taxalist, exo, model, scaler):
    print("Read new model data")
    new_timeseries, num_taxa, new_taxa = create_complete_df(timeseries, taxalist, exo)
    new_timeseries.sort_values(by=["Time"], inplace = True, ignore_index = True)
    print(new_timeseries)

    lag_targets =0
    for y in range(num_taxa):
        for i in range(1, 4):
            new_timeseries[f'Target{y+1}_Lag_{i}'] = new_timeseries[f'Target{y+1}'].shift(i)
            lag_targets += 1

    for i in range(1,4):
        new_timeseries[f'Temp_Lag_{i}'] = new_timeseries['temp'].shift(i)
        lag_targets += 1
        new_timeseries[f'Precip_Lag{i}'] = new_timeseries["precipitation"].shift(i)
        lag_targets += 1

    new_timeseries = new_timeseries.dropna()
    print("Scale new data")
    new_timeseries_woT = new_timeseries.drop(['Time'], axis=1)
    scaled_data = scaler.fit_transform(new_timeseries_woT)
    # Prepare data for LSTM
    X = np.array(scaled_data[:,num_taxa:(num_taxa+lag_targets+2)])
    #print(X.shape)
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape for LSTM input
    y = scaled_data[:, 0:num_taxa]
    train_size = int(len(X) * 0.7)
    val_size = int(len(X)*0.8)
    X_train, X_val, X_test = X[0:train_size], X[train_size:val_size], X[val_size:]
    y_train, y_val, y_test = y[0:train_size], y[train_size:val_size], y[val_size:]
    print("Fit Model")
    #model = keras.models.load_model("/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/multitraining/DonorB/1LayerLSTM_retrained.h5")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=400, verbose=0, batch_size=5, callbacks = [es])
    print(history.history.keys())
    #plot_loss(history)
    predictions = model.predict(X_test)
    predictions_train = model.predict(X_train)
    #plot_loss(history)
    model.save("1LayerLSTM_retrained.h5")
    #print("Create ensemble")
    #ensemble = model_confidence.retrain_confidence_model("/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/multitraining/DonorB/ensemblemodels", trainX, trainY, valX, valY, plotpath)
    #print("Create Interval")
    #y_hat, error_hat = model_confidence.predict_with_pi(ensemble, X, scaler.inverse_transform(Y), scaler, species, n_steps) 

    print("Rescaling data")
    #print(predictions_reshaped.shape)
    #print(predictions.shape)
    array = []
    for i in range(predictions.shape[1]):
        liste = predictions[:, i].reshape(-1, 1)
        array.append(liste)
    predictions_reshaped = np.concatenate((array),axis=1)
    #print(predictions_reshaped.shape)
    #print(X_test.shape)
    #predictions_reshaped = np.concatenate((predictions_reshaped,X_val.reshape(X_val.shape[0],X_val.shape[2])), axis=1)
    print(predictions_reshaped.shape)
    print(X_test.shape)
    predictions_reshaped = np.concatenate((predictions_reshaped,X_test.reshape(X_test.shape[0],X_test.shape[2])), axis=1)
    #print(predictions_reshaped.shape)
    predictions_reshaped = scaler.inverse_transform(predictions_reshaped)[:, 0:num_taxa]
    y_test = scaler.inverse_transform(np.concatenate([y_test, X_test.reshape(X_test.shape[0],X_test.shape[2])], axis=1))[:, 0:num_taxa]

    val_predictions = model.predict(X_val)

    val_pred_reshaped = val_predictions
    array_val = []
    for i in range(val_predictions.shape[1]):
        liste = val_predictions[:,i].reshape(-1,1)
        array_val.append(liste)
    val_pred_reshaped = np.concatenate((array_val),axis = 1)
    val_pred_reshaped = np.concatenate((val_pred_reshaped,X_val.reshape(X_val.shape[0], X_val.shape[2])), axis = 1)
    val_pred_reshaped = scaler.inverse_transform(val_pred_reshaped)[:, 0:num_taxa]


    train_pred_reshaped = predictions_train
    array_train = []
    for i in range(predictions_train.shape[1]):
        liste = predictions_train[:, i].reshape(-1, 1)
        array_train.append(liste)
    train_pred_reshaped = np.concatenate((array_train),axis=1)
    #train_pred_reshaped = np.concatenate((train_pred_reshaped,X_val.reshape(X_val.shape[0],X_val.shape[2])), axis=1)
    train_pred_reshaped = np.concatenate((train_pred_reshaped,X_train.reshape(X_train.shape[0],X_train.shape[2])), axis=1)
    train_pred_reshaped = scaler.inverse_transform(train_pred_reshaped)[:, 0:num_taxa]

    mae = mean_absolute_error(y_test, predictions_reshaped)
    rmse = math.sqrt(mean_squared_error(y_test, predictions_reshaped))
    nrmse = rmse/np.std(predictions_reshaped)
    mae_train = mean_absolute_error(y_train, train_pred_reshaped)
    rmse_train = math.sqrt(mean_squared_error(y_train, train_pred_reshaped))
    nrmse_train = rmse_train/np.std(train_pred_reshaped)

    error = errorInterval.predict_with_pi(predictions_reshaped, X_train, num_taxa, nrmse)

    print(f'Mean Absolute Error: {mae_train}')
    print(f'Root Mean Squared Error: {rmse_train}')
    print(f'Normalized Root Mean Squared Error: {nrmse_train}')

    print(f'Mean Absolute Error: {mae}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Normalized Root Mean Squared Error: {nrmse}')

    y_train = scaler.inverse_transform(np.concatenate([y_train, X_train.reshape(X_train.shape[0],X_train.shape[2])], axis=1))[:, 0:num_taxa]
    print("Print results")
    for i in range(num_taxa):
        # Visualize predictions for Targets
        plt.plot(new_timeseries['Time'], new_timeseries[f'Target{i+1}'], label='True Values')
        # plt.plot(complete.iloc[val_size:]['Time'], predictions_reshaped[:, i], label='Predictions test')
        plt.plot(new_timeseries.iloc[:train_size]['Time'], train_pred_reshaped[:, i], label='Training')
        plt.plot(new_timeseries.iloc[train_size:val_size]['Time'], val_pred_reshaped[:,i], label = 'Validation')
        plt.plot(new_timeseries.iloc[val_size:]['Time'],error[i][0][1:len(error[i][0])], label = "upper")
        plt.plot(new_timeseries.iloc[val_size:]['Time'],error[i][1][1:len(error[i][1])], label = "lower")
        #plt.plot(complete.iloc[val_size:]['Time'],prediction_inter[i][2][1:len(prediction_inter[i][2])], label = "mean")
        plt.plot(new_timeseries.iloc[val_size:]['Time'], predictions_reshaped[:,i], label = 'Test')
        plt.fill_between(new_timeseries.iloc[val_size:]['Time'],error[i][0][1:len(error[i][0])], error[i][1][1:len(error[i][1])], alpha=0.2)
        # plt.plot(complete['Time'], complete['temp'], label='Predictions train')
        plt.title('Multivariate Time Series '+num_taxa[f"Target{i+1}"], fontsize = 6)
        plt.xlabel('Time')
        plt.ylabel(f'Target{i+1}')
        plt.legend()
        plt.savefig(f"Milwaukee/LSTM_new/plot{i+1}_retrainDinslaken.png")
        plt.cla()
    
#retrain_model("/local/work/16S/snakemake_qiime/16S/MachineLearning/Wastewater/Dinslaken/timeseries.tsv", "/local/work/16S/snakemake_qiime/16S/MachineLearning/Wastewater/Dinslaken/index_only.tsv", "/local/work/16S/snakemake_qiime/16S/MachineLearning/Wastewater/Dinslaken/metadata.tsv",model,scaler)