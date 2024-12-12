import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, get_scorer_names
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Embedding, GRU
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import math
import ensemble
import errorInterval

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
    exogenous = pd.read_csv(path, header = None, sep = "\t")
    exogenous.drop(0, axis = 1, inplace = True)
    #print(exogenous)
    return exogenous

def get_timeframe(df):
    timelist = df.columns.values.tolist()
    return timelist

def create_complete_df(path_prob,path_tax,path_exo):
    #index_list,tax = merge_dataframes(path_tax, path_prob)
    index_list,tax = read_reduce_dataframe(path_prob)
    #exo = read_exogenous(path_exo)
    #exo_trans = exo.T
    #print(exo_trans)
    time = get_timeframe(tax)
    complete = pd.DataFrame({"Time": time})
    number_taxa = len(tax.values)
    dic_TargTax = {}
    for i in range(len(tax.values)):
        complete[f"Target{i+1}"] = tax.iloc[i].values
        #print(tax.index[i])
        dic_TargTax[f"Target{i+1}"] = tax.index[i]
        #print(i)
    return complete, number_taxa, dic_TargTax

complete, num_taxa, dic_taxa = create_complete_df("/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/female_bacteroides.txt","/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/diet/bacteria_list_wDiet.tsv","Milwaukee/metadata_JonesIsland.tsv")
complete.sort_values(by=["Time"], inplace = True, ignore_index = True)
complete.fillna(0, axis = 1, inplace = True)
# Feature engineering: Creating lag features
lag_targets =0
for y in range(num_taxa):
    for i in range(1, 4):
        complete[f'Target{y+1}_Lag_{i}'] = complete[f'Target{y+1}'].shift(i)
        lag_targets += 1
        #complete[f'Exogenous_Lag_{i}'] = complete['Exo'].shift(i)
#print(lag_targets)
# Drop rows with NaN values resulting from lag features
complete = complete.dropna()
#print(complete)
print("Scaling data")
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
complete_woT = complete.drop(['Time'], axis=1)
#print(complete_woT.shape)
scaled_data = scaler.fit_transform(complete_woT)


# Prepare data for LSTM
X = np.array(scaled_data[:,num_taxa:(num_taxa+lag_targets)])
#print(X.shape)
X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape for LSTM input
y = scaled_data[:, 0:num_taxa]
#print(y.shape)

# Split the data into training and testing sets
train_size = int(len(X) * 0.7)
val_size = int(len(X)*0.8)
X_train, X_val, X_test = X[0:train_size], X[train_size:val_size], X[val_size:]
y_train, y_val, y_test = y[0:train_size], y[train_size:val_size], y[val_size:]

#print(X_train.shape)
#print(y_train.shape)
print("Creating model")
model = Sequential()
model.add(GRU(units=4, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(num_taxa, name='target'))

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=400, batch_size=5, verbose=0, callbacks = [es])
#model.save("1LayerLSTM.h5")
#print(X_test)
# Make predictions on the test set
#model.save("1LayerGRU.h5")
# Make predictions on the test set
predictions = model.predict(X_test)
predictions_train = model.predict(X_train)

# Inverse transform the predictions and actual values to the original scale
predictions_reshaped = predictions

#ensemble_list = ensemble.fit_ensemble(5,X_train,X_val,X_test,y_train,y_val,scaler,num_taxa)
#print("Ensemble generated!")
#prediction_inter = ensemble.predict_with_pi(ensemble_list, X_train, X_test, y_test, scaler, num_taxa)
#print("Errors calculated!")

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



#print(predictions_reshaped.shape)
#print(y_test.shape)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions_reshaped)
rmse = math.sqrt(mean_squared_error(y_test, predictions_reshaped))
nrmse = rmse/np.std(predictions_reshaped)
mae_train = mean_absolute_error(y_train, train_pred_reshaped)
rmse_train = math.sqrt(mean_squared_error(y_train, train_pred_reshaped))
nrmse_train = rmse_train/np.std(train_pred_reshaped)

print(f'Mean Absolute Error: {mae_train}')
print(f'Root Mean Squared Error: {rmse_train}')
print(f'Normalized Root Mean Squared Error: {nrmse_train}')

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Normalized Root Mean Squared Error: {nrmse}')

error = errorInterval.predict_with_pi(predictions_reshaped, X_train, num_taxa, nrmse)


print("Plotting results")
#print(prediction_inter[1])
#print(complete['Target1'])
for i in range(num_taxa):
    # Visualize predictions for Targets
    plt.plot(complete['Time'], complete[f'Target{i+1}'], label='True Values')
    # plt.plot(complete.iloc[val_size:]['Time'], predictions_reshaped[:, i], label='Predictions test')
    plt.plot(complete.iloc[:train_size]['Time'], train_pred_reshaped[:, i], label='Training')
    plt.plot(complete.iloc[train_size:val_size]['Time'], val_pred_reshaped[:,i], label = 'Validation')
    plt.plot(complete.iloc[val_size:]['Time'],error[i][0][1:len(error[i][0])], label = "upper")
    plt.plot(complete.iloc[val_size:]['Time'],error[i][1][1:len(error[i][1])], label = "lower")
    #plt.plot(complete.iloc[val_size:]['Time'],prediction_inter[i][2][1:len(prediction_inter[i][2])], label = "mean")
    plt.plot(complete.iloc[val_size:]['Time'], predictions_reshaped[:,i], label = 'Test')
    plt.fill_between(complete.iloc[val_size:]['Time'],error[i][0][1:len(error[i][0])], error[i][1][1:len(error[i][1])], alpha=0.2)
    # plt.plot(complete['Time'], complete['temp'], label='Predictions train')
    plt.title('Multivariate Time Series '+dic_taxa[f"Target{i+1}"], fontsize = 6)
    plt.xlabel('Time')
    plt.ylabel(f'Target{i+1}')
    plt.legend()
    plt.show()
    plt.savefig(f"small/GRU/plot{i+1}_female.png")
    plt.cla()


"""
print("Plotting results")
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
    plt.fill_between(complete.iloc[val_size:]['Time'],prediction_inter[i][0][1:len(prediction_inter[i][0])], prediction_inter[i][1][1:len(prediction_inter[i][1])], alpha=0.2)
    # plt.plot(complete['Time'], complete['temp'], label='Predictions train')
    plt.title('Multivariate Time Series '+dic_taxa[f"Target{i+1}"], fontsize = 6)
    plt.xlabel('Time')
    plt.ylabel(f'Target{i+1}')
    plt.legend()
    plt.show()
    plt.savefig(f"diet/GRU/plot{i+1}_female.png")
    plt.cla()


def retrain_model(timeseries, taxalist, exo, model, scaler):
    print("Read new model data")
    new_timeseries, num_taxa, new_taxa = create_complete_df(timeseries, taxalist, exo)
    new_timeseries.sort_values(by=["Time"], inplace = True, ignore_index = True)
    lag_targets =0
    for y in range(num_taxa):
        for i in range(1, 4):
            new_timeseries[f'Target{y+1}_Lag_{i}'] = new_timeseries[f'Target{y+1}'].shift(i)
            lag_targets += 1
    new_timeseries = new_timeseries.dropna()
    print("Scale new data")
    new_timeseries_woT = complete.drop(['Time'], axis=1)
    scaled_data = scaler.fit_transform(new_timeseries_woT)
    # Prepare data for LSTM
    X = np.array(scaled_data[:,num_taxa:(num_taxa+lag_targets)])
    #print(X.shape)
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape for LSTM input
    y = scaled_data[:, 0:num_taxa]
    train_size = int(len(X) * 0.7)
    val_size = int(len(X)*0.8)
    X_train, X_val, X_test = X[0:train_size], X[train_size:val_size], X[val_size:]
    y_train, y_val, y_test = y[0:train_size], y[train_size:val_size], y[val_size:]
    print("Fit Model")
    #model = keras.models.load_model("/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/multitraining/DonorB/1LayerLSTM_retrained.h5")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    history = model.fit(trainX, trainY, validation_data=(valX,valY), epochs=400, verbose=0, batch_size=5, callbacks = [es])
    print(history.history.keys())
    ml_plots.plot_loss(history)

    model.save(plotpath+"1LayerGRU_retrained.h5")
    print("Create ensemble")
    ensemble = model_confidence.retrain_confidence_model("/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/multitraining/DonorB/ensemblemodels", trainX, trainY, valX, valY, plotpath)
    print("Create Interval")
    y_hat, error_hat = model_confidence.predict_with_pi(ensemble, X, scaler.inverse_transform(Y), scaler, species, n_steps) 

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
    print("Print results")
    for i in range(num_taxa):
        # Visualize predictions for Targets
        plt.plot(complete['Time'], complete[f'Target{i+1}'], label='True Values')
        # plt.plot(complete.iloc[val_size:]['Time'], predictions_reshaped[:, i], label='Predictions test')
        plt.plot(complete.iloc[:train_size]['Time'], train_pred_reshaped[:, i], label='Training')
        plt.plot(complete.iloc[train_size:val_size]['Time'], val_pred_reshaped[:,i], label = 'Validation')
        plt.plot(complete.iloc[val_size:]['Time'],prediction_inter[i][0][1:len(prediction_inter[i][0])], label = "upper")
        plt.plot(complete.iloc[val_size:]['Time'],prediction_inter[i][1][1:len(prediction_inter[i][1])], label = "lower")
        plt.plot(complete.iloc[val_size:]['Time'],prediction_inter[i][2][1:len(prediction_inter[i][2])], label = "mean")
        plt.fill_between(complete.iloc[val_size:]['Time'],prediction_inter[i][0][1:len(prediction_inter[i][0])], prediction_inter[i][1][1:len(prediction_inter[i][1])], alpha=0.2)
        # plt.plot(complete['Time'], complete['temp'], label='Predictions train')
        plt.title('Multivariate Time Series '+dic_taxa[f"Target{i+1}"], fontsize = 6)
        plt.xlabel('Time')
        plt.ylabel(f'Target{i+1}')
        plt.legend()
        plt.show()
        plt.savefig(f"diet/GRU/plot{i+1}_retrained.png")
        plt.cla()
"""