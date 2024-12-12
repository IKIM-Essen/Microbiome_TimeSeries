import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, get_scorer_names
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Embedding
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import math

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
    time = get_timeframe(tax)
    complete = pd.DataFrame({"Time": time})
    number_taxa = len(tax.values)
    for i in range(len(tax.values)):
        complete[f"Target{i+1}"] = tax.iloc[i].values
        #print(i)
    #complete["Exo"] = exo.iloc[0]
    return complete,number_taxa

complete, num_taxa = create_complete_df("/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/female_bacteroides.txt","/local/work/16S/snakemake_qiime/16S/MachineLearning/HostMicrobiome/bacteria_list_final4.tsv","/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/female_shannon.txt")

lags = 3

# Feature engineering: Creating lag features
lag_targets =0
for y in range(num_taxa):
    for i in range(1, lags+1):
        complete[f'Target{y+1}_Lag_{i}'] = complete[f'Target{y+1}'].shift(i)
        lag_targets += 1
        #complete[f'Exogenous_Lag_{i}'] = complete['Exo'].shift(i)
#print(lag_targets)
# Drop rows with NaN values resulting from lag features
complete = complete.dropna()
#print(complete)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
complete_woT = complete.drop(['Time'], axis=1)
print(complete_woT.shape)
scaled_data = scaler.fit_transform(complete_woT)


# Prepare data for LSTM
X = np.array(scaled_data[:,num_taxa:(num_taxa+lag_targets)])
#print(X.shape)
X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape for LSTM input
y = scaled_data[:, 0:num_taxa]
#print(y.shape)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]

#print(X_train.shape)
#print(y_train.shape)

# Build LSTM model with an exogenous variable for two targets
model = Sequential()
model.add(LSTM(2048, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))#, return_sequences = True
model.add(Dropout(0.2))
model.add(LSTM(2048, activation='relu', return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(2048, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_taxa, name='target'))  # Output layer for Target_1
#model.add(Dense(1, name='target_2'))  # Output layer for Target_2
model.compile(loss='mean_absolute_error', optimizer='adam')

# Train the model
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
model.fit(X_train, y_train, epochs=400, batch_size=1, verbose=2, callbacks = [es])

# Make predictions on the test set
predictions = model.predict(X_test)
predictions_train = model.predict(X_train)

# Inverse transform the predictions and actual values to the original scale
print(predictions[1])
predictions_reshaped = predictions
#print(predictions_reshaped.shape)
#print(predictions.shape)
array = []
for i in range(predictions.shape[1]):
    liste = predictions[:, i].reshape(-1, 1)
    array.append(liste)
predictions_reshaped = np.concatenate((array),axis=1)
#print(predictions_reshaped.shape)
#print(X_test.shape)
predictions_reshaped = np.concatenate((predictions_reshaped,X_test.reshape(X_test.shape[0],X_test.shape[2])), axis=1)
#print(predictions_reshaped)
predictions_reshaped = scaler.inverse_transform(predictions_reshaped)[:, 0:num_taxa]
y_test = scaler.inverse_transform(np.concatenate([y_test, X_test.reshape(X_test.shape[0],X_test.shape[2])], axis=1))[:, 0:num_taxa]

train_pred_reshaped = predictions_train
array_train = []
for i in range(predictions_train.shape[1]):
    liste = predictions_train[:, i].reshape(-1, 1)
    array_train.append(liste)
train_pred_reshaped = np.concatenate((array_train),axis=1)
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


print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Sqaured Error: {rmse}')
print(f'Normalized Root Mean Sqaured Error: {nrmse}')

print(f'Mean Absolute Error: {mae_train}')
print(f'Root Mean Sqaured Error: {rmse_train}')
print(f'Normalized Root Mean Sqaured Error: {nrmse_train}')
"""
# Visualize predictions for Target 1
plt.plot(complete['Time'], complete['Target1'], label='True Values (Target 1)')
plt.plot(complete.iloc[train_size:]['Time'], predictions_reshaped[:, 0], label='Predictions (Target 1)')
plt.title('Multivariate Time Series Prediction with LSTM and Exogenous Variable (Target 1)')
plt.xlabel('Time')
plt.ylabel('Target 1')
plt.legend()
plt.show()
plt.savefig("/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/plot1_LSTM.png")
plt.cla()

# Visualize predictions for Target 2
plt.plot(complete['Time'], complete['Target2'], label='True Values (Target 2)')
plt.plot(complete.iloc[train_size:]['Time'], predictions_reshaped[:, 1], label='Predictions (Target 2)')
plt.title('Multivariate Time Series Prediction with LSTM and Exogenous Variable (Target 2)')
plt.xlabel('Time')
plt.ylabel('Target 2')
plt.legend()
plt.show()
plt.savefig("/local/work/16S/snakemake_qiime/16S/MachineLearning/1_modelUpdate/plot2_LSTM.png")
plt.cla()
"""