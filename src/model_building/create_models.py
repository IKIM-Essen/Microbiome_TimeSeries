from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Embedding
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import keras
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Build TCN model ---
def build_tcn(input_shape, output_dim, horizon=1):
    inp = layers.Input(shape=input_shape)
    x = inp
    for i in range(4):  # 4 layers of dilated causal convs
        x = layers.Conv1D(filters=16,
                          kernel_size=3,
                          dilation_rate=4**i,
                          padding="causal",
                          activation="relu")(x)
        x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(output_dim * horizon)(x)
    out = layers.Reshape((horizon, output_dim))(out)
    return models.Model(inputs=inp, outputs=out)

def build_lstm(input_shape, output_dim, horizon=1):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(1024, return_sequences=False)(inp)
    out = layers.Dense(output_dim * horizon)(x)
    out = layers.Reshape((horizon, output_dim))(out)
    return models.Model(inputs=inp, outputs=out)

def ensemble_predict(tcn,lstm,X):
    y_tcn = tcn.predict(X)
    y_lstm_residual = lstm.predict(X)
    return y_tcn + y_lstm_residual

def fit_model(X_train, y_train, X_val, y_val, n_features, tcn_path, lstm_path):

    # Example setup
    time_steps = 1                        # input window length
    num_features = X_train.shape[2]       # number of input features
    num_targets = X_train.shape[2]        # number of target variables
    horizon = 1                           # forecast 7 steps ahead

    tcn_model = build_tcn((time_steps, num_features), num_targets, horizon)
    tcn_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    lstm_model = build_lstm((time_steps, num_features), num_targets, horizon)
    lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # --- Train TCN first ---
    tcn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data = (X_val,y_val))
    # --- Compute residuals ---
    y_tcn_pred = tcn_model.predict(X_train)
    residuals = y_train - y_tcn_pred

    y_pred_tcn_val = tcn_model.predict(X_val)
    residuals_val = y_val - y_pred_tcn_val

    # --- Train LSTM on residuals ---
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
    lstm_model.fit(X_train, residuals, epochs=100, batch_size=32, validation_data = (X_val, residuals_val), callbacks = [es])

    tcn_model.save(tcn_path)
    lstm_model.save(lstm_path)

    return tcn_model,lstm_model