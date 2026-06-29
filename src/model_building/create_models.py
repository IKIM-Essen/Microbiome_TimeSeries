import logging
import os

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Embedding
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import keras
import tensorflow as tf
from tensorflow.keras import layers, models

from src.utils.config import load_model_if_path, load_config

# Setup logging for model-building and training operations
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
LOG_DIR = os.path.join(ROOT_DIR, "logs", "training")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "model_building.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)


# --- Build TCN model ---
# This model uses causal 1D convolution to predict the next step for each target.
def build_tcn(input_shape, output_dim, horizon=1):
    inp = layers.Input(shape=input_shape)
    x = inp
    for i in range(4):  # 4 layers of dilated causal convs
        x = layers.Conv1D(
            filters=16,
            kernel_size=3,
            dilation_rate=4**i,
            padding="causal",
            activation="relu",
        )(x)
        x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(output_dim * horizon)(x)
    out = layers.Reshape((horizon, output_dim))(out)
    return models.Model(inputs=inp, outputs=out)


# Build a standard regression LSTM model for the same output shape
def build_lstm(input_shape, output_dim, horizon=1):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(1024, return_sequences=False)(inp)
    out = layers.Dense(output_dim * horizon)(x)
    out = layers.Reshape((horizon, output_dim))(out)
    return models.Model(inputs=inp, outputs=out)


def ensemble_predict(tcn, lstm, X):
    y_tcn = tcn.predict(X)
    y_lstm_residual = lstm.predict(X)
    return y_tcn + y_lstm_residual


# Build a standard regression LSTM model as standalone model
def build_standalone_lstm(input_shape, output_dim, horizon=1):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(2048, activation='relu', return_sequences=False)(inp)
    out = layers.Dense(output_dim * horizon)(x)
    out = layers.Reshape((horizon, output_dim))(out)
    return models.Model(inputs=inp, outputs=out)


# Fit the TCN model first, then train the LSTM on the TCN residuals.
def fit_model(X_train, y_train, X_val, y_val, n_features, model_path, model_architecture=None):
    # model_architecture can be passed directly; if not provided, read from config
    if model_architecture is None:
        CONFIG_PATH = "config/profile.yaml"
        config = load_config(CONFIG_PATH)
        model_type = config.get("model_architecture")
    else:
        model_type = model_architecture
    # Ensure model path exists
    os.makedirs(model_path, exist_ok=True)
    print("Fitting!")
    try:
        logger.info(
            "Starting model fitting with X_train shape %s, y_train shape %s",
            X_train.shape,
            y_train.shape,
        )

        # Example setup
        time_steps = 1  # input window length
        num_features = X_train.shape[2]  # number of input features
        num_targets = y_train.shape[1]  # number of target variables
        horizon = 1  # prediction horizon
        logger.info(
            "Model configuration: time_steps=%s, num_features=%s, num_targets=%s, horizon=%s",
            time_steps,
            num_features,
            num_targets,
            horizon,
        )
        print("yes")
        if model_type == "tcn_lstm":
            print("good")
        # Build models and compile them for regression
            tcn_model = build_tcn((time_steps, num_features), num_targets, horizon)
            tcn_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            lstm_model = build_lstm((time_steps, num_features), num_targets, horizon)
            lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            logger.info("Models built and compiled successfully")

            # --- Train TCN first ---
            logger.info("Starting TCN model training")
            tcn_model.fit(
                X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val)
            )
            logger.info("TCN model training completed")

            # --- Compute residuals ---
            # Residuals are the difference between the true target and the TCN prediction.
            logger.info("Computing residuals for LSTM training")
            y_tcn_pred = tcn_model.predict(X_train)
            residuals = y_train - y_tcn_pred

            y_pred_tcn_val = tcn_model.predict(X_val)
            residuals_val = y_val - y_pred_tcn_val
            logger.info("Residuals computed successfully")

            # --- Train LSTM on residuals ---
            logger.info("Starting LSTM model training on residuals")
            es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=10)
            lstm_model.fit(
                X_train,
                residuals,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, residuals_val),
                callbacks=[es],
            )
            logger.info("LSTM model training completed")

            tcn_path = os.path.join(model_path, "tcn_model.h5")
            lstm_path = os.path.join(model_path, "lstm_model.h5")
            logger.info("Saving TCN model to %s", tcn_path)
            tcn_model.save(tcn_path)
            logger.info("Saving LSTM model to %s", lstm_path)
            lstm_model.save(lstm_path)
            logger.info("Model fitting completed successfully")

            return tcn_model, lstm_model
        elif model_type == "lstm":
            print("better")
            lstm = build_standalone_lstm((time_steps, num_features), num_targets, horizon)
            lstm.compile(optimizer="adam", loss="mse", metrics=["mae"])
            logger.info("Models built and compiled successfully")
            es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
            lstm.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=4, batch_size=5, verbose=0, callbacks = [es])
            logger.info("LSTM model training completed")
            lstm_path = os.path.join(model_path, "lstm_model.h5")
            logger.info("Saving LSTM model to %s", lstm_path)
            lstm.save(lstm_path)
            logger.info("Model fitting completed successfully")

            return lstm
        
        # elif model_type = "attention":       

    except Exception as e:
        logger.error("Error during model fitting: %s", str(e), exc_info=True)
        raise


def fit_model_retraining(X_train, y_train, X_val, y_val, tcn, lstm):

    tcn = load_model_if_path(tcn)
    lstm = load_model_if_path(lstm)

    tcn.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    # --- Compute residuals ---
    y_pred_tcn_train = tcn.predict(X_train)
    residuals = y_train - y_pred_tcn_train

    y_pred_tcn_val = tcn.predict(X_val)
    residuals_val = y_val - y_pred_tcn_val

    lstm.fit(
        X_train,
        residuals,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, residuals_val),
    )
    return tcn, lstm
