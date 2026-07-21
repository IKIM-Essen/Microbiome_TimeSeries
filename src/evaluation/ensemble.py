import numpy as np
from pickle import dump, load
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping

from src.model_building.create_models import (
    fit_model,
    ensemble_predict,
    fit_model_retraining,
)
from src.utils.config import reshape, load_config
from src.preprocessing.scaling import inverse_scale_data


def _prepare_prediction_array(predictions):
    """Convert windowed model outputs to a 2D target array for evaluation."""
    predictions = np.asarray(predictions)
    if predictions.ndim == 3:
        return predictions[:, -1, :]
    if predictions.ndim == 2:
        return predictions
    raise ValueError(f"Unsupported prediction shape for interval evaluation: {predictions.shape}")

CONFIG_PATH = "config/profile.yaml"

config = load_config(CONFIG_PATH)

mode = config["model_architecture"]


def predict_interval(
    number_models,
    Xtrain,
    Ytrain,
    Xval,
    Yval,
    Xtest,
    Ytest,
    scaler_path,
    species,
    tcn_path,
    lstm_path,
    retraining=False,
):
    # make predictions

    ensemble = []
    ensemble_2 = []
    yhat_list = []
    for i in range(number_models):
        # define and fit the model on the training set
        if mode == "tcn_lstm":
            if retraining == False:
                tcn_model, lstm_model = fit_model(
                    Xtrain,
                    Ytrain,
                    Xval,
                    Yval,
                    species,
                    tcn_path,
                    str(mode),
                    save_model=False,
                )

            elif retraining == True:
                tcn_model, lstm_model = fit_model_retraining(
                    Xtrain, Ytrain, Xval, Yval, species, tcn_path, lstm_path
                )
            # Make predictions on the test set
            predictions_val = ensemble_predict(tcn_model, lstm_model, Xval)
        elif mode == "lstm":
            lstm = fit_model(
                Xtrain,
                Ytrain,
                Xval,
                Yval,
                species,
                lstm_path,
                str(mode),
                save_model=False,
            )
            predictions_val = lstm.predict(Xval)

        # elif mode == "attention":

        y_val_tcn = _prepare_prediction_array(predictions_val)

        y_val_tcn = inverse_scale_data(y_val_tcn, scaler_path)

        yhat = y_val_tcn[: len(species)]
        ensemble.append(yhat)
        yhat_species = []
        y = 0
        while y < len(yhat[1]):
            lst2 = [item[y] for item in yhat]
            yhat_species.append(lst2)
            y += 1
        yhat_list.append(yhat_species)

    stacked = np.stack(ensemble, axis=0)
    # Reshape to (3*14, 1544)
    reshaped = stacked.reshape(-1, Ytrain.shape[1])  # shape: (42, 1544)

    # Variance over timepoints and repetitions
    feature_variance = np.var(reshaped, axis=0, ddof=1)  # shape: (1544,)

    mean_prediction = np.mean(stacked, axis=0)

    residuals = mean_prediction - Yval
    # -- In case of standard deviation use ---
    mse = np.mean(residuals**2, axis=0)
    var_residual = mse

    # -- In case of non-distributional use --
    # For asymmetric intervals compute quantiles of residuals:
    alpha = 0.05
    q_low = np.quantile(residuals, alpha / 2, axis=0)  # e.g. 0.025 quantile
    q_high = np.quantile(residuals, 1 - alpha / 2, axis=0)

    # Widths relative to mean
    lower_width = np.abs(q_low)
    upper_width = np.abs(q_high)

    """ Test for normality in case of normality assumption
    # 1. Shapiro-Wilk test
    feature_results = {}

    for feature_idx in range(residuals.shape[1]):
        vals = residuals[:, feature_idx]  # 14 values for this feature
        stat, p = shapiro(vals)
        
        feature_results[feature_idx] = {
            "stat": stat,
            "p_value": p,
            "normal": p > 0.05  # True if we fail to reject normality
        }

    # Summarize results
    num_normal = sum(v["normal"] for v in feature_results.values())
    num_non_normal = len(feature_results) - num_normal

    print(f"Out of {residuals.shape[1]} features:")
    print(f" - {num_normal} look Gaussian (p > 0.05)")
    print(f" - {num_non_normal} do NOT look Gaussian (p <= 0.05)")
    
    z = norm.ppf(0.975)

    total_variance = feature_variance + var_residual
    std_total = np.sqrt(total_variance)
    """
    for i in range(number_models):
        # define and fit the model on the training set
        if mode == "tcn_lstm":
            predictions_test = ensemble_predict(tcn_model, lstm_model, Xtest)

        elif mode == "lstm":
            predictions_test = lstm.predict(Xtest)

        y_test_tcn = _prepare_prediction_array(predictions_test)

        predictions_reshaped = inverse_scale_data(y_test_tcn,scaler_path)

        yhat = predictions_reshaped[: len(species)]
        ensemble_2.append(yhat)

    stacked_2 = np.stack(ensemble_2, axis=0)
    mean_prediction_2 = np.mean(stacked_2, axis=0)

    mean_prediction_2 = np.swapaxes(mean_prediction_2, 0, 1)

    list_yhat = []

    i = 0
    while i < len(species):
        list_error = []
        list_lower = [np.nan] * Xtrain.shape[1]
        list_upper = [np.nan] * Xtrain.shape[1]
        list_mean = [np.nan] * Xtrain.shape[1]
        y = 0
        while y < mean_prediction_2.shape[1]:
            lower = mean_prediction_2[i][y] - lower_width[i]
            upper = mean_prediction_2[i][y] + upper_width[i]
            mean = mean_prediction_2[i][y]
            upper, lower, mean = [max(0, x) for x in (upper, lower, mean)]
            list_lower.append(lower)
            list_upper.append(upper)
            list_mean.append(mean)
            y += 1
        # Slice from there

        list_error.append(list_upper)

        # Slice from there

        list_error.append(list_lower)

        # Slice from there
        list_error.append(list_mean)
        list_yhat.append(list_error)

        i += 1

    return list_yhat


def TCNBlock(input_layer, filters=128, kernel_size=3, num_layers=4, dropout=0.2):
    x = input_layer
    for i in range(num_layers):
        dilation = 2 ** i
        x = layers.Conv1D(
            filters,
            kernel_size,
            dilation_rate=dilation,
            padding="causal",
            activation="relu"
        )(x)
        x = layers.Dropout(dropout)(x)
    return x


def build_big_architecture_model(bact_shape, meta_shape, output_dim, horizon=1):
    """Build the same dual-branch TCN+LSTM+attention model used in big_architecture.py."""
    bact_input = Input(shape=bact_shape, name="bacterial_input")
    meta_input = Input(shape=meta_shape, name="metadata_input")

    tcn = TCNBlock(
        bact_input,
        filters=128,
        kernel_size=3,
        num_layers=4,
        dropout=0.2
    )
    tcn = layers.GlobalAveragePooling1D()(tcn)

    lstm_seq = layers.LSTM(128, return_sequences=True, name="bact_lstm")(bact_input)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, name="temporal_attention")(lstm_seq, lstm_seq)
    attn = layers.GlobalAveragePooling1D()(attn)

    bact_embed = layers.Concatenate(name="bact_concat")([tcn, attn])

    meta_seq = layers.LSTM(64, return_sequences=True, name="meta_lstm")(meta_input)
    meta_embed = layers.GlobalAveragePooling1D()(meta_seq)

    gate_input = layers.Concatenate()([bact_embed, meta_embed])
    gate = layers.Dense(2, activation="softmax", name="branch_gate")(gate_input)
    gate_bact = layers.Lambda(lambda x: x[:, 0:1])(gate)
    gate_meta = layers.Lambda(lambda x: x[:, 1:2])(gate)

    bact_weighted = layers.Multiply()([bact_embed, gate_bact])
    meta_weighted = layers.Multiply()([meta_embed, gate_meta])
    combined = layers.Concatenate(name="final_concat")([bact_weighted, meta_weighted])

    x = layers.Dense(256, activation="relu")(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)

    out = layers.Dense(output_dim * horizon)(x)
    out = layers.Reshape((horizon, output_dim))(out)

    model = models.Model(
        inputs=[bact_input, meta_input],
        outputs=out,
        name="Microbiome_TCN_LSTM_Attention_Gated"
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def fit_member_model(
    X_bact_train,
    X_meta_train,
    y_train,
    X_bact_val,
    X_meta_val,
    y_val,
    output_dim,
    horizon=1,
    batch_size=32,
    epochs=100,
    patience=5,
):
    model = build_big_architecture_model(
        bact_shape=(X_bact_train.shape[1], X_bact_train.shape[2]),
        meta_shape=(X_meta_train.shape[1], X_meta_train.shape[2]),
        output_dim=output_dim,
        horizon=horizon,
    )

    es = EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True, verbose=1)
    model.fit(
        [X_bact_train, X_meta_train],
        y_train,
        validation_data=([X_bact_val, X_meta_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )
    return model


def fit_ensemble(
    n_members,
    X_bact_train,
    X_meta_train,
    y_train,
    X_bact_val,
    X_meta_val,
    y_val,
    output_dim,
    horizon=1,
    batch_size=32,
    epochs=100,
    patience=5,
):
    ensemble = []
    for i in range(n_members):
        model = fit_member_model(
            X_bact_train,
            X_meta_train,
            y_train,
            X_bact_val,
            X_meta_val,
            y_val,
            output_dim,
            horizon=horizon,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
        )
        ensemble.append(model)
    return ensemble


def _reshape_model_predictions(predictions):
    if predictions.ndim != 3:
        raise ValueError("Expected model predictions with shape (samples, horizon, features)")
    if predictions.shape[1] != 1:
        raise ValueError("This helper currently supports horizon=1 only")
    return predictions.reshape(predictions.shape[0], predictions.shape[2])


def inverse_transform_predictions(predictions, X_bact, scaler, num_taxa):
    predictions = _reshape_model_predictions(predictions)
    stacked = np.concatenate(
        [predictions, X_bact.reshape(X_bact.shape[0], X_bact.shape[2])],
        axis=1,
    )
    inverted = scaler.inverse_transform(stacked)
    return inverted[:, 0:num_taxa]


def predict_with_pi(ensemble, X_bact_test, X_meta_test, scaler, num_taxa, coverage=0.95):
    """Return upper/lower/mean intervals for each target using ensemble spread."""
    all_preds = []
    for model in ensemble:
        preds = model.predict([X_bact_test, X_meta_test], verbose=0)
        inv_preds = inverse_transform_predictions(preds, X_bact_test, scaler, num_taxa)
        all_preds.append(inv_preds)

    all_preds = np.stack(all_preds, axis=0)
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0, ddof=1)

    z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.575}
    z = z_values.get(coverage, 1.96)

    upper = mean_pred + z * std_pred
    lower = mean_pred - z * std_pred
    lower = np.maximum(lower, 0.0)

    intervals = []
    for j in range(num_taxa):
        intervals.append([upper[:, j], lower[:, j], mean_pred[:, j]])

    return intervals


def prediction_interval_from_train(
    y_train_true,
    y_train_pred,
    y_test_pred,
    coverage=0.95,
):
    """Compute constant-width intervals from training residuals."""
    if y_train_true.shape != y_train_pred.shape:
        raise ValueError("y_train_true and y_train_pred must have same shape")

    z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.575}
    z = z_values.get(coverage, 1.96)

    residuals = y_train_true - y_train_pred
    sigma = np.std(residuals, axis=0, ddof=1)

    intervals = []
    for j in range(y_test_pred.shape[1]):
        mean_pred = y_test_pred[:, j]
        upper = mean_pred + z * sigma[j]
        lower = mean_pred - z * sigma[j]
        lower = np.maximum(lower, 0.0)
        intervals.append([upper, lower, mean_pred])
    return intervals
