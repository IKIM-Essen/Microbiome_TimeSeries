import numpy as np
from pickle import dump, load

from src.model_building.create_models import (
    fit_model,
    ensemble_predict,
    fit_model_retraining,
)
from src.utils.config import reshape, load_config
from src.preprocessing.scaling import inverse_scale_data

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

        y_val_tcn = reshape(predictions_val)

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
    reshaped = stacked.reshape(-1, Xtrain.shape[2])  # shape: (42, 1544)

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

        y_test_tcn = reshape(predictions_test)

        predictions_reshaped = inverse_scale_data(y_test_tcn)

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
