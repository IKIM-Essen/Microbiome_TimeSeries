from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    get_scorer_names,
)

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import csv


def mae(actual, predicted):
    """
    Calculate the Mean Absolute Error between the actual and the predicted values
    """
    return mean_absolute_error(actual, predicted)


def mse(actual, predicted):
    """
    Calculate the Mean Squared Error between the actual and the predicted values
    """
    return mean_squared_error(actual, predicted)


def rmse(actual, predicted):
    """
    Calculate the Root Mean Squared Error between the actual and the predicted values
    """
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    return rmse


def r2(actual, predicted):
    """
    Calculate the measure of certainty for the predictions
    """
    return r2_score(actual, predicted)


def stdev(predicted):
    """
    Calculate the standarddeviation for the predicted values
    """
    s = np.std(predicted)
    return s


def nrmse(stdev, rmse):
    """
    Calculate the Normalized Root Mean Squared Error for the predictions
    The error is normalized by dividing it with the standarddeviation
    """
    nrmse = rmse / stdev
    return nrmse


def eval_dict(
    actualtrain, predictedtrain, actualval, predictedval, actualtest, predictedtest
):
    """
    Creating a dictionary holding all interesting evaluation metrics results
    """
    eval_dict = {
        "mae_train": mae(actualtrain, predictedtrain),
        "mae_val": mae(actualval, predictedval),
        "mae_test": mae(actualtest, predictedtest),
        "mse_train": mse(actualtrain, predictedtrain),
        "mse_val": mse(actualval, predictedval),
        "mse_test": mse(actualtest, predictedtest),
        "r2": r2(actualtest, predictedtest),
        "rmse_train": rmse(actualtrain, predictedtrain),
        "rmse_test": rmse(actualtest, predictedtest),
        "nrmse_test": nrmse(stdev(predictedtest), rmse(actualtest, predictedtest)),
    }
    return eval_dict


def outlier(predictionInterval, actualValues, bacteria):
    """
    Comparing the actual measured values with the upper and lower border of the prediction interval.
    If the value for one of the bacterial genera lies outside of this interval, it is added to a list of outliers.
    """
    i = 0
    outliers = {}
    actualValues1 = np.transpose(actualValues)
    while i < len(actualValues1):
        upper = predictionInterval[i][0]
        lower = predictionInterval[i][1]
        y = 0
        while y < len(predictionInterval[i][0]):
            if actualValues1[i][y] < lower[y]:
                outliers[bacteria[i], y] = actualValues1[i][y]
            elif actualValues1[i][y] > upper[y]:
                outliers[bacteria[i], y] = actualValues1[i][y]
            y = y + 1
        i = i + 1
    with open(plotpath + "outliers_testset.csv", "w") as f:
        w = csv.DictWriter(f, outliers.keys())
        w.writeheader()
        w.writerow(outliers)
    return outliers
