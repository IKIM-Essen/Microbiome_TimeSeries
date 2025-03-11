from numpy import array
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import AnchoredText
from matplotlib import pyplot
from tensorflow.compat.v1.keras.backend import get_session

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import shap

tf.compat.v1.disable_v2_behavior()

plotpath = "../allGutFemale/"


def shap_featureimportance_plots(model, Xtrain, Xtest, species):
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = (
        shap.explainers._deep.deep_tf.passthrough
    )
    DE = shap.DeepExplainer(model, Xtrain)  # X_train is 3d numpy.ndarray
    shap_values = DE.shap_values(
        Xtrain, check_additivity=False
    )  # X_validate is 3d numpy.ndarray
    # print(shap_values[0][0].shape)
    # print(shap_values[0].shape)
    # print(type(species))
    species_list = species.tolist()
    shap_values_2D = shap_values[0].reshape(-1, len(species))
    X_train_2D = Xtrain.reshape(-1, len(species))
    # print(shap_values_2D.shape, X_train_2D.shape)

    plt.clf()
    plt.figure(figsize=(20, 40))
    shap.initjs()
    shap.summary_plot(
        shap_values_2D,
        X_train_2D,
        feature_names=species_list,
        plot_type="bar",
        show=False,
    )
    plt.savefig(plotpath + "shap.png", bbox_inches="tight")

    plt.clf()
    shap.summary_plot(
        shap_values_2D, X_train_2D, feature_names=species_list, show=False
    )
    plt.savefig(plotpath + "shap_dotplot.png", bbox_inches="tight")

    plt.clf()
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = (
        shap.explainers._deep.deep_tf.passthrough
    )
    Xflatten = model.predict(Xtrain).reshape(-1, len(species))
    # Xtestflatten = model.predict(Xtest).reshape(-1,38)
    Xtestflatten = Xtest.reshape(-1, len(species))
    # print(Xtestflatten.shape)
    # print(Xflatten.shape)
    DE = shap.DeepExplainer(model, Xtrain)  # X_train is 3d numpy.ndarray
    shap_values = DE.shap_values(
        Xtest, check_additivity=False
    )  # X_validate is 3d numpy.ndarray
    shap_values_2D = shap_values[0].reshape(-1, len(species))
    # print(testY.shape)
    # print(predictTest.shape)
    # print(shap_values_2D.shape)
    # print(shap_values[0])
    # print(shap_values[0].shape)
    # print(shap_values[0][0])
    # print(shap_values[0][0].shape)
    # print(shap_values[0][0][0])
    # df_shapvalue = pd.DataFrame(shap_values, columns=species.tolist())
    # print(df_shapvalue["d__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Bacteroidaceae; g__Bacteroides"].values)
    # shap.dependence_plot(0,shap_values[0][0],Xtest[0],species.tolist())
    shap.dependence_plot(0, shap_values_2D, Xtestflatten, species.tolist())
    plt.savefig(plotpath + "dependence.png", bbox_inches="tight")
    # print(DE.expected_value)
    # print(DE.expected_value.shape)

    plt.clf()
    shap.plots.force(
        DE.expected_value[0],
        shap_values[0][0][0],
        Xtest[0][0],
        species.tolist(),
        matplotlib=True,
        show=False,
    )
    plt.savefig(plotpath + "force.png", bbox_inches="tight")

    plt.clf()
    shap.plots._waterfall.waterfall_legacy(
        DE.expected_value[0], shap_values[0][0][0], feature_names=species.tolist()
    )
    plt.savefig(plotpath + "waterfall.png", bbox_inches="tight")
