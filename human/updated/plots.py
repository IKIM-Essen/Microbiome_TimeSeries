from numpy import array
from numpy import hstack
from numpy import asarray
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import AnchoredText
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    PolynomialFeatures,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import tensorflow as tf
import shap


def time_series_analysis_plot(species_series, dataframe_complete, filename, bacteria):
    """ """
    # Creating colors for the plot
    np.random.seed(100)
    mycolors = np.random.choice(
        list(mpl.colors.XKCD_COLORS.keys()), len(bacteria), replace=False
    )
    # Creating a plot displaying the OTU's found
    figure, ax = plt.subplots(figsize=(30, 10))
    figure.legend(loc=2, prop={"size": 6})
    plt.rcParams["figure.figsize"] = (20, 30)
    figure.legend(bbox_to_anchor=(1.1, 1.05))
    for i, y in enumerate(bacteria):
        if i > 0:
            ax.plot(
                dataframe_complete.index,
                dataframe_complete[y],
                color=mycolors[i],
                label=y,
            )
    d = {"down": 30, "up": -30}
    handles, labels = ax.get_legend_handles_labels()
    lgd = figure.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.9, 0, 0.07, 0.9)
    )
    # loc='upper center', bbox_to_anchor=(0.5,-0.1)
    plt.gca().set(ylabel="Number of species found", xlabel="Date")
    plt.yticks(fontsize=12, alpha=0.7)
    if len(dataframe_complete.index) < 50:
        plt.xticks(fontsize=10, rotation=45)
    if len(dataframe_complete.index) > 50:
        plt.xticks(fontsize=8, rotation=45)
    figure.canvas.mpl_connect("scroll_event", func)
    plt.title("Display of microbial species in patient dependent on time", fontsize=20)
    plt.show()
    figure.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches="tight")


def plot_loss(history):
    """ """
    figure, axs = plt.subplots(2, figsize=(20, 10))
    axs[0].set_title("Loss Function")
    axs[1].set_title("Accuracy")
    axs[0].plot(history.history["loss"], label="train")
    axs[0].plot(history.history["val_loss"], label="val")
    handles, labels = axs[0].get_legend_handles_labels()
    lgd = figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.5, 1))
    axs[0].legend(bbox_to_anchor=(1.1, 1.05))
    axs[0].set(xlabel="Epochs", ylabel="Loss")
    axs[1].plot(history.history["acc"])
    axs[1].plot(history.history["val_acc"])
    axs[1].legend(bbox_to_anchor=(1.1, 1.05))
    axs[1].set(xlabel="Epochs", ylabel="Accuracy")
    figure.savefig(
        plotpath + "loss_newBact.png", bbox_extra_artists=(lgd,), bbox_inches="tight"
    )


def plot_residuals(path, trainY, predictTrain, species):
    """ """
    residuals_training = np.subtract(trainY, predictTrain)
    num1 = 0
    residuals = []
    while num1 < len(species):
        lst = [item[num1] for item in residuals_training]
        residuals.append(lst)
        num1 += 1
    it1 = 0
    while it1 < len(species):
        plt.plot(residuals[it1], "o")
        plt.axhline(y=0, color="r", linestyle="-")
        plt.savefig(path + species[it1] + "-residuals.png")
        plt.cla()
        it1 += 1


def prediction_plot(
    path,
    totalValues,
    predictTrain,
    predictVal,
    predictTest,
    species,
    confidence_list,
    n_steps,
    eval_dictionary,
):  # confidence_list,
    """ """
    # Iterating through train and val results to create lists that can be plotted
    # with the original values to visualise the model
    i = 0
    predictListsX = []
    while i < len(species):
        lst2 = [item[i] for item in predictTrain]
        empty = [np.nan] * n_steps
        empty.extend(lst2)
        # print(lst2)
        predictListsX.append(empty)
        i += 1
    j = 0
    predictListsY = []
    while j < len(species):
        lst2 = [item[j] for item in predictVal]
        # We need the nan values at the beginning of the list, so that the plot can start on a later timepoint for the predicted val values
        empty = [np.nan] * (len(predictListsX[1]) + n_steps)
        empty.extend(lst2)
        # print(lst2)
        predictListsY.append(empty)
        j += 1
    b = 0
    predictListsTest = []
    while b < len(species):
        lst2 = [item[b] for item in predictTest]
        # We need the nan values at the beginning of the list, so that the plot can start on a later timepoint for the predicted val values
        empty = [np.nan] * (len(predictListsY[1]) + n_steps)
        empty.extend(lst2)
        # print(lst2)
        predictListsTest.append(empty)
        b += 1
    x_confidence = np.arange(len(predictListsY[1]), len(predictListsTest[1]))
    y = 0
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    while y < len(species):
        # fig,ax = plt.subplots(figsize=(12,5), dpi=100)
        ax.set(
            title="Moving Pictures",
            xlabel="Date timepoints",
            ylabel="Number of sequences found",
        )
        ax.plot(totalValues[species[y]].values, label="actual")
        ax.plot(predictListsX[y], label="training")
        ax.plot(predictListsY[y], label="val")
        # ax.plot(predictListsTest[y], label = "test")
        ax.plot(x_confidence, confidence_list[y][0], label="upper")
        ax.plot(x_confidence, confidence_list[y][1], label="lower")
        ax.plot(x_confidence, confidence_list[y][2], label="mean")
        ax.fill_between(
            x_confidence, confidence_list[y][0], confidence_list[y][1], alpha=0.2
        )
        ax.legend(loc="upper left")
        # mae_text = 'MAE training-set: ' + str(round(eval_dictionary["mae_train"], 2)) + "\n" + "MAE validation-set: " + str(round(eval_dictionary["mae_val"], 2)) + "\n" + "MAE test-set: " + str(round(eval_dictionary["mae_test"], 2)) + "\n" + "RMSE train-set: " + str(round(eval_dictionary["rmse_train"], 2))+ "\n"+ "RMSE test-set: " + str(round(eval_dictionary["rmse_test"], 2)) #+ "R2 trainig-set: " + str(round(eval_dictionary["r2"], 2)) + "\n"
        mae_text = (
            "MAE : "
            + str(round(eval_dictionary["mae"], 2))
            + "\n"
            + "RMSE test-set: "
            + str(round(eval_dictionary["rmse"], 2))
        )
        anchored_text = AnchoredText(
            mae_text,
            loc="upper left",
            frameon=False,
            bbox_to_anchor=(1.0, 1.0),
            bbox_transform=ax.transAxes,
            prop=dict(fontsize="small"),
        )
        ax.add_artist(anchored_text)
        # ax.text(
        #    0, 0, 'MAE of training-set: ' + str(mae_train) + "\n" + "MAE if validation-set: " + str(mae_val),
        #    horizontalalignement = "right",
        #    verticalalignment = "top",
        #    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
        #    )
        plt.savefig(path + species[y] + "-predictLSTMplt.png")
        plt.cla()
        y += 1


def confidence_plot(path, species, predictTest, confidence_list, n_steps):
    """ """
    b = 0
    predictListsTest = []
    while b < len(species):
        lst2 = [np.nan] * n_steps
        lst2 = [item[b] for item in predictTest]
        # We need the nan values at the beginning of the list, so that the plot can start on a later timepoint for the predicted val values
        predictListsTest.append(lst2)
        b += 1
    x = np.arange(0, len(confidence_list[1][0]))
    print(len(confidence_list[1][0]))
    print(len(confidence_list[1][1]))
    y = 0
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    while y < len(species):
        ax.set(
            title="Moving Pictures",
            xlabel="Date timepoints",
            ylabel="Number of sequences found",
        )
        ax.plot(predictListsTest[y], label="test")
        ax.plot(confidence_list[y][0], label="upper")
        ax.plot(confidence_list[y][1], label="lower")
        ax.fill_between(x, confidence_list[y][0], confidence_list[y][1], alpha=0.2)
        ax.legend(loc="upper left")
        plt.savefig(path + species[y] + "-confidence.png")
        plt.cla()
        y += 1


def shap_featureimportance_plots(model, Xtrain, Xtest, species):
    """ """
    shap.explainers._deep.deep_tf.op_handlers[
        "AddV2"
        ] = shap.explainers._deep.deep_tf.passthrough
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
    shap.explainers._deep.deep_tf.op_handlers[
        "AddV2"
        ] = shap.explainers._deep.deep_tf.passthrough
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
