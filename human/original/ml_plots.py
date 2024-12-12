from numpy import array
from numpy import hstack
from numpy import asarray
from matplotlib.transforms import Bbox
from matplotlib.offsetbox import AnchoredText
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, PolynomialFeatures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

plotpath="../allGutFemale/"

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
    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(bacteria), replace=False)
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


def plot_scaled_family(df):
    scale = MinMaxScaler(feature_range=(0, 1))
    df_scale = df
    df_scale.index = df_scale.index.str.split("; g__").str[0]   
    df_family = df_scale.groupby(df_scale.index).sum()
    families = df_family.index.unique()
    df_family = df_family.T 
    df_family.index = pd.to_datetime(df_family.index, yearfirst = True)
    df_family.sort_index(axis=0, inplace=True)
    df_family = pd.DataFrame(scale.fit_transform(df_family.values), columns=df_family.columns, index=df_family.index)
    return df_family, families


def plot_loss(history):
    figure, axs = plt.subplots(2, figsize=(20,10))
    axs[0].set_title('Loss Function')
    axs[1].set_title('Accuracy')
    axs[0].plot(history.history['loss'], label='train')
    axs[0].plot(history.history['val_loss'], label='val')
    handles, labels = axs[0].get_legend_handles_labels()
    lgd = figure.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.5,1))
    axs[0].legend(bbox_to_anchor=(1.1, 1.05))
    axs[0].set(xlabel='Epochs', ylabel='Loss')
    axs[1].plot(history.history["acc"])
    axs[1].plot(history.history["val_acc"])
    axs[1].legend(bbox_to_anchor=(1.1, 1.05))
    axs[1].set(xlabel='Epochs', ylabel='Accuracy')
    figure.savefig(plotpath+"loss_newBact.png", bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_residuals(path, trainY, predictTrain, species):
    residuals_training = np.subtract(trainY,predictTrain)
    num1 = 0
    residuals = []
    while num1 < len(species):
        lst = [item[num1] for item in residuals_training]
        residuals.append(lst)
        num1 += 1
    it1 = 0
    while it1 < len(species):
        plt.plot(residuals[it1], "o")
        plt.axhline(y = 0, color = "r", linestyle = "-")
        plt.savefig(path+species[it1] + "-residuals.png")
        plt.cla()
        it1 += 1


def prediction_plot(path, totalValues, predictTrain, predictVal, predictTest, species, confidence_list, error_confidence, n_steps, eval_dictionary, outlier_dictionary): #confidence_list,
    # Iterating through train and val results to create lists that can be plotted
    # with the original values to visualise the model
    i = 0
    predictListsX = []
    while i < len(species):
        lst2 = [item[i] for item in predictTrain]
        empty = [np.nan]*n_steps
        empty.extend(lst2)
        #print(lst2)
        predictListsX.append(empty)
        i += 1
    j = 0
    predictListsY = []
    while j < len(species):
        lst2 = [item[j] for item in predictVal]
        #We need the nan values at the beginning of the list, so that the plot can start on a later timepoint for the predicted val values
        empty = [np.nan]*(len(predictListsX[1])+n_steps)
        empty.extend(lst2)
        #print(lst2)
        predictListsY.append(empty)
        j += 1
    b = 0
    predictListsTest = []
    while b < len(species):
        lst2 = [item[b] for item in predictTest]
        #We need the nan values at the beginning of the list, so that the plot can start on a later timepoint for the predicted val values
        empty = [np.nan]*(len(predictListsY[1])+n_steps)
        empty.extend(lst2)
        #print(lst2)
        predictListsTest.append(empty)
        b += 1
    x_confidence = np.arange(len(predictListsY[1]), len(predictListsTest[1]))
    y = 0
    fig,ax = plt.subplots(figsize=(20, 10), dpi=100)
    while y < len(species):
        #fig,ax = plt.subplots(figsize=(12,5), dpi=100)
        ax.set(title= "Moving Pictures", xlabel='Date timepoints', ylabel='Number of sequences found')
        ax.plot(totalValues[species[y]].values, label = "actual")
        ax.plot(predictListsX[y], label = "training")
        ax.plot(predictListsY[y], label = "val")
        #ax.plot(predictListsTest[y], label = "test")
        ax.plot(x_confidence,confidence_list[y][0], label = "upper")
        ax.plot(x_confidence,confidence_list[y][1], label = "lower")
        ax.plot(x_confidence,confidence_list[y][2], label = "mean")
        #Plot prediction interval based on error
        #ax.plot(x_confidence,error_confidence[y][0], label = "upper")
        #ax.plot(x_confidence,error_confidence[y][1], label = "lower")

        print(outlier_dictionary.keys())
        print(species)
        if species[y] in outlier_dictionary.keys():
            print(species[y])
            while species[y] in outlier_dictionary.keys():
                start = outlier_dictionary.keys()[1]+len(x_confidence)
                print(start)
                ax.plot(start, outlier_dictionary.get(species[y]), label="outlier")

        ax.fill_between(x_confidence,confidence_list[y][0], confidence_list[y][1], alpha=0.2)
        #ax.fill_between(x_confidence,error_confidence[y][0], error_confidence[y][1], alpha=0.2)
        ax.legend(loc = "upper left")
        #mae_text = 'MAE training-set: ' + str(round(eval_dictionary["mae_train"], 2)) + "\n" + "MAE validation-set: " + str(round(eval_dictionary["mae_val"], 2)) + "\n" + "MAE test-set: " + str(round(eval_dictionary["mae_test"], 2)) + "\n" + "RMSE train-set: " + str(round(eval_dictionary["rmse_train"], 2))+ "\n"+ "RMSE test-set: " + str(round(eval_dictionary["rmse_test"], 2)) #+ "R2 trainig-set: " + str(round(eval_dictionary["r2"], 2)) + "\n"
        mae_text = "MAE : " + str(round(eval_dictionary["mae"], 2)) + "\n" + "MSE: " +  str((round(eval_dictionary["mse"], 2))) + "\n" + "R2: " + str((round(eval_dictionary["r2"], 2))) + "\n" + "RMSE test-set: " + str(round(eval_dictionary["rmse"], 2)) + "\n" + "NRMSE: " + str(round(eval_dictionary["nrmse"], 2))
        anchored_text = AnchoredText(mae_text, loc= "upper left", frameon=False, bbox_to_anchor=(1., 1.),
                        bbox_transform=ax.transAxes, prop=dict(fontsize="small"))
        ax.add_artist(anchored_text)
        #ax.text(
        #    0, 0, 'MAE of training-set: ' + str(mae_train) + "\n" + "MAE if validation-set: " + str(mae_val),
        #    horizontalalignement = "right",
        #    verticalalignment = "top",
        #    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
        #    )
        plt.savefig(path+ species[y] + "-predictLSTMplt.png")
        plt.cla()
        y += 1


def confidence_plot(path, species, predictTest, confidence_list, n_steps):
    b = 0
    predictListsTest = []
    while b < len(species):
        lst2 = [np.nan]*n_steps
        lst2 = [item[b] for item in predictTest]
        #We need the nan values at the beginning of the list, so that the plot can start on a later timepoint for the predicted val values
        predictListsTest.append(lst2)
        b += 1
    x = np.arange(0,len(confidence_list[1][0]))
    print(len(confidence_list[1][0]))
    print(len(confidence_list[1][1]))
    y = 0
    fig,ax = plt.subplots(figsize=(20, 10), dpi=100)
    while y < len(species):
        ax.set(title= "Moving Pictures", xlabel='Date timepoints', ylabel='Number of sequences found')
        ax.plot(predictListsTest[y], label = "test")
        ax.plot(confidence_list[y][0], label = "upper")
        ax.plot(confidence_list[y][1], label = "lower")
        ax.fill_between(x, confidence_list[y][0], confidence_list[y][1], alpha=0.2)
        ax.legend(loc = "upper left")
        plt.savefig(path+species[y] + "-confidence.png")
        plt.cla()
        y += 1