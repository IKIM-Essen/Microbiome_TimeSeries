import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.utils.config import load_config

#time_series_analysis_plot(species, complete, "/local/work/16S/snakemake_qiime/16S/MachineLearning/Wastewater/BigTimeseriesMetadata/plot_original_all.png", dic_taxa)

def _load_bacteria_mapping(bacteria):
    if isinstance(bacteria, str):
        with open(bacteria, "rb") as file:
            bacteria = pickle.load(file)
    if isinstance(bacteria, dict):
        columns = list(bacteria.keys())
        labels = list(bacteria.values())
    elif isinstance(bacteria, list):
        columns = bacteria
        labels = bacteria
    else:
        raise ValueError("bacteria must be a mapping path, dict, or list of column names")
    return columns, labels


def time_series_analysis_plot(dataframe_complete_path, filename, bacteria):
    print("Plotting")
    dataframe_complete = pd.read_csv(dataframe_complete_path, index_col=0)
    plot_columns, plot_labels = _load_bacteria_mapping(bacteria)
    # Creating colors for the plot
    np.random.seed(100)
    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(plot_columns), replace=True)
    # Creating a plot displaying the OTU's found
    figure, ax = plt.subplots(figsize=(30,10))
    figure.legend(loc=2, prop={'size': 6})
    plt.rcParams["figure.figsize"] = (20,30)
    figure.legend(bbox_to_anchor=(1.1, 1.05))
    for column, label, color in zip(plot_columns, plot_labels, mycolors):
        if column in dataframe_complete.columns:
            ax.plot(dataframe_complete.index, dataframe_complete[column], color=color, label=label)
        else:
            raise KeyError(f"Column '{column}' not found in dataframe_complete")
    d = {"down" : 30, "up" : -30}
    handles, labels = ax.get_legend_handles_labels()
    lgd = figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.9, 0, 0.07, 0.9))  
    plt.gca().set(ylabel='Number of species found', xlabel='Date')
    plt.yticks(fontsize=12, alpha=.7)
    if len(dataframe_complete.index) < 50:
        plt.xticks(fontsize=10, rotation=45)
    if len(dataframe_complete.index) > 50:
        plt.xticks(fontsize=8, rotation=45)
    plt.title("Display of microbial species in patient dependent on time", fontsize=20)
    plt.show()
    figure.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')