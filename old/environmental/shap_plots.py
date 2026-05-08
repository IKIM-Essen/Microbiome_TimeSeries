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

plotpath="Dinslaken/LSTM/"

def shap_featureimportance_plots(model, Xtrain, ytrain, Xtest, taxa_dict, columns):
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    DE = shap.DeepExplainer(model, Xtrain) # X_train is 3d numpy.ndarray
    shap_values = DE.shap_values(Xtrain, check_additivity=False) # X_validate is 3d numpy.ndarray
    print(shap_values[0][0].shape)
    print(shap_values[0].shape)
    print(shap_values[0][0])
    print(Xtrain.shape)
    #print(type(species))
    shap_values_2D = shap_values[0].reshape(shap_values[0].shape[0],shap_values[0].shape[2])
    #shap_values_2D = shap_values[0].reshape(-1,len(species)+3)
    X_train_2D = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[2])
    #print(shap_values_2D.shape, X_train_2D.shape)

    new_columns = [entry.split('_')[0] for entry in columns]
    print(new_columns)
    new_list = []
    for entry in columns:
        if '_' in entry:
            # Split the entry at the underscore
            parts = entry.split('_')
            # Replace the first part with the dictionary value if it exists
            if parts[0] in taxa_dict:
                new_entry = taxa_dict[parts[0]] + '_' + '_'.join(parts[1:])
            else:
                new_entry = entry  # If no mapping is found, keep the original entry
        else:
            new_entry = entry  # Keep the original entry if no underscore is found
        new_list.append(new_entry)
    print(new_list)

    plt.clf()
    plt.figure(figsize=(20,40))
    shap.initjs()
    shap.summary_plot(
        shap_values_2D,
        X_train_2D,
        feature_names=new_list,
        plot_type='bar',
        show=False)
    plt.savefig(plotpath+"shap_wM.png", bbox_inches='tight')


    plt.clf()
    shap.summary_plot(
        shap_values_2D,
        X_train_2D,
        feature_names=new_list,
        show=False)
    plt.savefig(plotpath+"shap_dotplot_wM.png", bbox_inches='tight')

    plt.clf()
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    print(ytrain.shape)
    Xflatten = model.predict(Xtrain).reshape(ytrain.shape[0],ytrain.shape[1])
    #Xtestflatten = model.predict(Xtest).reshape(-1,38)
    Xtestflatten = Xtest.reshape(Xtest.shape[0],Xtest.shape[2])
    #print(Xtestflatten.shape)
    #print(Xflatten.shape)
    DE = shap.DeepExplainer(model, Xtrain) # X_train is 3d numpy.ndarray
    shap_values = DE.shap_values(Xtest, check_additivity=False) # X_validate is 3d numpy.ndarray
    shap_values_2D = shap_values[0].reshape(shap_values[0].shape[0],shap_values[0].shape[2])
    #print(testY.shape)
    #print(predictTest.shape)
    #print(shap_values_2D.shape)
    #print(shap_values[0])
    #print(shap_values[0].shape)
    #print(shap_values[0][0])
    #print(shap_values[0][0].shape)
    #print(shap_values[0][0][0])
    #df_shapvalue = pd.DataFrame(shap_values, columns=species.tolist())
    #print(df_shapvalue["d__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Bacteroidaceae; g__Bacteroides"].values)
    #shap.dependence_plot(0,shap_values[0][0],Xtest[0],species.tolist())
    shap.dependence_plot(0,shap_values_2D, Xtestflatten, new_list)
    plt.savefig(plotpath+"dependence_wM.png", bbox_inches='tight')
    #print(DE.expected_value)
    #print(DE.expected_value.shape)

    plt.clf()
    shap.plots.force(DE.expected_value[0],shap_values[0][0][0], Xtest[0][0], new_list, matplotlib=True, show = False)
    plt.savefig(plotpath+"force_wM.png", bbox_inches='tight')


    plt.clf()
    shap.plots._waterfall.waterfall_legacy(DE.expected_value[0],shap_values[0][0][0], feature_names = new_list)
    plt.savefig(plotpath+"waterfall_wM.png", bbox_inches='tight')
    plt.clf()
    plt.cla()