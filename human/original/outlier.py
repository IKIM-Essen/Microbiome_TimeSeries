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
import csv

plotpath="../allGutFemale/"

def outlier_detection(predictionInterval, actualValues, bacteria):
    i = 0
    outliers = {}
    #print(len(predictionInterval))
    #print(len(predictionInterval[1][0]))
    #print(predictionInterval[1][0][1])
    #print(len(actualValues))
    #print(actualValues[1][0])
    actualValues1 = np.transpose(actualValues)
    #print(len(actualValues1))
    #print(actualValues1.shape)
    while i < len(actualValues1):
        #print("i="+str(i))
        upper = predictionInterval[i][0]
        #print("upper")
        lower = predictionInterval[i][1]
        #print("lower")
        #print("while2")
        y = 0
        while y < len(predictionInterval[i][0]):
            #print("y="+str(y))
            if actualValues1[i][y] < lower[y]:
                outliers[bacteria[i],y] = actualValues1[i][y]
            elif actualValues1[i][y] > upper[y]:
                outliers[bacteria[i],y] = actualValues1[i][y]
            y = y+1
            #print(y)
        i = i+1
        #print(i)
    with open(plotpath+'outliers_testset.csv', 'w') as f:
        w = csv.DictWriter(f, outliers.keys())
        w.writeheader()
        w.writerow(outliers)
    return outliers
