import os
import sys
import json

import numpy as np 
import pandas as pd 
from scipy import signal

from database import DataBase
from utils import calculate_profit

def write(database_path, labeling_method_name, labels, complete_array, specifications):
    """
    Utility funtion for setting up the files
    """

    #check if the labels order alredy exists if not create one
    if not os.path.isdir(f"{database_path}/labels"):
        os.mkdir(f"{database_path}/labels")

    path = f"{database_path}/labels/{labeling_method_name}"

    #create folder for labeling method if it does not already exist
    if not os.path.isdir(path):
        os.mkdir(path)

    #write the labels
    labels = pd.DataFrame(labels)
    labels.to_csv(path_or_buf=f"{path}/labels.csv", header=False)

    #write the complete array
    complete_array = pd.DataFrame(complete_array)
    complete_array.to_csv(path_or_buf=f"{path}/complete_array.csv", header=False)

    #write the specifications
    with open(f"{path}/specifications.json", 'w') as fp:
        json.dump(specifications, fp)

def smoothing_extrema_labeling(database, window_length, poly_order, min_order, max_order):
    """
    Labeling method for labeling time-series data
    """
    
    #function for actually doing the labeling
    def labeling(close, window_length, poly_order, min_order, max_order):
        #smooth the close data
        smoothed_close = signal.savgol_filter(x=close, window_length=window_length, polyorder=poly_order)
        
        #get the minima
        minima_indices = signal.argrelmin(smoothed_close, order=min_order)
        minima = np.zeros(close.shape[0])
        minima[:] = np.nan
        minima[minima_indices] = smoothed_close[minima_indices]

        #get the maxima
        maxima_indices = signal.argrelmax(smoothed_close, order=max_order)
        maxima = np.zeros(close.shape[0])
        maxima[:] = np.nan
        maxima[maxima_indices] = smoothed_close[maxima_indices]
        
        #create the labels
        labels = np.zeros(close.shape[0])
        labels[minima_indices] = 1
        labels[maxima_indices] = 2

        return labels

    #get the data
    close = database["5m", "close"].to_numpy().reshape(-1)

    #get the labels
    labels = labeling(close=close, window_length=window_length, poly_order=poly_order, min_order=min_order, max_order=max_order)

    #calculate the profit
    array = np.empty((close.shape[0],2))
    array[:,0] = close
    array[:,1] = labels

    specific_profit, complete_array = calculate_profit(array, 0.075)

    #create the specs dic
    specs = {
        "specific_profit": specific_profit,
        "method_params": {
            "window_length": window_length,
            "poly_order": poly_order,
            "min_order": min_order,
            "max_order": max_order
        }
    }

    #write to harddrive
    write(database.path, "smoothing_extrema_labeling", labels, complete_array, specs)

    return specific_profit
    
if __name__ == "__main__":
    db = DataBase(path="./databases/ethtest")

    smoothing_extrema_labeling(database=db, window_length=21, poly_order=3, min_order=5, max_order=5)