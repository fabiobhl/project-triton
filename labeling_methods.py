import os
import json

import numpy as np
from numpy.lib.polynomial import poly 
import pandas as pd 
from scipy import signal

from database import DataBase
import utils


def check_if_labels_exist(database_path, candlestick_interval, label_name):
    return os.path.isdir(f"{database_path}/{candlestick_interval}/labels/{label_name}")

def write_labels(database_path, candlestick_interval, label_name, labels, complete_array, specifications):
    """
    Utility funtion for setting up the files
    """

    #check if the labels order alredy exists if not create one
    if not os.path.isdir(f"{database_path}/{candlestick_interval}/labels"):
        os.mkdir(f"{database_path}/{candlestick_interval}/labels")

    path = f"{database_path}/{candlestick_interval}/labels/{label_name}"

    #create folder for labeling method if it does not already exist
    if check_if_labels_exist(database_path=database_path, candlestick_interval=candlestick_interval, label_name=label_name):
        raise Exception("Your chosen label-name already exists")
    else:
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

def calculate_profit(close, labels):
    #calculate the profit
    array = np.empty((close.shape[0],2))
    array[:,0] = close
    array[:,1] = labels

    return utils.calculate_profit(array, 0.075)

def labelingmethod(func):
    def inner(*args, **kwargs):
        #get the labels
        labels = func(*args, **kwargs)

        #initialize the database
        db = DataBase(kwargs["database_path"])
        #get the data
        close = db[kwargs["candlestick_interval"], "close"].to_numpy().reshape(-1)

        #calculate specific profit
        specific_profit, complete_array = calculate_profit(close=close, labels=labels)

        #create the specs dic
        specs = {
            "labeling_method": func.__name__,
            "specific_profit": specific_profit,
            "method_params": {
                "window_length": kwargs["window_length"],
                "poly_order": kwargs["poly_order"],
                "min_order": kwargs["min_order"],
                "max_order": kwargs["max_order"]
            }
        }

        if kwargs["write"]:
            #check if name was specified
            if "name" not in kwargs.keys():
                raise Exception("If you want to write your labels to disk you need to specify a name")
            #write to harddrive
            write_labels(database_path=db.path, candlestick_interval=kwargs["candlestick_interval"], label_name=kwargs["name"], labels=labels, complete_array=complete_array, specifications=specs)

        return labels

    return inner

class LabelingMethods():

    labeling_methods = ["smoothing_extrema_labeling"]

    @staticmethod
    @labelingmethod
    def smoothing_extrema_labeling(database_path, candlestick_interval, write, window_length, poly_order, min_order, max_order, name=None):
        #load in the data
        db = DataBase(database_path)
        #load in the close data
        close = db[candlestick_interval, "close"].to_numpy().reshape(-1)
        
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




    
if __name__ == "__main__":
    print(LabelingMethods.smoothing_extrema_labeling(database_path="./databases/eth2", candlestick_interval="15m", write=True, name="test", window_length=11, poly_order=3, min_order=3, max_order=3))