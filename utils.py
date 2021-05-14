import json
import numpy as np
import datetime
import time

"""
To-Do:
    -test the calculate_profit function (not tested at all!)
"""

def read_config(path=None):
    """
    Function for reading in the config.json file
    """
    #create the filepath
    if path:
        if "config.json" in path:
            file_path = path
        else:
            file_path = f"{path}/config.json"
    else:
        file_path = "config.json"
    
    #load in config
    try:
        with open(file_path, "r") as json_file:
            config = json.load(json_file)
    except Exception:
        raise Exception("Your config file is corrupt (wrong syntax, missing values, ...)")

    return config

def read_json(path):
    #try to read in the dictionary
    try:
        with open(path, "r") as json_file:
            dictionary = json.load(json_file)
    except Exception:
        raise Exception("Your json file is corrupt (wrong syntax, missing values, wrong path, ...)")

    return dictionary

def calculate_profit(input_array, trading_fee):
    """
    Description:
        This function takes a nx2 numpy array, where the first column consists of n prices and the second column consists of n, corresponding labels (0: Hold, 1: Buy, 2: Sell)
    Arguments:
        -input_array[2xn np.array]:     0: prices
                                        1: labels

        -trading_fee[float]:            The trading fee of the market in percentage!!
    Return:
        - specific_profit[float]:       the specific profit, calculated from the labels
        
        - ret_array (5xn np.array):     0: prices
                                        1: labels
                                        2: the price if the action was hold
                                        3: the price if the action was buy
                                        4: the price if the action was sell
    """
    
    #check if the array has the correct shape:
    if len(input_array.shape) != 2 or input_array.shape[1] != 2:
        raise Exception("Your provided input_array does not satisfy the correct shape conditions")
    
    #convert trading fee from percentage to decimal
    trading_fee = trading_fee/100

    #create output_array
    output_array = np.zeros(shape=(input_array.shape[0], 5))
    output_array[:,0] = input_array[:,0]    #price column
    output_array[:,1] = input_array[:,1]    #label column
    output_array[:,2] = np.nan              #hold_price column
    output_array[:,3] = np.nan              #buy_price column
    output_array[:,4] = np.nan              #sell_price column

    #create the specific_profit variable
    specific_profit = 0

    #set the mode to buy
    mode = 'buy'

    #calculate the profit
    for i in range(0, output_array.shape[0]):
        #get the action
        action = output_array[i,1]

        #save the action
        if action == 0:     #hold
            output_array[i, 2] = output_array[i, 0]
        elif action == 1:   #buy
            output_array[i, 3] = output_array[i, 0]
        elif action == 2:   #sell
            output_array[i, 4] = output_array[i, 0]
        else:
            raise Exception(f"Your labels contained values that are not valid! value:{action}")

        #do the trading
        if mode == 'buy' and action == 1:
            tc_buyprice = output_array[i, 0]                                     #tc = tradingcoin
            #set mode to sell
            mode = 'sell'

        elif mode == 'sell' and action == 2:
            #get the sellprice
            tc_sellprice = output_array[i, 0]
            #calculate specific_profit of this trade
            local_specific_profit = (tc_sellprice/tc_buyprice)*(1-trading_fee)*(1-trading_fee)-1
            #add specific_profit to overall profit
            specific_profit += local_specific_profit
            #set mode to buy
            mode = 'buy'
    
    return specific_profit, output_array

def rolling_window(input_array, window_size):
    """
    Description:
        This function takes numpy array in the shape: (timesteps, features) and rolls it with a rolling window (stepsize=1).
    Arguments:
        -input_array[np.array]:         array in the shape: (timesteps, features) that gets rolled, this array should not contain the labels!
        -window_size[int]:              How many timesteps a window should contain
    Return:
        - windows[np.array]:            The rolled array in the shape: (windows_amount, timesteps, features)
    """
    #save the shape
    shape = input_array.shape
    
    #save the length 
    length = shape[0]

    #flatten the array
    array_flat = input_array.flatten()

    #amount of features
    features_amount = shape[1]
    #amount of elements in one window
    window_elements = window_size*features_amount
    #amount of elements to skip in next window
    elements_skip = features_amount
    #amount of windows to generate
    windows_amount = length-window_size+1

    #define the indexer
    indexer = np.arange(window_elements)[None, :] + elements_skip*np.arange(windows_amount)[:, None]

    #get the rolling windows
    windows = array_flat[indexer]
    #resize to correct dimensions
    windows = windows.reshape(windows_amount, window_size, features_amount)

    return windows

def timer(candlestick_interval):
    if candlestick_interval != "5m":
        raise Exception("Your wished candlestick_interval has not been implemented yet")
    
    #incase the timer got called immediately after a 5 minute
    minute = int(candlestick_interval[0])
    while datetime.datetime.now().minute % minute == 0:
        time.sleep(1)
    while datetime.datetime.now().minute % minute != 0:
        time.sleep(1)

if __name__ == "__main__":
    pass