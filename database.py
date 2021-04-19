#standard python libraries
import json
import atexit
import datetime
import os
import warnings
import time
import sys
import uuid
import shutil
import math

#external libraries
from binance.client import Client
import pandas as pd
import numpy as np
import ta
from sklearn import preprocessing
from matplotlib import pyplot as plt
import torch

#external methods
from utils import read_config, rolling_window
"""
ToDo:
    -implement full candlestick_interval choosing: database.create(), labeling imports/creationgs, test everything
    -implement scaling every window by itself (more accurate to usecase)
"""

class dbid():
    """
    Description:
        Class which can be used like a dictionary.
        This Class is not threadsafe! The changes to the dictionary, only get written to disk when the instance goes out of scope!
    Arguments:
        -path (string):     Path of the database
    """

    def __init__(self, path):
        self.path = f"{path}/dbid.json"

        #load in the dbid
        with open(self.path) as json_file:
            self.dbid = json.load(json_file)

        #register the dump at the end of lifetime
        atexit.register(self.dump)
        
    def __getitem__(self, key):
        return self.dbid[key]

    def __setitem__(self, key, item):
        #change the dict in ram
        self.dbid[key] = item

    def dump(self):
        #save changes to json file
        with open(self.path, 'w') as fp:
            json.dump(self.dbid, fp,  indent=4)

class DataBase():
    """
    Description:
        This is the base Database class, on which every other Database Objects builds upon.
    Arguments:
        -path[string]:  Path of the Database
    """
    def __init__(self, path):
        #save the params
        self.path = path

        #check if the path exists and is a database
        if not os.path.isdir(path):
            raise Exception("The path you chose is not existing")
        if not os.path.isfile(f"{path}/dbid.json"):
            raise Exception("The path you chose is not a DataBase")
        
        #setup dbid
        self.dbid = dbid(path=self.path)
    
    def __getitem__(self, index):
        """
        Description:
            Method for accessing data of the database. The access is direct from the harddrive (slower but more memory efficient)
        Arguments:
            -index[string, list]:   Generally: [candlestick_interval, list of features]. To access the whole dataframe only specify the candlestick_interval you want e.g. db["5m"].
                                    To access only one feature specify the datatype and the feature you want e.g. db["5m", "close"]
                                    To access multiple features specify the datatype and a list of features you want e.g. db["5m", ["close", "open"]]

        Return:
            -data[pd.DataFrame]:    Returns always a DataFrame in the shape (rows, number of specified features) 
        """
        #access whole dataframe of certain datatype
        if type(index) == str:
            #load in the data and return
            try:
                data = pd.read_csv(filepath_or_buffer=f"{self.path}/{index}", index_col="index")
                
                #convert the date columns
                data["close_time"]= pd.to_datetime(data["close_time"])
                data["open_time"]= pd.to_datetime(data["open_time"])
                
                return data
            
            except FileNotFoundError:
                raise Exception("Your chosen datatype is not available in this DataBase")
            
        #access one column of a dataframe of certain datatype
        elif type(index) == tuple and len(index) == 2 and type(index[0]) == str and type(index[1]) == str:
            #load in the data and return
            try:
                data = pd.read_csv(filepath_or_buffer=f"{self.path}/{index[0]}", usecols=[index[1]])
                
                #convert the date columns
                if "close_time" in data.columns:
                    data["close_time"]= pd.to_datetime(data["close_time"])
                if "open_time" in data.columns:
                    data["open_time"]= pd.to_datetime(data["open_time"])

                return data
            
            except FileNotFoundError:
                raise Exception("Your chosen datatype is not available in this DataBase")
            
        #access list of columns of a dataframe of certain datatype
        elif type(index) == tuple and len(index) == 2 and type(index[0]) == str and type(index[1]) == list:
            #load in the data and return
            try:
                data = pd.read_csv(filepath_or_buffer=f"{self.path}/{index[0]}", usecols=index[1])
                
                #convert the date columns
                if "close_time" in data.columns:
                    data["close_time"]= pd.to_datetime(data["close_time"])
                if "open_time" in data.columns:
                    data["open_time"]= pd.to_datetime(data["open_time"])

                return data
            
            except FileNotFoundError:
                raise Exception("Your chosen datatype is not available in this DataBase")
        
        #throw error on all other accesses
        else:
            raise Exception("Your index is not possible, please check your index and the documentation on the DataBase object")

    def get_labels(self, labeling_method):
        """
        Description:
            Method for accessing labels (only the indexlables) of the database. The access is direct from the harddrive (slower but more memory efficient)
        Arguments:
            -labeling_method[string]:       The labels of which labelingmethod you want. (Needs to be applied to this database already!)
        Return:
            -labels[pd.DataFrame]:          Returns always a DataFrame in the shape (rows, 1) 
        """
        #check that labeling_method is a string and exists
        if type(labeling_method) == str and os.path.isdir(f"{self.path}/labels/{labeling_method}"):
            pass
        else:
            raise Exception("Your chosen labelingmethod is not available/has not been applied yet")

        labels = pd.read_csv(filepath_or_buffer=f"{self.path}/labels/{labeling_method}/labels.csv", header=None, index_col=0, names=["index", "labels"])

        return labels

    def get_extended_labels(self, labeling_method):
        """
        Description:
            Method for accessing labels and the corresponding price levels per label of the database. The access is direct from the harddrive (slower but more memory efficient)
        Arguments:
            -labeling_method[string]:       The labels of which labelingmethod you want. (Needs to be applied to this database already!)
        Return:
            -labels[pd.DataFrame]:          Returns always a DataFrame in the shape (rows, 5) where the columns are:    "close":        The normal close prices
                                                                                                                        "labels":       The labels in indexlabel form
                                                                                                                        "hold_price":   The price of the close price where the action is hold
                                                                                                                        "buy_price":    The price of the close price where the action is buy
                                                                                                                        "sell_price":   The price of the close price where the action is sell
        """
        #check that labeling_method is a string and exists
        if type(labeling_method) == str and os.path.isdir(f"{self.path}/labels/{labeling_method}"):
            pass
        else:
            raise Exception("Your chosen labelingmethod is not available/has not been applied yet")

        labels = pd.read_csv(filepath_or_buffer=f"{self.path}/labels/{labeling_method}/complete_array.csv", header=None, index_col=0, names=["index", "close", "labels", "hold_price", "buy_price", "sell_price"])

        return labels
    
    @classmethod
    def create(cls, save_path, symbol, date_span, candlestick_interval, config_path=None):
        """
        Description:
            This method creates a DataBase-Folder at a given location with the specified data.           
        Arguments:
            -save_path[string]:             The location, where the folder gets created (Note: The name of the folder should be in the save_path e.g: "C:/.../desired_name")
            -symbol[string]:                The Cryptocurrency you want to trade (Note: With accordance to the Binance API)
            -date_span[tuple]:              Tuple of datetime.date objects in the form: (startdate, enddate)
            -candlestick_interval[string]:  On what interval the candlestick data should be downloaded
            -config_path[string]:           Path to the config file, if none is given, it is assumed that the config-file is in the same folder as the file this method gets called from
        Return:
            -DataBase[DataBase object]:     Returns the created DataBase object
        """
        #check if the specified directory already exists
        if os.path.isdir(save_path):
            raise Exception("Please choose a directory, that does not already exist")
        
        """
        Download the data, add the tas
        """
        #read in the config
        config = read_config()

        #create the client
        client = Client(api_key=config["binance"]["key"], api_secret=config["binance"]["secret"])

        #get the dates
        startdate = date_span[0].strftime("%d %b, %Y")
        enddate = date_span[1].strftime("%d %b, %Y")

        #download the data and safe it in a dataframe
        raw_data = client.get_historical_klines(symbol=symbol, interval=candlestick_interval, start_str=startdate, end_str=enddate)
        data = pd.DataFrame(raw_data)

        #clean the dataframe
        data = data.astype(float)
        data.drop(data.columns[[7,8,9,10,11]], axis=1, inplace=True)
        data.rename(columns = {0:'open_time', 1:'open', 2:'high', 3:'low', 4:'close', 5:'volume', 6:'close_time'}, inplace=True)

        #set the correct times
        data['close_time'] += 1
        data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
        data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')

        #check for nan values
        if data.isna().values.any():
            raise Exception("Nan values in data, please discard this object and try again")
        
        #add the technical analysis data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            data = ta.add_all_ta_features(data, open='open', high="high", low="low", close="close", volume="volume", fillna=True)

        #drop first 60 rows
        data = data.iloc[60:]

        #reset the index
        data.reset_index(inplace=True, drop=True)

        """
        create the directory and save the csv's
        """
        #create the directory
        os.mkdir(save_path)

        #save the data to csv's
        data.to_csv(path_or_buf=f"{save_path}/{candlestick_interval}", index_label="index")
        
        #creating the dbid
        dbid = {
            "symbol": symbol,
            "date_range": (startdate, enddate),
            "candlestick_interval": candlestick_interval
        }

        #save the dbid
        with open(f"{save_path}/dbid.json", 'w') as fp:
            json.dump(dbid, fp,  indent=4)
        
        return cls(path=save_path)

class TrainDataBase(DataBase):
    """
    Description:
        This Database can be used for supervised training of time-series models. You need a DataBase first with all the data.
    Arguments:
        -path[string]:                  The path of your DataBase
        -candlestick_interval[string]:  The candlestick interval you want to use for your training, (Note: According to the python-binance constants)
        -derived[boolean]:              If you want your data to be derived choose True, if not choose False
        -features[list]:                The features you want fo use for your training
        -batch_size[int]:               The size of the batches
        -window_size[int]:              The size of the rolling window
        -labeling_method[string]:       What kind of labels you want to use (from what method the labels were created)
        -scaling_method[string]:        What kind of scaling you want you can choose between:   "global":   The maximum of the whole dataset will correspond to: 1 and the minimum to -1
                                                                                                "local":    The maximum of the window will correspond to: 1 and the minimum to -1
                                                                                                "none":     There will be no scaling
        -test_percentage[float]:        What percentage of your dataset should be used as tests
        -device[torch.device]:          The device you want your data on, if set to None then the device gets autodetected (utilises cuda if it is available)
    """
    def __init__(self, path, DHP, device=None):
        #calling the inheritance
        super().__init__(path)

        #get id
        self.id = uuid.uuid1()
        #create temporary folder
        os.mkdir(f"{self.path}/{self.id}")

        #cleanup when we go out of scope
        atexit.register(self._cleanup)

        #safe the variables
        self.DHP = DHP
        #auto detection for device
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        #prepare the data
        self._prepare_data()

        #load in the mmap
        self.train_data = np.load(file=f"{self.path}/{self.id}/train_data.npy", mmap_mode="r")
        self.train_labels = np.load(file=f"{self.path}/{self.id}/train_labels.npy", mmap_mode="r")
        self.test_data = np.load(file=f"{self.path}/{self.id}/test_data.npy", mmap_mode="r")
        self.test_labels = np.load(file=f"{self.path}/{self.id}/test_labels.npy", mmap_mode="r")

    def train(self):
        """
        Description:
            Method for training your nn.
        Arguments:
            -none
        Return:
            -iterable_data[generator]:      Returns a generator on which you can call next() until it is empty (Note: Throws error!)
        """
        for i in range(self.train_data.shape[0]):
            data = np.array(self.train_data[i])
            labels = np.array(self.train_labels[i])
            
            yield torch.tensor(data, device=self.device), torch.tensor(labels, dtype=torch.long, device=self.device)
    
    def test(self):
        """
        Description:
            Method for testing your neural network.
        Arguments:
            -none
        Return:
            -iterable_data[generator]:      Returns a generator on which you can call next() until it is empty (Note: Throws error!)
        """
        for i in range(self.test_data.shape[0]):
            data = np.array(self.test_data[i])
            labels = np.array(self.test_labels[i])

            yield torch.tensor(data, device=self.device), torch.tensor(labels, dtype=torch.long, device=self.device)

    def _prepare_data(self):
        """
        Description:
            Method for preparing the data (rolling, scaling, batching, ...)
        Arguments:
            -none
        Return:
            -nothing
        """
        #get the labels
        labels = self.get_labels(self.DHP["labeling_method"]).to_numpy()

        #select the features
        data = self[self.DHP["candlestick_interval"], self.DHP["features"]]

        #derive the data
        if self.DHP["derived"]:
            data = self._derive(data)
            data = data.iloc[1:,:]
        
        #convert data to numpy array
        data = data.to_numpy()

        #scale
        self.scaler_params = None
        if self.DHP["scaling_method"] == "global":
            scaler = preprocessing.MaxAbsScaler(copy=False)
            scaler.fit_transform(data)

            #safe the scaler params
            self.scaler_params = scaler.get_params()

        #roll the data (rolling window)
        windows = rolling_window(data, self.DHP["window_size"])
        
        #cut first (window_size-1) elements from labels
        labels = labels[self.DHP["window_size"]-1:]

        #batch the windows
        batches_amount = math.floor(windows.shape[0]/self.DHP["batch_size"])
        windows = windows[0:batches_amount*self.DHP["batch_size"],:,:]
        batches = windows.reshape(batches_amount, self.DHP["batch_size"], self.DHP["window_size"], -1)

        #batch the labels
        labels = labels[0:batches_amount*self.DHP["batch_size"]]
        labels = labels.reshape(batches_amount, self.DHP["batch_size"])

        #split into train/test data
        train_amount = batches_amount - math.floor(batches_amount*self.DHP["test_percentage"])
        train_data = batches[0:train_amount]
        test_data = batches[train_amount:]
        train_labels = labels[0:train_amount]
        test_labels = labels[train_amount:]

        #save into folder
        np.save(f"{self.path}/{self.id}/train_data",train_data)
        np.save(f"{self.path}/{self.id}/test_data",test_data)
        np.save(f"{self.path}/{self.id}/train_labels",train_labels)
        np.save(f"{self.path}/{self.id}/test_labels",test_labels)
    
    def _derive(self, data):
        """
        Method for deriving the data: from absolute values to relative values
        """

        pct = ["open", "high", "low", "close", "volume_nvi", "volume_vwap", "volatility_atr", "volatility_bbm", "volatility_bbh", "volatility_bbl", "volatility_bbw", "volatility_kcc", "volatility_kch", "volatility_kcl", "volatility_kcw", "volatility_dcl", "volatility_dch", "volatility_dcm", "volatility_dcw", "trend_sma_fast", "trend_sma_slow", "trend_ema_fast", "trend_ema_slow", "trend_mass_index", "trend_ichimoku_conv", "trend_ichimoku_base", "trend_ichimoku_a", "trend_ichimoku_b", "trend_visual_ichimoku_a", "trend_visual_ichimoku_b", "trend_psar_up", "trend_psar_down", "momentum_uo", "momentum_kama"]
        diff = ["volume", "volume_adi", "volume_obv", "volume_mfi", "volatility_ui", "trend_adx", "momentum_rsi", "momentum_wr", "others_cr"]
        none = ["volume_cmf", "volume_fi", "volume_em", "volume_sma_em", "volume_vpt", "volatility_bbhi", "volatility_bbli", "volatility_bbp", "volatility_kcp", "volatility_kchi", "volatility_kcli", "volatility_dcp", "trend_macd", "trend_macd_signal", "trend_macd_diff", "trend_adx_pos", "trend_adx_neg", "trend_vortex_ind_pos", "trend_vortex_ind_neg", "trend_vortex_ind_diff", "trend_trix", "trend_cci", "trend_dpo", "trend_kst", "trend_kst_sig", "trend_kst_diff", "trend_aroon_up", "trend_aroon_down", "trend_aroon_ind", "trend_psar_up_indicator", "trend_psar_down_indicator", "trend_stc", "momentum_stoch_rsi", "momentum_stoch_rsi_k", "momentum_stoch_rsi_d", "momentum_tsi", "momentum_stoch", "momentum_stoch_signal", "momentum_ao", "momentum_roc", "momentum_ppo", "momentum_ppo_signal", "momentum_ppo_hist", "others_dr", "others_dlr"]

        #extract the chosen features
        pct_features = [x for x in pct if x in data.columns]
        diff_features = [x for x in diff if x in data.columns]

        data[pct_features] = data[pct_features].pct_change()
        data[diff_features] = data[diff_features].diff()

        return data

    def _cleanup(self):
        """
        This method gets called when the TrainDataBase goes out of scoope and deletes the temporary folder
        """
        shutil.rmtree(f"{self.path}/{self.id}")

    @classmethod
    def control_dataprep(cls, path, window_size, batch_size=10):
        """
        Description:
            A method for checking your data and the rolling process
        Arguments:
            -path[string]:          Path to the DataBase
            -window_size[int]:      The size of your windows
            -batch_size[int]:       The size of the batches
        Return:
            -nothing, runs a interactive graph which can be iterated by clicking any button.
        """
        DHP = {
            "candlestick_interval": "5m",
            "derived": False,
            "features": ["close", "open", "volume"],
            "batch_size": batch_size,
            "window_size": window_size,
            "labeling_method": "smoothing_extrema_labeling",
            "scaling_method": "none",
            "test_percentage": 0.2
        }
        #load in the tdb
        tdb = cls(path=path, DHP=DHP, device="cpu")
        #get the windows generator
        windows = tdb.train()
        #get the data unbatches/rolled
        data = tdb["raw_data", ["close", "open", "volume"]].iloc[0:1000, :]
        minimum = data["open"].min()
        maximum = data["open"].max()
        labels = tdb.get_extended_labels("smoothing_extrema_labeling").iloc[0:1000,:]
        #craete figure
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        fig.show()

        for batch_index in range(10):
            #get the batch
            batch, batch_labels = next(windows)
            
            for i in range(batch_size):
                index = batch_index*10 + i
                print(index)

                #ax2
                ax2.cla()
                ax2.grid()
                ax2.plot(batch[i, :, 1])
                ax2.set_title(f"{batch_labels[i]}")
                ax2.set_xlim(left=-2, right=window_size-1+2)
                ax2.set_ylim(bottom=ax2.get_ylim()[0]-2, top=ax2.get_ylim()[1]+2)
                
                #ax1
                ax1.cla()
                ax1.grid()
                ax1.plot(data["close"])
                ax1.plot(labels["hold_price"], marker="o", linestyle="", color="gray")
                ax1.plot(labels["buy_price"], marker="o", linestyle="", color="green")
                ax1.plot(labels["sell_price"], marker="o", linestyle="", color="red")
                ax1.vlines(x=[index, index+window_size-1], ymin=minimum, ymax=maximum, color="red")
                ax1.set_xlim(left=index-2, right=index+window_size-1+2)
                ax1.set_ylim(bottom=ax2.get_ylim()[0], top=ax2.get_ylim()[1])
                fig.canvas.draw()
                
                plt.waitforbuttonpress()

if __name__ == "__main__":
    pass
