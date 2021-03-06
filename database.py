#standard python libraries
import json
import atexit
import datetime
import os
import warnings
import math
import shutil
import joblib

#external libraries
from binance.client import Client
import numpy as np
import pandas as pd
import ta
from sklearn import preprocessing
import torch

#external methods
from utils import read_config, read_json
from hyperparameters import HyperParameters, CandlestickInterval, Derivation, Scaling, Balancing, Shuffle, ScalerType

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
        #make sure that candlestick interval is of type CandlestickInterval
        if type(index) == tuple:
            if not isinstance(index[0], CandlestickInterval):
                raise Exception(f"Make sure your candlestick interval is of type CandlestickInterval and not {type(index[0])}")
        elif not isinstance(index, CandlestickInterval):
            raise Exception(f"Make sure your candlestick interval is of type CandlestickInterval and not {type(index)}")
        
        #set the path
        if type(index) == tuple:
            path = f"{self.path}/{index[0].value}" 
        elif isinstance(index, CandlestickInterval):
            path = f"{self.path}/{index.value}"
        else:
            raise Exception("Your chosen index is not valid")

        #check if path is available
        if not os.path.isdir(path):
            raise Exception("Your chosen kline-interval is not available")

        #access whole dataframe of certain kline-interval
        if isinstance(index, CandlestickInterval):
            #load in the data and return
            try:
                data = pd.read_csv(filepath_or_buffer=f"{path}/{index.value}", index_col="index")
                
                #convert the date columns
                data["close_time"]= pd.to_datetime(data["close_time"])
                data["open_time"]= pd.to_datetime(data["open_time"])
                
                return data

            except:
                raise Exception("Your chosen kline-interval is not available in this DataBase")

        #access all the labels
        elif type(index) == tuple and len(index) == 2 and isinstance(index[0], CandlestickInterval) and index[1] == "labels":
            try:
                #get all the label names
                label_names = next(os.walk(f"{path}/labels"))[1]

                #load in all the labels
                labels = pd.DataFrame()
                for label_name in label_names:
                    df = pd.read_csv(filepath_or_buffer=f"{path}/labels/{label_name}/labels.csv", header=None, index_col=0, names=["index", "labels"])
                    labels[label_name] = df["labels"]
                
                return labels
            except:
                raise Exception("There are no labels in your database")

        #access one label
        elif type(index) == tuple and len(index) == 3 and isinstance(index[0], CandlestickInterval) and index[1] == "labels" and type(index[2]) == str:
            try:
                #load in the labels
                labels = pd.read_csv(filepath_or_buffer=f"{path}/labels/{index[2]}/labels.csv", header=None, index_col=0, names=["index", index[2]])
                return labels
            
            except:
                raise Exception("Your chosen label-type is not available")
        
        #access a list of labels
        elif type(index) == tuple and len(index) == 3 and isinstance(index[0], CandlestickInterval) and index[1] == "labels" and type(index[2]) == list:
            try:
                #load in the labels
                labels = pd.DataFrame()
                for label_name in index[2]:
                    df = pd.read_csv(filepath_or_buffer=f"{path}/labels/{label_name}/labels.csv", header=None, index_col=0, names=["index", label_name])
                    labels[label_name] = df[label_name]

                return labels[index[2]]

            except:
                raise Exception("Your chosen label-type is not available")

        #access one feature of a kline-interval
        elif type(index) == tuple and len(index) == 2 and isinstance(index[0], CandlestickInterval) and type(index[1]) == str:
            try:
                data = pd.read_csv(filepath_or_buffer=f"{path}/{index[0].value}", usecols=[index[1]])
                
                #convert the date columns
                if "close_time" in data.columns:
                    data["close_time"]= pd.to_datetime(data["close_time"])
                if "open_time" in data.columns:
                    data["open_time"]= pd.to_datetime(data["open_time"])

                return data
            
            except:
                raise Exception("Your chosen feature is not available in this DataBase")
            
        #access list of features of a kline-interval
        elif type(index) == tuple and len(index) == 2 and isinstance(index[0], CandlestickInterval) and type(index[1]) == list:
            try:
                data = pd.read_csv(filepath_or_buffer=f"{path}/{index[0].value}", usecols=index[1])
                
                #convert the date columns
                if "close_time" in data.columns:
                    data["close_time"]= pd.to_datetime(data["close_time"])
                if "open_time" in data.columns:
                    data["open_time"]= pd.to_datetime(data["open_time"])

                return data[index[1]]
            
            except:
                raise Exception("One/multiple of your chosen feature/s is/are not available in this DataBase")
        
        #throw error on all other accesses
        else:
            raise Exception("Your index is not possible, please check your index and the documentation on the DataBase object")
    
    @staticmethod
    def _download_kline_interval(symbol, start_date, end_date, candlestick_interval, config_path):   
        #read in the config
        config = read_config(path=config_path)

        #create the client
        client = Client(api_key=config["binance"]["key"], api_secret=config["binance"]["secret"])

        #download the data and safe it in a dataframe
        print(f"Downloading {candlestick_interval} klines...")
        raw_data = client.get_historical_klines(symbol=symbol, interval=candlestick_interval, start_str=start_date, end_str=end_date)
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

        return data

    def add_candlestick_interval(self, candlestick_interval, config_path=None):
        #check if interval already exists
        if os.path.isdir(f"{self.path}/{candlestick_interval.value}"):
            raise Exception("Your chosen candlestick_interval already exists")

        #download interval
        data = self._download_kline_interval(symbol=self.dbid["symbol"], start_date=self.dbid["date_range"][0], end_date=self.dbid["date_range"][1], candlestick_interval=candlestick_interval.value, config_path=config_path)

        #create the directory
        os.mkdir(f"{self.path}/{candlestick_interval.value}")

        #save the data to csv's
        data.to_csv(path_or_buf=f"{self.path}/{candlestick_interval.value}/{candlestick_interval.value}", index_label="index")

        #add candlestick_interval to dbid
        self.dbid["candlestick_interval"].append(candlestick_interval.value)
        self.dbid.dump()

    @classmethod
    def create(cls, save_path, symbol, date_span, candlestick_intervals, config_path=None):
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
        
        #create the directory
        os.mkdir(save_path)

        #get the dates and format them
        startdate = date_span[0].strftime("%d %b, %Y")
        enddate = date_span[1].strftime("%d %b, %Y")

        """
        Download the datas, add the tas and save it to directory
        """
        try:
            for candlestick_interval in candlestick_intervals:
                #download the data
                data = cls._download_kline_interval(symbol=symbol, start_date=startdate, end_date=enddate, candlestick_interval=candlestick_interval.value, config_path=config_path)

                #create the directory
                os.mkdir(f"{save_path}/{candlestick_interval.value}")

                #save the data to csv's
                data.to_csv(path_or_buf=f"{save_path}/{candlestick_interval.value}/{candlestick_interval.value}", index_label="index")
        except Exception as e:
            shutil.rmtree(save_path)
            raise e
        
        print("Finished downloading")
        
        """
        Creating the dbid and saving it
        """
        #create the dbid
        dbid = {
            "symbol": symbol,
            "date_range": (startdate, enddate),
            "candlestick_interval": [candlestick_interval.value for candlestick_interval in candlestick_intervals]
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
    def __init__(self, path, HP, scaler=None, device=None, seed=None):
        #calling the inheritance
        super().__init__(path)

        #safe the hyperparameters
        if not isinstance(HP, HyperParameters):
            raise Exception("The passed Hyperparameters for this TrainDataBase were not of instance HyperParameters")
        self.HP = HP

        #save the seed
        self.seed = seed

        #auto detection for device
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        #save the scaler
        if scaler is not None and type(scaler) == str:
            #load in the scaler
            self.scaler = joblib.load(filename=scaler)
        elif scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = None

        #prepare the data
        self.prepd_data, self.labels, self.fixed_index = self._prepare_data()

        #calculate data variables
        self.windows_amount = self.fixed_index.shape[0]
        self.batches_amount = math.floor(self.windows_amount/self.HP.batch_size)
        self.train_batches_amount = self.batches_amount - math.floor(self.batches_amount*self.HP.test_percentage)
        self.test_batches_amount = self.batches_amount - self.train_batches_amount

    def train(self):
        """
        Description:
            Method for training your nn.
        Arguments:
            -none
        Return:
            -iterable_data[generator]:      Returns a generator on which you can call next() until it is empty
        """
        #get the batchsize
        bs = self.HP.batch_size

        #create the index
        index = 0

        for _ in range(self.train_batches_amount):
            #get the slice
            fixed_index_slice = self.fixed_index[index:index + bs].copy()
            
            #shuffle locally
            if self.HP.shuffle is Shuffle.LOCAL:
                generator = np.random.default_rng(seed=self.seed)
                generator.shuffle(fixed_index_slice)

            #get the batch
            batch, labels = self._get_batch(fixed_index_slice)

            #update indeces
            index += bs

            yield torch.tensor(batch, device=self.device), torch.tensor(labels, dtype=torch.long, device=self.device).squeeze()

    def test(self):
        """
        Description:
            Method for testing your neural network.
        Arguments:
            -none
        Return:
            -iterable_data[generator]:      Returns a generator on which you can call next() until it is empty (Note: Throws error!)
        """
        #get the batchsize
        bs = self.HP.batch_size

        #create the index
        index = self.train_batches_amount*bs

        for _ in range(self.test_batches_amount):
            #get the slice
            fixed_index_slice = self.fixed_index[index:index + bs]

            #shuffle locally
            if self.HP.shuffle is Shuffle.LOCAL:
                generator = np.random.default_rng(seed=self.seed)
                generator.shuffle(fixed_index_slice)

            #get the batch
            batch, labels = self._get_batch(fixed_index_slice)

            #update indeces
            index += bs

            yield torch.tensor(batch, device=self.device), torch.tensor(labels, dtype=torch.long, device=self.device).squeeze()

    @staticmethod
    def _raw_data_prep(data, derive, scaling_method, scaler_type, preloaded_scaler=None):
        
        def derive_data(data):
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
        
        #derive the data
        if derive is Derivation.TRUE:
            data = derive_data(data)
        
        #remove first row
        data = data.iloc[1:,:]
        
        #scale
        scaler = None
        if scaling_method is Scaling.GLOBAL:
            if preloaded_scaler is not None:
                #set the scaler
                scaler = preloaded_scaler
                scaler.copy = False
            else:
                #create the scaler
                if scaler_type is ScalerType.MAXABS:
                    scaler = preprocessing.MaxAbsScaler(copy=False)
                elif scaler_type is ScalerType.STANDARD:
                    scaler = preprocessing.StandardScaler(copy=False)

                #fit scaler to data
                scaler.fit(data)
            
            #fit the data
            scaler.transform(data)
        else:
            warnings.warn("Attention you are not scaling your data at all!")
        
        return data, scaler

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
        labels = self[self.HP.candlestick_interval, "labels", self.HP.labeling]

        #select the features
        data = self[self.HP.candlestick_interval, self.HP.features]

        #data operations that can be made on the whole dataset
        data, scaler = self._raw_data_prep(data=data, derive=self.HP.derivation, scaling_method=self.HP.scaling, scaler_type=self.HP.scaler_type, preloaded_scaler=self.scaler)

        #remove first row from labels (because of derivation)
        labels = labels.iloc[1:,:]

        #reset the index
        data.reset_index(inplace=True, drop=True)

        #create the flat data array
        flat_data = data.to_numpy().flatten()
        
        #create the flat labels array
        labels = labels.to_numpy().flatten()

        #save the scaler parameters
        self.scaler = scaler

        #create the fixed index array
        fixed_index = np.arange(self.HP.window_size-1, data.shape[0])

        #oversampling
        if self.HP.balancing is Balancing.OVERSAMPLING:
            #count the label occurences
            hold_amount = (labels == 0).sum()
            buy_amount = (labels == 1).sum()
            sell_amount = (labels == 2).sum()

            #calculate the oversampling factors
            buy_oversampling = math.floor(hold_amount/buy_amount)
            sell_oversampling = math.floor(hold_amount/sell_amount)

            #create mask for oversampling
            mask = (labels[self.HP.window_size-1:] == 0)*1 + (labels[self.HP.window_size-1:] == 1)*buy_oversampling + (labels[self.HP.window_size-1:] == 2)*sell_oversampling

            #oversample the fixed index
            fixed_index = np.repeat(fixed_index, mask)
        
        #shuffling
        if self.HP.shuffle is Shuffle.GLOBAL:
            #shuffle the fixed_index array
            generator = np.random.default_rng(seed=self.seed)
            generator.shuffle(fixed_index)

        return flat_data, labels, fixed_index

    def _get_batch(self, fixed_index):
        #get the mapper
        mapper = self._create_mapper(fixed_index=fixed_index)

        #get the window
        windows = self.prepd_data[mapper]

        #get the label
        labels = self.labels[fixed_index]

        return windows, labels
    
    def _create_mapper(self, fixed_index):
        #get the window size
        window_length = self.HP.window_size
        #get the number of features
        n_features = len(self.HP.features)

        #the windows we want to get
        index_array = fixed_index.copy()

        #expand the index array
        expanded_index_array = np.expand_dims((index_array+1)*n_features, axis=[1,2])

        #helperwindow for creating the mapper
        helper_window = np.arange(1, n_features*window_length+1)[::-1].reshape(window_length, n_features)*(-1)

        #create the mapper
        mapper = np.full(shape=(index_array.shape[0], window_length, n_features), fill_value=1)
        mapper *= expanded_index_array
        mapper += helper_window

        return mapper

    def get_label_count(self):
        return torch.tensor([(self.labels == 0).sum(), (self.labels == 1).sum(), (self.labels == 2).sum()], dtype=torch.float64)

class PerformanceAnalyticsDataBase(DataBase):

    def __init__(self, path, HP, scaler, additional_window_size=100):
        #calling the inheritance
        super().__init__(path)

        #check that HP are hyperparameters
        if not isinstance(HP, HyperParameters):
            raise Exception("The passed Hyperparameters for this TrainDataBase were not of instance HyperParameters")

        #save variables
        self.path = path
        self.HP = HP
        self.additional_window_size = additional_window_size
        self.iterator = 0

        #load in the dataframe
        self.complete_data = self[self.HP.candlestick_interval, ["open_time", "open", "high", "low", "close", "volume", "close_time"]]

        #defining initial data
        self.data = self.complete_data.iloc[self.iterator: self.iterator + self.HP.window_size+additional_window_size,:]

        #get the length of the undelying data
        self.data_length = self.complete_data.shape[0]

        #save the scaler
        if scaler is not None and type(scaler) == str:
            #load in the scaler
            self.scaler = joblib.load(filename=scaler)
        elif scaler is not None:
            self.scaler = scaler
        else:
            raise Exception("Please provide either a scaler location or a scaler instance")

    def update_data(self): 
        #update the iterator
        self.iterator += 1

        #check for boundary
        if self.iterator + self.HP.window_size+self.additional_window_size >= self.data_length:
            return False 

        #update the data
        self.data = self.complete_data.iloc[self.iterator: self.iterator + self.HP.window_size+self.additional_window_size,:].reset_index(drop=True)

        return True
    
    def get_state(self, device="cpu"):
        #get the data
        data = self.data.copy()

        #add the technical analysis data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            data = ta.add_all_ta_features(data, open='open', high="high", low="low", close="close", volume="volume", fillna=True)
        #select the features
        data = data[self.HP.features]                                                    

        #prep the data (data is now a numpy array)
        data, _ = TrainDataBase._raw_data_prep(data=data, derive=self.HP.derivation, scaling_method=self.HP.scaling, preloaded_scaler=self.scaler, scaler_type=self.HP.scaler_type)

        #get correct size
        data = data.iloc[-self.HP.window_size:, :]

        #convert to pytorch tensor and move to device
        data = torch.tensor(data.to_numpy(), device=device)

        #add the batch dimension
        data = data.unsqueeze(dim=0)
        
        return data

    def get_time(self):
        return self.data["close_time"].iloc[-1]

    def get_price(self):
        return self.data["close"].iloc[-1]

    def get_total_iterations(self):
        return self.data_length - self.HP.window_size - self.additional_window_size

class LiveDataBase():

    def _download(self, candlestick_interval, limit):
        """
        Description:
            Method for downloading a certain amount of timesteps of candlestick data.

        Arguments:
            candlestick_interval[string]:   The candlestick_interval you want to download
                                            (in accordance to the Binance API)
            limit[integer]:                 The amount of timesteps you want to download

        Return:
            data[pd.DataFrame]:             The data in a pandas DataFrame with cleaned columns
        """
        
        #download raw data
        if self.market_endpoint == "futures":
            raw_data = self.client.futures_klines(symbol=self.symbol, interval=candlestick_interval, limit=limit)
        elif self.market_endpoint == "spot":
            raw_data = self.client.get_klines(symbol=self.symbol, interval=candlestick_interval, limit=limit)
        else:
            raise Exception("Please choose a valid market_endpoint in the config file. You can choose from: [futures, spot]")
        
        #create df
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

        return data

    def _setup(self):
        #download the data
        data = self._download(candlestick_interval=self.candlestick_interval, limit=self.info["window_size"]+100)

        #safety reset the index
        data.reset_index(inplace=True, drop=True)

        return data

    def __init__(self, symbol, run_path, config_path=None):
        #save the call time
        self.init_call_time = datetime.datetime.now()
        
        #save the paths
        self.config_path = config_path
        self.run_path = run_path
        self.info_path = run_path + "/info.json"

        #read in the config file
        self.config = read_config(path=self.config_path)

        #read in the info file
        self.info = read_json(path=self.info_path)
        #load in the scaler
        self.scaler = joblib.load(filename=f"{self.run_path}/scaler.joblib")

        #svve variables
        self.symbol = symbol
        self.market_endpoint = self.config["binance"]["market_endpoint"]
        self.candlestick_interval = self.info["candlestick_interval"]

        #create client for interacting with binance
        self.client = Client(api_key=self.config["binance"]["key"], api_secret=self.config["binance"]["secret"])
                
        #download the initial data
        self.data = self._setup()
        
    def update_data(self):
        """
        Description:
            Method for updating our data
        """
            
        #save old values for checking the update
        old_lasttime = self.data.iloc[-1,0]
        old_shape = self.data.shape

        #download data
        new_klines = self._download(candlestick_interval=self.candlestick_interval, limit=2)

        #check if data is full
        if new_klines.shape != (2,7):
            raise Exception("Downloaded data not complete")

        """
        Update the dataframe
        """
        #replace last item
        self.data.iloc[-1,:] = new_klines.iloc[0,:]
        #add new item
        self.data = self.data.append(other=new_klines.iloc[1,:], ignore_index=True)
        #remove first item
        self.data.drop(index=0,axis=0,inplace=True)
        #reset index
        self.data.reset_index(inplace=True, drop=True)

        """
        Check if update was successfull
        """
        #shape check
        if self.data.shape != old_shape:
            raise Exception("Something went wrong with your dataupdate, the dataframe did not remain its shape")

        #lasttime check
        if self.data.iloc[-1,0] == old_lasttime:
            raise Exception("Something went wrong with your dataupdate, the last time did not get updated")

        #equidistance check
        diff = self.data["open_time"].diff().iloc[1:]
        count = diff != pd.Timedelta(self.candlestick_interval)
        count = count.sum()
        
        if count > 0:
            raise Exception("Something went wrong with your dataupdate, the rows are not equidistant")

    def get_state(self, device="cpu"):
        #get the data
        data = self.data.copy()

        #remove last row
        data = data.iloc[:-1,:]

        #add the technical analysis data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            data = ta.add_all_ta_features(data, open='open', high="high", low="low", close="close", volume="volume", fillna=True)

        #select the features
        data = data[self.info["features"]]

        #prep the data (data is now a numpy array)
        data, _ = TrainDataBase._raw_data_prep(data=data, derive=self.info["derived"], scaling_method=self.info["scaling_method"], preloaded_scaler=self.scaler)

        #get correct size
        data = data[-self.info["window_size"]:, :]

        #convert to pytorch tensor and move to device
        data = torch.tensor(data, device=device)

        #add the batch dimension
        data = data.unsqueeze(dim=0)
        
        return data

    @classmethod
    def create(cls, symbol, info_path, config_path=None):
        instance = cls(symbol=symbol, info_path=info_path, config_path=config_path)
        return instance

if __name__ == "__main__":
    DataBase.create("./databases/ethsmall", symbol="ETHUSDT", date_span=(datetime.date(2021, 6, 5), datetime.date(2021, 6, 10)), candlestick_intervals=[CandlestickInterval.M15])
