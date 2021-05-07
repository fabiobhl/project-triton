#external libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm

#external files import
from database import TrainDataBase, PerformanceAnalyticsDataBase

#pytorch imports
import torch

"""
    ToDo:
"""

class PerformanceAnalytics():
    """
    Class for testing a neural network model on a certain interval
    """
    def __init__(self, path, DHP, scaler=None, device=None):
        #save the variables
        self.DHP = DHP.copy()
        self.DHP["test_percentage"] = 0
        self.path = path

        #save the scaler
        if scaler is not None and type(scaler) == str:
            #load in the scaler
            self.scaler = joblib.load(filename=scaler)
        elif scaler is not None and isinstance(scaler, preprocessing.MaxAbsScaler):
            self.scaler = scaler
        else:
            self.scaler = None
            
        #load the database
        self.tdb = TrainDataBase(path=path, DHP=self.DHP, device=device, scaler=scaler)

        #get the device
        if device == None:
            #check if there is a cuda gpu available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def evaluate_model(self, model, trading_fee=0.075):
        """
        Setup
        """
        #set the model evaluation mode
        model.eval()
        #move our network to device
        model = model.to(self.device)

        #convert percentage to decimal
        #trading_fee = trading_fee/100
        trading_fee_dez = trading_fee/100

        """
        Calculations
        """
        #create bsh array (The array, where the predictions are going to be safed)
        bsh = np.empty((self.DHP["window_size"]-1))
        bsh[:] = np.nan
        bsh = torch.as_tensor(bsh).to(self.device)
            
        #getting the predictions
        data = self.tdb.train()
        with tqdm(total=self.tdb.train_data.shape[0], desc="Performance", unit="batches", leave=False, colour="magenta") as progressbar:
            for batch in data:
                #get the samples
                samples = batch[0]

                #feed data to nn
                prediction = model(samples)
                
                #calculate choice
                prediction = torch.softmax(prediction, dim=1)
                pred = prediction.argmax(dim=1)
                
                #append to bsh array
                bsh = torch.cat((bsh, pred),0)

                #update the progressbar
                progressbar.update(1)
        
        #move bsh to cpu and create numpy array
        bsh = bsh.to('cpu').numpy()

        #create a trading frame
        trading_frame = self.tdb[self.DHP["candlestick_interval"], ["close_time", "close"]]
        trading_frame["hold"] = np.nan
        trading_frame["buy"] = np.nan
        trading_frame["sell"] = np.nan
        
        #cut first row (derivation lost)
        trading_frame = trading_frame.iloc[1:,:]

        #shorten the trading frame to predictions length (some data goes lost due to batching in data prep)
        trading_frame = trading_frame.iloc[0:bsh.shape[0],:]
        trading_frame["bsh"] = bsh
        trading_frame["specific_profit"] = np.nan
        trading_frame["specific_profit_accumulated"] = np.nan
        
        #reset index
        trading_frame.reset_index(inplace=True, drop=True)
        
        """
        Profit calculation
        """
        #calculate the profit
        mode = "buy"
        specific_profit = 0

        trading_frame["hold"] = trading_frame.loc[trading_frame["bsh"]==0, "close"]
        trading_frame["buy"] = trading_frame.loc[trading_frame["bsh"]==1, "close"]
        trading_frame["sell"] = trading_frame.loc[trading_frame["bsh"]==2, "close"]              

        for index, row in trading_frame.iterrows():
            #get the predition
            pred = row["bsh"]

            #get the price
            price = row["close"]

            #do the trading
            if mode == "buy":
                if pred == 1:
                    #save the price
                    buy_price = price
                    #change mode
                    mode = 'sell'

            elif mode == "sell":
                if pred == 2:
                    #save the price
                    sell_price = price

                    #calculate the profit
                    local_specific_profit = sell_price/buy_price * (1-trading_fee_dez)**2 - 1
                    specific_profit += local_specific_profit

                    #add metrics to trading frame
                    trading_frame.loc[index,"specific_profit"] = local_specific_profit
                    trading_frame.loc[index,"specific_profit_accumulated"] = specific_profit

                    #change mode
                    mode = 'buy'

        """
        Calculate and save the metrics
        """
        return_dict = self._create_return_dict(specific_profit=specific_profit, trading_frame=trading_frame)

        return return_dict

    def evaluate_model_slow(self, model, additional_window_size=100, trading_fee=0.075):        
        #create a padb
        padb = PerformanceAnalyticsDataBase(database_path=self.path, HP=self.DHP, scaler=self.scaler, additional_window_size=additional_window_size)
        
        #convet tradingfee to dezimal
        trading_fee_dez = trading_fee/100

        #set the model to eval mode
        model.eval()
        model.to(torch.device("cpu"))

        #create a trading frame
        trading_frame = padb[self.DHP["candlestick_interval"], ["close_time", "close"]].copy()
        trading_frame["hold"] = np.nan
        trading_frame["buy"] = np.nan
        trading_frame["sell"] = np.nan
        trading_frame["bsh"] = np.nan
        trading_frame["specific_profit"] = np.nan
        trading_frame["specific_profit_accumulated"] = np.nan
        
        #cut first row (derivation lost)
        trading_frame = trading_frame.iloc[1:,:]

        #reset index
        trading_frame.reset_index(inplace=True, drop=True)

        #set the variables
        specific_profit = 0
        mode = "buy"

        #mainloop
        with tqdm(total=padb.get_total_iterations(), desc="Performance", unit="steps", leave=False, colour="red") as progressbar:
            while padb.update_data():
                
                #get the state
                state = padb.get_state()

                #get the action
                pred = model(state)
                pred = torch.softmax(pred, dim=1)
                action = pred.argmax(dim=1).item()

                #get the index
                index = trading_frame[trading_frame["close_time"] == padb.get_time()].index

                #add action to bsh
                trading_frame.loc[index,"bsh"] = action

                #calculate profit
                if mode == "buy":
                    if action == 1:
                        buy_price = padb.get_price()
                        
                        mode = "sell"
                
                elif mode == "sell":
                    if action == 2:
                        sell_price = padb.get_price()
                        
                        sp = sell_price/buy_price * (1-trading_fee_dez)**2 - 1
                        specific_profit += sp
                        
                        #add metrics to trading frame
                        trading_frame.loc[index,"specific_profit"] = sp
                        trading_frame.loc[index,"specific_profit_accumulated"] = specific_profit
                        
                        mode = "buy"
                
                #update progressbar
                progressbar.update(1)

        #fill hold, buy, sell columns
        trading_frame["hold"] = trading_frame.loc[trading_frame["bsh"]==0, "close"]
        trading_frame["buy"] = trading_frame.loc[trading_frame["bsh"]==1, "close"]
        trading_frame["sell"] = trading_frame.loc[trading_frame["bsh"]==2, "close"]

        #create the return dict
        return_dict = self._create_return_dict(specific_profit=specific_profit, trading_frame=trading_frame)

        return return_dict

    def _create_return_dict(self, specific_profit, trading_frame):
        #create the return_dict
        return_dict = {}

        #specific profit
        return_dict["specific_profit"] = specific_profit
        
        #specific profit rate (specific profit per candlestick_interval)
        amount_of_intervals = (trading_frame.shape[0]-self.DHP["window_size"])
        specific_profit_rate = specific_profit/amount_of_intervals
        return_dict["specific_profit_rate"] = specific_profit_rate

        #specific profit stability
        movement = 2*(trading_frame.iloc[-1,1] - trading_frame.iloc[0,1])/(trading_frame.iloc[-1,1] + trading_frame.iloc[0,1])
        specific_profit_stability = specific_profit_rate/(1+movement)
        return_dict["specific_profit_stability"] = specific_profit_stability

        #interval infos
        interval_info = {
            "movement": round(movement*100,2),
            "duration": round(amount_of_intervals,1),
            "date_interval": f"{self.tdb.dbid['date_range'][0]} -- {self.tdb.dbid['date_range'][1]}"
        }
        return_dict["interval_info"] = interval_info

        #trading_frame
        trading_frame["specific_profit_accumulated_ff"] = trading_frame["specific_profit_accumulated"].ffill()
        return_dict["trading_frame"] = trading_frame

        return return_dict

    def compare_efficient_slow(self, model, trading_fee=0.075):
        result_efficient = self.evaluate_model(model=model)
        result_slow = self.evaluate_model_slow(model=model)

        df_efficient = result_efficient["trading_frame"]
        df_slow = result_slow["trading_frame"]

        plt.plot(df_efficient["close_time"], df_efficient["specific_profit_accumulated_ff"], drawstyle="steps-post")
        plt.plot(df_slow["close_time"], df_slow["specific_profit_accumulated_ff"], drawstyle="steps-post", linestyle="--", color="orange")
        plt.show()


if __name__ == "__main__":
    from pretrain import Network
    
    HPS = {
        "candlestick_interval": "5m",
        "derived": True,
        "features": ["close", "open", "volume"],
        "batch_size": 100,
        "window_size": 20,
        "labeling_method": "smoothing_extrema_labeling",
        "scaling_method": "global",
        "test_percentage": 0.2,
        "balancing_method": "criterion_weights"
    }
    
    model = Network(MHP={
        "hidden_size": 10,
        "num_layers": 2,
        "lr": 0.01,
        "epochs": 10,
        "features": ["close", "open", "volume"]
    })

    #load in the pretrained weights
    state_dict = torch.load("./experiments/testeth2/Run1/Epoch1", map_location=torch.device("cpu"))
    
    #create the neural network
    model.load_state_dict(state_dict)
    
    pa = PerformanceAnalytics(path="./databases/ethmai", DHP=HPS, device="cpu")
    result = pa.evaluate_model(model=model)
