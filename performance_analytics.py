#external libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#external files import
from database import TrainDataBase

#pytorch imports
import torch

class PerformanceAnalytics():
    """
    Class for testing a neural network model on a certain interval
    """
    def __init__(self, path, DHP, device=None):
        #save the variables
        self.DHP = DHP.copy()
        self.DHP["test_percentage"] = 0
        
        #load the database
        self.tdb = TrainDataBase(path=path, DHP=self.DHP, device=device)
        self.train_data = np.array(self.tdb.train_data)

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
        trading_fee = trading_fee/100

        """
        Calculations
        """
        #create bsh array (The array, where the predictions are going to be safed)
        bsh = np.empty((self.DHP["window_size"]-1))
        bsh[:] = np.nan
        bsh = torch.as_tensor(bsh).to(self.device)
            
        #getting the predictions
        data = self.tdb.train()
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
        
        #move bsh to cpu and create numpy array
        bsh = bsh.to('cpu').numpy()

        #create a trading frame
        trading_frame = self.tdb[self.DHP["candlestick_interval"], "close"]
        trading_frame["hold"] = np.nan
        trading_frame["buy"] = np.nan
        trading_frame["sell"] = np.nan
        
        #shorten the trading frame to predictions length (some data goes lost due to batching in data prep)
        trading_frame = trading_frame.iloc[0:bsh.shape[0],:]
        trading_frame["bsh"] = bsh
        trading_frame["specific_profit"] = np.nan
        trading_frame["specific_profit_accumulated"] = np.nan

        #reset index
        trading_frame.reset_index(inplace=True, drop=True)
        
        #calculate the profit
        mode = 'buy'
        specific_profit = 0
        
        for i in range(self.DHP["window_size"]-1, trading_frame.shape[0]):
            #get the prediction
            pred = trading_frame.loc[i,"bsh"]

            #save the action
            if pred == 0:
                trading_frame.loc[i, "hold"] = trading_frame.loc[i, "close"]
                action = 'hold'
            elif pred == 1:
                trading_frame.loc[i, "buy"] = trading_frame.loc[i, "close"]
                action = 'buy'
            elif pred == 2:
                trading_frame.loc[i, "sell"] = trading_frame.loc[i, "close"]
                action = 'sell'

            #do the trading
            if mode == 'buy':
                if action == 'buy':
                    tc_buyprice = trading_frame.loc[i, "close"]                                     #tc = tradingcoin
                    """
                    brutto_tcamount = trading_amount/tc_buyprice
                    netto_tcamount = brutto_tcamount - brutto_tcamount*trading_fee
                    """
                    mode = 'sell'

            elif mode == 'sell':
                if action == 'sell':
                    tc_sellprice = trading_frame.loc[i, "close"]
                    """
                    brutto_bcamount = tc_sellprice*netto_tcamount                        #bc = basecoin
                    netto_bcamount = brutto_bcamount - brutto_bcamount*trading_fee
                    localprofit = netto_bcamount-trading_amount
                    """
                    local_specific_profit = (tc_sellprice/tc_buyprice)*(1-trading_fee)*(1-trading_fee)-1
                    specific_profit += local_specific_profit

                    trading_frame.loc[i,"specific_profit"] = local_specific_profit
                    trading_frame.loc[i,"specific_profit_accumulated"] = specific_profit
                    mode = 'buy'

        """
        Calculate and save the metrics
        """
        #create the return_dict
        return_dict = {}

        #specific profit
        return_dict["specific_profit"] = specific_profit
        
        #specific profit rate (specific profit per candlestick_interval)
        amount_of_intervals = (trading_frame.shape[0]-self.DHP["window_size"])
        specific_profit_rate = specific_profit/amount_of_intervals
        return_dict["specific_profit_rate"] = specific_profit_rate

        #specific profit stability
        movement = 2*(trading_frame.iloc[-1,0] - trading_frame.iloc[0,0])/(trading_frame.iloc[-1,0] + trading_frame.iloc[0,0])
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
        return_dict["trading_frame"] = trading_frame

        return return_dict
        
if __name__ == "__main__":
    pass
    """
    from pretrain import Network
    
    DHP = {
            "candlestick_interval": "5m",
            "derived": True,
            "features": ["close", "open", "volume"],
            "batch_size": 10,
            "window_size": 20,
            "labeling_method": "smoothing_extrema_labeling",
            "scaling_method": "none",
            "test_percentage": 0.2
        }
    
    model = Network(MHP={
        "hidden_size": 10,
        "num_layers": 2,
        "lr": 0.01,
        "epochs": 10,
        "features": ["close", "open", "volume"]
    })
    
    pa = PerformanceAnalytics(path="./databases/testdbbtc", DHP=DHP)
    result = pa.evaluate_model(model=model)
    print(result)
    """