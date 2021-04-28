#standard libraries
import time
from datetime import datetime
import csv
import os
import json
from concurrent import futures
import threading
import multiprocessing
import math

#external libraries
import numpy as np
import pandas as pd
from discord import Webhook, RequestsWebhookAdapter
from binance.client import Client
from matplotlib import pyplot as plt

#file imports
from database import LiveDataBase
from actor import NNActor

class Bot():

    @staticmethod
    def _read_config(path=None):
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

        #check for completeness
        if len(config["binance"]) != 4:
            raise Exception("Make sure your config file is complete, under section binance something seems to be wrong")
        
        if len(config["discord"]) != 4:
            raise Exception("Make sure your config file is complete, under section discord something seems to be wrong")

        return config

    def _setup(self):
        """
        Method for instantiating all the coin objects
        """
        start = time.time()

        with futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(Coin.create, symbol, self.config) for symbol in self.config["binance"]["symbol_list"]]

        for result in results:
            self.coin_dict[result.result().symbol] = result.result()

        print(f"Setup Duration: {time.time()-start}")

    def __init__(self, symbol, info_path, config_path=None, logging=False):
        """
        Arguments:
            -config_path[string]:   path to the config file, if there is no path specified, it is assumed that the config file and this python file are in the same directory
        """    
        #config dictionary
        self.config = self._read_config(path=config_path)

        #info dictionary
        try:
            with open(info_path, "r") as json_file:
                self.info = json.load(json_file)
        except Exception:
            raise Exception("Your info file is corrupt (wrong syntax, missing values, wrong path, ...)")
        
        #setup the ldb
        self.ldb = LiveDataBase(symbol=symbol, config=self.config, candlestick_interval={self.info["candlestick_interval"]: self.info["window_size"]+100})

        

    def update(self):
        start = time.time()

        #setup discord webhooks
        webhook = Webhook.partial(self.config["discord"]["webhook_id"], self.config["discord"]["webhook_token"], adapter=RequestsWebhookAdapter())
        prec_webhook = Webhook.partial(self.config["discord"]["prec_webhook_id"], self.config["discord"]["prec_webhook_token"], adapter=RequestsWebhookAdapter())

        #csv file 
        csv_frame = []

        coin_list = [symbol for symbol in self.coin_dict.keys() if symbol not in self.unsuccessfull_updates]
        while coin_list:
            #list of successfully updated coins
            succesfull_coins = []

            #update the coins
            with futures.ThreadPoolExecutor() as executor:
                results = [executor.submit(self.coin_dict[symbol].update_data) for symbol in coin_list]
            
            #check for errors in updates
            for result in results:
                try:
                    #check if exception occured, while updating data
                    symbol = result.result()

                    #add to succesfull coins
                    succesfull_coins.append(symbol)

                    #delete coin from coinlist
                    coin_list.remove(symbol)
                except Exception as e:
                    pass

            #collect data on good coins and send discord notifications
            for symbol in succesfull_coins:
                #get values
                action = self.coin_dict[symbol].buy_signal_detection()
                trend_state = self.coin_dict[symbol].trend_state
                symbol_string = symbol
                url = self.coin_dict[symbol].url

                #append to frame
                csv_frame.append([symbol, trend_state, action, url])
                
                #send notification to discord
                if action in "Long" or action in "Short":
                    #create the message
                    message = f"{symbol_string}: {action} \n {url}"
                    #send message
                    webhook.send(message)
                
                elif action in "PrecLong" or action in "PrecShort":
                    #create the message
                    message = f"{symbol_string}: {action}"
                    #send message
                    prec_webhook.send(message)

            #check if timeout has been reached
            if time.time()-start > 60:
                #save the unsuccesfull coins
                self.unsuccessfull_updates += coin_list
                print("Unsuccesfull updates because of timeout:", coin_list)
                break
        
        #write to csv
        df = pd.DataFrame(csv_frame)
        df.to_csv(path_or_buf="./live_data/actions.csv", header=False, index=False)
        
        #calculate update duration
        duration = time.time()-start

        print(f"Update took {duration} seconds")

        #write metadata to json file
        metadata = {
            "duration": duration
        }
        with open("./live_data/metadata.json", "w") as jsonfile:
            json.dump(metadata, jsonfile)

    def _round5(self, number):
        return 5 * math.ceil((number+1)/5)

    @staticmethod
    def _timer():
        #incase the timer got called immediately after a 5 minute
        while datetime.now().minute % 5 == 0:
            pass
        while datetime.now().minute % 5 != 0:
            pass
    
    def _reinitializer(self, attempts=15):
        """
        Method for reinitializing coins that were unable to update
        """
        reinitialized_coins = {}
        for i in range(attempts):
            #reinitialize the coins
            with futures.ThreadPoolExecutor() as executor:
                results = [executor.submit(Coin.create, symbol, self.config) for symbol in self.unsuccessfull_updates]
            
            #check if they were initialized correctly
            for result in results:
                try:
                    coin = result.result()
                    #add to local_coin_dict
                    reinitialized_coins[coin.symbol] = coin
                
                except Exception:
                    time.sleep(1)

        return reinitialized_coins

    def _checker(self):
        """
        Method for checking on the coin objects
        """
        for symbol in self.coin_dict.keys():
            coin = self.coin_dict[symbol]
            
            """
            Check the 5m candles
            """
            if coin.klines_5m.iloc[-1,-2].minute != self._round5(datetime.now().minute):
                print("got a mistake")
                self.unsuccessfull_updates.append(coin.symbol)

            """
            Check the 1h candles
            """
            print(coin.klines_1h)

    def _between_worker(self):
        """
        Function for work that needs to be done between updates
        """
        ret_dict = {}

        ret_dict["reinitializer"] = self._reinitializer()

        return ret_dict

    def _logger(self):
        #create new folder
        minute = self._round5(datetime.now().minute)-5
        new_directory = f"{self.log_path}/coins/{datetime.now().strftime('%d-%m-%y=%H-')}{minute}"
        os.makedirs(new_directory)

        #log all the coins
        for symbol in self.coin_dict:
            self.coin_dict[symbol].klines_5m.to_csv(path_or_buf=f"{new_directory}/{symbol}_5m", index=False)
            self.coin_dict[symbol].klines_1h.to_csv(path_or_buf=f"{new_directory}/{symbol}_1h", index=False)

    def run(self):
        #log initial state
        if self.logging:
            self._logger()

        #main loop
        while True:
            #wait for time to get to 5 minutes
            self._timer()
            
            #update the coins
            self.update()

            #log all the data
            if self.logging:
                self._logger()


            #do tasks between
            print("Cleaning Garbage")
            
            successfull_cleanup = False
            with futures.ProcessPoolExecutor() as executor:
                future = executor.submit(self._between_worker)

                #calculate remaining time
                minutes = self._round5(datetime.now().minute) - 1 - datetime.now().minute
                seconds = datetime.now().second
                remaining_seconds = minutes*60 - seconds
                print(f"We have {remaining_seconds} seconds to clean up")

                try:
                    ret_dict = future.result(timeout=remaining_seconds)
                    successfull_cleanup = True
                except Exception as e:
                    print("Not able to do all the Work between because of timeout")
                    print(e)

            if successfull_cleanup:
                #get results from _reinitialiter and put them in coin_dict
                reinitialized_coins = ret_dict["reinitializer"]
                for symbol in reinitialized_coins.keys():
                    #add symbol to coindict
                    self.coin_dict[symbol] = reinitialized_coins[symbol]
                    #remove from unsuccessfull_updates
                    self.unsuccessfull_updates.remove(symbol)

            print("Done with cleaning")

if __name__ == "__main__":
    from pretrain import Network

    #load in the actor
    actor = NNActor(neural_network=Network, load_path="./experiments/testeth/Run1", epoch=0)

    #create a live database
    ldb = LiveDataBase(symbol="ETHUSDT", config=Bot._read_config(), candlestick_interval={"5m": 100})

    #create the fig
    fig, ax = plt.subplots()
    fig.show()

    #create the action lists
    hold = [np.nan]*500
    buy = [np.nan]*500
    sell = [np.nan]*500

    #mainloop
    while True:
        #wait for update
        Bot._timer()

        #wait to not update too early
        time.sleep(10)

        #update the database
        ldb.update_data(candlestick_interval="5m")

        #get the state
        state = ldb.get_state(candlestick_interval="5m", features=["close", "open", "volume"], derive=True, scaling_method="global", window_size=20, device="cpu")
        
        #get the action
        action = actor.get_action(state=state)
        print(action)

        #update plot
        data = ldb.data["5m"].iloc[:-1]

        hold.append(np.nan)
        buy.append(np.nan)
        sell.append(np.nan)

        if action == 0:
            hold[-1] = data["close"].iloc[-1]
        elif action == 1:
            buy[-1] = data["close"].iloc[-1]
        else:
            sell[-1] = data["close"].iloc[-1]

        window_size = 50
        
        window = pd.DataFrame()
        window["close"] = data["close"].iloc[-window_size:]
        window.reset_index(inplace=True, drop=True)
        window["hold"] = pd.Series(hold[-window_size:])
        window["buy"] = pd.Series(buy[-window_size:])
        window["sell"] = pd.Series(sell[-window_size:])
        window.reset_index(inplace=True, drop=True)

        ax.cla()
        ax.plot(window["close"])
        ax.plot(window["hold"], marker="o", linestyle="", color="gray")
        ax.plot(window["buy"], marker="o", linestyle="", color="green")
        ax.plot(window["sell"], marker="o", linestyle="", color="red")
        
        fig.canvas.draw()
        plt.pause(0.01)