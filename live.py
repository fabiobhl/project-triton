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
from binance.enums import *
from matplotlib import pyplot as plt

#dash imports
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

#file imports
from database import LiveDataBase
from actor import NNActor
from utils import read_json, read_config, timer

class Gui():

    def __init__(self, hook):
        #data setup
        self.hook = hook

        #app setup
        self.app = dash.Dash(__name__)

        title = html.Div(id="title", children=[html.H1(f"Trading {self.hook.ldb.symbol}, on the {self.hook.ldb.market_endpoint} market")])
        profit = html.Div(id="profit")
        live_graph = html.Div(id="live-graph-wrapper")
        interval = dcc.Interval(id='interval', interval=1*1000, n_intervals=0)

        self.app.layout = html.Div(children=[title, profit, live_graph, interval])

        @self.app.callback(Output('live-graph-wrapper', 'children'),
                           Input('interval', 'n_intervals'))
        def update_live_graph(n):
            #get the data
            data = self.hook.actionlog.get_data_frame(self.hook.ldb.data.iloc[:-1,:])

            #create the figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data["close_time"], y=data["close"], mode="lines", name="close price", line=dict(color="black")))
            fig.add_trace(go.Scatter(x=data["close_time"], y=data["hold"], mode="markers", name="hold", line=dict(color="gray")))
            fig.add_trace(go.Scatter(x=data["close_time"], y=data["buy"], mode="markers", name="buy", line=dict(color="green")))
            fig.add_trace(go.Scatter(x=data["close_time"], y=data["sell"], mode="markers", name="sell", line=dict(color="red")))
            
            return dcc.Graph(id="live-graph", figure=fig)

        @self.app.callback(Output('profit', 'children'),
                           Input('interval', 'n_intervals'))
        def update_profit(n):
            #get the specific profit
            specific_profit = self.hook.broker.specific_profit
            
            return html.H2(f"Specific Profit since start: {specific_profit}")

    def run(self):
        self.app.run_server(host="0.0.0.0", debug=False, dev_tools_silence_routes_logging=True)

class ActionLog():

    def __init__(self, size=200):
        self.size = size

        #action memory
        self.action = [np.nan]*self.size

        #actual price memory
        self.actual_price = [np.nan]*self.size

    def append(self, action, actual_price):
        #save the action
        if action is None:
            self.action.append(np.nan)
        elif action == 0 or action == 1 or action == 2:
            self.action.append(action)
        else:
            raise Exception(f"Your chosen action {action} is not valid!")

        #save the actual price
        if actual_price is None:
            self.actual_price.append(np.nan)
        else:
            self.actual_price.append(actual_price)

        #cut the first elements off
        self.action.pop(0)
        self.actual_price.pop(0)

    def get_data_frame(self, df):
        data = df[["close_time", "close"]].copy()

        #set the length
        length = data.shape[0]
        if length > self.size:
            length = self.size

        #shorten the data
        data = data.iloc[-length:,:]

        #add the actions
        data["action"] = np.array(self.action[-length:])

        #add the action prices
        data["hold"] = np.nan
        data.loc[data["action"] == 0, "hold"] = data.loc[data["action"] == 0, "close"]
        data["buy"] = np.nan
        data.loc[data["action"] == 1, "buy"] = data.loc[data["action"] == 1, "close"]
        data["sell"] = np.nan
        data.loc[data["action"] == 2, "sell"] = data.loc[data["action"] == 2, "close"]
        
        #add the actual prices
        data["actual_price"] = np.array(self.actual_price[-length:])

        #reset the index
        data.reset_index(inplace=True, drop=True)
        
        return data

class Broker():

    def __init__(self, symbol, testing=True, config_path=None):
        #save/create neccessary variables
        self.symbol = symbol
        self.testing = testing
        self.profit = 0
        self.specific_profit = 0
        self.mode = "buy"

        #load in the config
        self.config = read_config(path=config_path)        

        #create the client
        self.client = Client(api_key=self.config["binance"]["key"], api_secret=self.config["binance"]["secret"])
        
        """
        Testnet:
            self.client = Client(api_key=self.config["binance"]["key_testnet"], api_secret=self.config["binance"]["secret_testnet"])
            
            self.client.API_URL = "https://testnet.binance.vision/api"

            order = self.client.create_order(symbol="ETHUSDT", side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=2)
            print(order)

            print(self.client.get_asset_balance(asset="ETH"))
            print(self.client.get_asset_balance(asset="USDT"))
        """
    
    def _get_current_price(self):
        market_endpoint = self.config["binance"]["market_endpoint"]
        
        if market_endpoint == "spot":
            price_dict = self.client.get_symbol_ticker(symbol=self.symbol)
            price = price_dict["price"]
        elif market_endpoint == "futures":
            price_dict = self.client.futures_symbol_ticker(symbol=self.symbol)
            price = price_dict["price"]
        else:
            raise Exception(f"Your chosen market endpoint: {market_endpoint} is not available, change in config.json")
        
        print(price)

        return float(price)

    def buy(self, amount):
        if self.testing:
            return self._test_buy(amount=amount)
        
        raise Exception("Real buying has not been implemented yet")

        return

    def _test_buy(self, amount):
        if self.mode == "buy":
            #get the current price
            price = self._get_current_price()
            
            #set as buyprice
            self.buy_price = price

            self.mode = "sell"
        else:
            return

    def sell(self):
        if self.testing:
            return self._test_sell()
        
        raise Exception("Real selling has not been implemented yet")

    def _test_sell(self):
        if self.mode == "sell":
            #get the current price
            price = self._get_current_price()
            
            #calculate profit
            specific_profit = price/self.buy_price * (1-0.00075)**2 - 1

            #add to specific profit count
            self.specific_profit += specific_profit

            self.mode = "buy"
        else:
            return

    def trade(self, action, amount):
        if action == 0:
            return
        elif action == 1:
            self.buy(amount=amount)
        elif action == 2:
            self.sell()
        else:
            raise Exception(f"Your chosen action: {action} is not valid")

class Bot():

    def __init__(self, symbol, run_path, actor, config_path=None):    
        #save the variables
        self.symbol = symbol
        self.run_path = run_path
        self.info_path = self.run_path + "/info.json"
        self.config_path = config_path
        
        #config dictionary
        self.config = read_config(path=config_path)

        #info dictionary
        self.info = read_json(path=self.info_path)
        
        #setup the ldb
        self.ldb = LiveDataBase(symbol=self.symbol, run_path=self.run_path, config_path=self.config_path)

        #save the actor
        self.actor = actor

        #setup the actionlog
        self.actionlog = ActionLog(size=100)

        #setup the broker
        self.broker = Broker(symbol=self.symbol, testing=True)

        #setup the gui
        self.gui = Gui(hook=self)

    def update(self):
        start = time.time()

        #setup discord webhooks
        webhook = Webhook.partial(self.config["discord"]["webhook_id"], self.config["discord"]["webhook_token"], adapter=RequestsWebhookAdapter())
        prec_webhook = Webhook.partial(self.config["discord"]["prec_webhook_id"], self.config["discord"]["prec_webhook_token"], adapter=RequestsWebhookAdapter())

        #update our ldb
        try:
            self.ldb.update_data()
        except Exception as e:
            print("Unsuccesfull ldb update resetting and conducting no action!")
            print("Exception: ", e)

            #reset our database
            self.ldb = LiveDataBase(symbol=self.symbol, run_path=self.run_path, config_path=self.config_path)

            #save no action
            self.actionlog.append(action=None, actual_price=None)

            #end the update method
            return

        #get the new state
        state = self.ldb.get_state()

        #get the action for that new state
        action = self.actor.get_action(state)

        #do something with this action
        self.broker.trade(action=action, amount=1000)

        #save the action
        self.actionlog.append(action=action, actual_price=100)
        
        #calculate update duration
        duration = time.time()-start

        print(f"Update took {round(duration,2)} seconds")

    def run(self):
        #startup the gui
        gui_thread = threading.Thread(target=self.gui.run)
        gui_thread.start()
        
        #main loop
        while True:
            #wait for time to get to candlestick_interval
            timer(candlestick_interval=self.info["candlestick_interval"])
            #wait a little time
            time.sleep(2)
            
            #update the coins
            self.update()
        
        gui_thread.join()

if __name__ == "__main__":
    from pretrain import Network

    #load in the actor
    Actor = NNActor(neural_network=Network, load_path="./experiments/testeth2/Run1", epoch=0)

    bot = Bot(symbol="ETHUSDT", run_path="./experiments/testeth2/Run1", actor=Actor)

    bot.run()