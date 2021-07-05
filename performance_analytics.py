#builtin libraries
from hyperparameters import HyperParameters, Balancing, Shuffle
import joblib
import dataclasses

#external libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm

#external files import
from database import TrainDataBase, PerformanceAnalyticsDataBase, DataBase

#pytorch imports
import torch

#dash imports
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_daq as daq

class PerformanceAnalytics():
    """
    Class for testing a neural network model on a certain interval
    """
    def __init__(self, path, HP, scaler=None, device=None):
        #check that HP are hyperparameters
        if not isinstance(HP, HyperParameters):
            raise Exception("The passed Hyperparameters for this TrainDataBase were not of instance HyperParameters")
        
        #save the variables
        self.HP = dataclasses.replace(HP, test_percentage=0.0, balancing=Balancing.NONE, shuffle=Shuffle.NONE)
        self.path = path

        #save the scaler
        if scaler is not None and type(scaler) == str:
            #load in the scaler
            self.scaler = joblib.load(filename=scaler)
        elif scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = None
            
        #load the database
        self.tdb = TrainDataBase(path=path, HP=self.HP, device=device, scaler=self.scaler)

        #get the device
        if device == None:
            #check if there is a cuda gpu available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def evaluate_model(self, model, trading_fee=0.075, stop_loss=False, max_loss=None):
        """
        Setup
        """
        #set the model evaluation mode
        model.eval()
        #move our network to device
        model.to(self.device)

        #convert percentage to decimal
        trading_fee_dez = trading_fee/100

        """
        Calculations
        """
        #create bsh array (The array, where the predictions are going to be safed)
        bsh = np.empty((self.HP.window_size-1))
        bsh[:] = np.nan
        bsh = torch.as_tensor(bsh).to(self.device)
            
        #getting the predictions
        data = self.tdb.train()
        with tqdm(total=self.tdb.train_batches_amount, desc="Performance", unit="batches", leave=False, colour="magenta") as progressbar:
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
        trading_frame = self.tdb[self.HP.candlestick_interval, ["close_time", "close"]]
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
        trading_frame["max_loss"] = np.nan
        if stop_loss:
            trading_frame["stop_loss_specific_profit"] = np.nan
            trading_frame["stop_loss_specific_profit_accumulated"] = np.nan
        
        #reset index
        trading_frame.reset_index(inplace=True, drop=True)
        
        """
        Profit calculation
        """
        mode = "buy"
        stop_loss_mode = "buy"
        specific_profit = 0
        stop_loss_specific_profit = 0

        trading_frame["hold"] = trading_frame.loc[trading_frame["bsh"]==0, "close"]
        trading_frame["buy"] = trading_frame.loc[trading_frame["bsh"]==1, "close"]
        trading_frame["sell"] = trading_frame.loc[trading_frame["bsh"]==2, "close"]              

        for index, row in trading_frame.iterrows():
            #get the predition
            pred = row["bsh"]

            #get the price
            price = row["close"]

            #do the normal trading
            if mode == "buy" and pred == 1:
                #save the price
                buy_price = price

                #reset furthest distance
                furthest_distance = 0
                #change mode
                mode = 'sell'

            elif mode == "sell" and pred == 2:
                #save the price
                sell_price = price

                #calculate the profit
                local_specific_profit = sell_price/buy_price * (1-trading_fee_dez)**2 - 1
                specific_profit += local_specific_profit

                #add metrics to trading frame
                trading_frame.loc[index,"specific_profit"] = local_specific_profit
                trading_frame.loc[index,"specific_profit_accumulated"] = specific_profit
                trading_frame.loc[index,"max_loss"] = (furthest_distance + buy_price)/buy_price * (1-trading_fee_dez)**2 - 1

                #change mode
                mode = 'buy'

            else:
                #update furthest distance
                if mode == "sell" and price - buy_price < furthest_distance:
                    furthest_distance = price - buy_price
            
            #do the stoploss trading
            if stop_loss:
                if stop_loss_mode == "buy" and pred == 1:
                    #save the price
                    sl_buy_price = price

                    #change mode
                    stop_loss_mode = 'sell'

                elif stop_loss_mode == "sell" and pred == 2:
                    #save the price
                    sl_sell_price = price

                    #calculate the profit
                    local_specific_profit = sl_sell_price/sl_buy_price * (1-trading_fee_dez)**2 - 1
                    stop_loss_specific_profit += local_specific_profit

                    #add metrics to trading frame
                    trading_frame.loc[index,"stop_loss_specific_profit"] = local_specific_profit
                    trading_frame.loc[index,"stop_loss_specific_profit_accumulated"] = stop_loss_specific_profit

                    #change mode
                    stop_loss_mode = 'buy'

                elif stop_loss_mode == "sell" and (price/sl_buy_price * (1-trading_fee_dez)**2 - 1) <= max_loss:
                    #save the price
                    sl_sell_price = price

                    #calculate the profit
                    local_specific_profit = sl_sell_price/sl_buy_price * (1-trading_fee_dez)**2 - 1
                    stop_loss_specific_profit += local_specific_profit

                    #add metrics to trading frame
                    trading_frame.loc[index,"stop_loss_specific_profit"] = local_specific_profit
                    trading_frame.loc[index,"stop_loss_specific_profit_accumulated"] = stop_loss_specific_profit

                    #change mode
                    stop_loss_mode = 'buy'
            


        """
        Calculate and save the metrics
        """
        return_dict = self._create_return_dict(specific_profit=specific_profit, trading_frame=trading_frame)

        return return_dict

    def evaluate_model_slow(self, model, additional_window_size=100, trading_fee=0.075, device="cpu"):        
        #create a padb
        padb = PerformanceAnalyticsDataBase(path=self.path, HP=self.HP, scaler=self.scaler, additional_window_size=additional_window_size)

        #get the device
        if device == None:
            #check if there is a cuda gpu available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        #convert tradingfee to dezimal
        trading_fee_dez = trading_fee/100

        #set the model to eval mode
        model.eval()
        model.to(device)

        #create a trading frame
        trading_frame = padb[self.HP.candlestick_interval, ["close_time", "close"]].copy()
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
                state = padb.get_state(device=device)

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
        amount_of_intervals = (trading_frame.shape[0]-self.HP.window_size)
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
        #ffill specific profit accumulated
        trading_frame["specific_profit_accumulated_ff"] = trading_frame["specific_profit_accumulated"].ffill()
        #create max loss positive
        trading_frame["max_loss_positive"] = np.nan
        trading_frame.loc[trading_frame["specific_profit"] > 0, "max_loss_positive"] = trading_frame.loc[trading_frame["specific_profit"] > 0, "max_loss"]
        #add tradingframe to return dict
        return_dict["trading_frame"] = trading_frame

        return return_dict

    def compare_efficient_slow(self, model):
        result_efficient = self.evaluate_model(model=model)
        result_slow = self.evaluate_model_slow(model=model)

        df_efficient = result_efficient["trading_frame"]
        df_slow = result_slow["trading_frame"]

        plt.plot(df_efficient["close_time"], df_efficient["specific_profit_accumulated_ff"], drawstyle="steps-post")
        plt.plot(df_slow["close_time"], df_slow["specific_profit_accumulated_ff"], drawstyle="steps-post", linestyle="--", color="orange")
        plt.show()

def analyze_model(experiment_path, database_path, architecture, checkpoint):
    """
    Backend
    """
    #get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load in the hyperparameters
    HPS = architecture.hyperparameter_type.load(f"{experiment_path}/hyperparameters.json")
    #load in the state dict
    state_dict = torch.load(f"{experiment_path}/checkpoint_epoch{checkpoint}", map_location=device)["model_state_dict"]
    #load in the scaler
    scaler = joblib.load(f"{experiment_path}/scaler.joblib")
    
    #create the model
    model = architecture(HP=HPS, device=device)
    #load in the statedict
    model.load_state_dict(state_dict)

    #create the pa
    pa = PerformanceAnalytics(path=database_path, HP=HPS, scaler=scaler, device=device)
    #get the pa_data
    pa_data = pa.evaluate_model(model=model)
    trading_frame = pa_data["trading_frame"]

    """
    Figure
    """
    #create the figure
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0)
    #set the theme
    fig.update_layout(template="plotly_dark")
    fig.update_yaxes(zeroline=True)
    fig.update_yaxes(zerolinecolor="#723D46")
    
    #add the price history
    fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["close"], row=1, col=1, name="close price")
    #add the actions
    fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["hold"], mode="markers", marker={"color": "grey"}, row=1, col=1, name="hold")
    fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["buy"], mode="markers", marker={"color": "green"}, row=1, col=1, name="buy")
    fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["sell"], mode="markers", marker={"color": "red"}, row=1, col=1, name="sell")

    #add the accumulated profit
    fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["specific_profit_accumulated_ff"], row=2, col=1, name="specific profit accumulated")

    #add the profit
    fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["specific_profit"], mode="lines", connectgaps=True, row=3, col=1, name="specific profit")
    
    #add the maxloss
    trading_frame["max_loss_positive"]
    fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["max_loss_positive"], mode="lines", connectgaps=True, row=3, col=1, name="max loss positive")

    

    """
    App
    """
    #create the app
    app = dash.Dash(__name__)

    #set the layout
    app.layout = html.Div([
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview"),
            dcc.Tab(label="Empty", value="empty")
        ]),
        html.Div(id="content")
    ])

    
    
    overview_layout = html.Div(id="wrapper", children=[
        html.Div(id="graph", children=[
            dcc.Loading(id="loading", children=[
                dcc.Graph(id="overview-graph", figure=fig, config={"scrollZoom": True, "showAxisDragHandles": True, "responsive": True})
            ])
        ]),
        html.Div(id="settings", children=[
            html.Div(id="general", className="menu-item", children=[
                html.H3("General Settings"),
                dcc.Input(id="trading-fee-input", type="number", value=0.075, placeholder="trading fee")
            ]),
            html.Div(id="stop-loss", className="menu-item", children=[
                html.H3("Stop Loss Settings"),
                daq.BooleanSwitch(id="stop-loss-switch", on=False, color="#9B51E0"),
                dcc.Input(id='stop-loss-input', type="number", value=-0.2, placeholder="max loss")
            ]),
            html.Div(id="update", className="menu-item", children=[
                html.Button("refresh", id="update-button")
            ])
        ])
    ])

    #tab renderer
    @app.callback(
        Output("content", "children"),
        Input("tabs", "value"))
    def render_content(tab):
        if tab == "overview":
            return overview_layout
        elif tab == "empty":
            return html.H1("This page is empty")
        else:
            return html.H1("Something went wrong with your tabs")

    #callback for generating the figure
    @app.callback(
        Output("overview-graph", "figure"),
        [Input("update-button", "n_clicks"),
         State("trading-fee-input", "value"),
         State("stop-loss-switch", "on"),
         State("stop-loss-input", "value")])
    def update_graph(update_button, trading_fee, stoploss_switch, stoploss_value):

        #get the new data
        pa_data = pa.evaluate_model(model=model, trading_fee=trading_fee, stop_loss=stoploss_switch, max_loss=stoploss_value)
        trading_frame = pa_data["trading_frame"]

        #create the figure
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0)
        #set the theme
        fig.update_layout(template="plotly_dark")
        fig.update_yaxes(zeroline=True)
        fig.update_yaxes(zerolinecolor="#723D46")
        
        #add the price history
        fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["close"], row=1, col=1, name="close price")
        #add the actions
        fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["hold"], mode="markers", marker={"color": "grey"}, row=1, col=1, name="hold")
        fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["buy"], mode="markers", marker={"color": "green"}, row=1, col=1, name="buy")
        fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["sell"], mode="markers", marker={"color": "red"}, row=1, col=1, name="sell")

        #add the accumulated profit
        fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["specific_profit_accumulated_ff"], row=2, col=1, name="specific profit accumulated")

        #add the profit
        fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["specific_profit"], mode="lines", connectgaps=True, row=3, col=1, name="specific profit")
        
        #add the maxloss
        fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["max_loss_positive"], mode="lines", connectgaps=True, row=3, col=1, name="max loss positive")

        #add stoploss data if activated
        if stoploss_switch:
            #add the accumulated profit
            fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["stop_loss_specific_profit_accumulated"].ffill(), row=2, col=1, name="sl specific profit accumulated")

            #add the profit
            fig.add_scattergl(x=trading_frame["close_time"], y=trading_frame["stop_loss_specific_profit"], mode="lines", connectgaps=True, row=3, col=1, name="sl specific profit")

            #add the maxloss line
            #fig.add_scattergl(x=trading_frame["close_time"], y=stoploss_value, row=3, col=1, name="stop loss max loss")


        return fig



    app.run_server(debug=True, host="0.0.0.0")


if __name__ == "__main__":
    from architectures import LSTM

    path = "./experiments/Run5"
    
    analyze_model(experiment_path=path, database_path="./databases/ethtest", architecture=LSTM, checkpoint=5)