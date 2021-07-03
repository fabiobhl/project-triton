#standard library imports
import datetime
import os

#external libraries import
import pandas as pd
import numpy as np
from dateutil.parser import parse
from plotly.graph_objs import layout

#file imports
from database import DataBase
from labeling_methods import LabelingMethods, labelingmethod
from hyperparameters import CandlestickInterval

#dash imports
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_daq as daq
import plotly.graph_objects as go

"""
Variables
"""
DATABASE_PATH = "./databases/ethsmall"



"""
App Setup
"""
#create the app
app = dash.Dash(__name__)

#set the layout
app.layout = html.Div([
    dcc.Tabs(id="tabs", value="inspection", children=[
        dcc.Tab(label="Inspection", value="inspection"),
        dcc.Tab(label="Create", value="create")
    ]),
    html.Div(id="content")
])


"""
Database setup
"""
#create the database
db = DataBase(DATABASE_PATH)
#get the database name
name = DATABASE_PATH.split("/")[-1]
#get the available klines
klines = next(os.walk(DATABASE_PATH))[1]


"""
Inspection
"""
#title
inspection_title = html.Div(id="title-wrapper", children=[html.H1(f"Working on DataBase: {name}")])

#graph
inspection_graph = dcc.Graph(id="inspection-graph", config={"scrollZoom": True, "showAxisDragHandles": True})

#candlestick dropdown
klines = next(os.walk(DATABASE_PATH))[1]
inspection_kline_dropdown = dcc.Dropdown(id='inspection-kline-dropdown', options=[{"label": kline, "value": kline} for kline in klines], value=klines[0])

#labels dropdown
inspection_label_dropdown = dcc.Dropdown(id='inspection-label-dropdown')

inspection_layout = html.Div(children=[inspection_title, inspection_kline_dropdown, inspection_label_dropdown, inspection_graph])


#callback for generating the figure
@app.callback(
    Output("inspection-graph", "figure"),
    [Input("inspection-graph", "figure"),
    Input("inspection-kline-dropdown", "value"),
    Input("inspection-label-dropdown", "value")])
def update_graph(figure, candlestick_interval, label):
    #convert candlestick_interval into CandlestickInterval
    candlestick_interval = CandlestickInterval(candlestick_interval)
    
    #create the figure
    data = db[candlestick_interval, ["close_time", "close"]]
    fig = go.Figure()
    fig.add_scattergl(x=data["close_time"], y=data["close"])
    
    #add range if old figure is available
    if figure is not None:
        fig.update_xaxes(range=figure["layout"]["xaxis"]["range"])
        fig.update_yaxes(range=figure["layout"]["yaxis"]["range"])

    #add labels if labels are selected
    if label is not None:
        #load in the labels
        labels = db[candlestick_interval, "labels", label]
        
        #create the data to plot
        data.loc[labels[label] == 0, "hold"] = data.loc[labels[label] == 0, "close"]
        data.loc[labels[label] == 1, "buy"] = data.loc[labels[label] == 1, "close"]
        data.loc[labels[label] == 2, "sell"] = data.loc[labels[label] == 2, "close"]
        
        #plot
        fig.add_scattergl(x=data["close_time"], y=data["hold"], mode="markers", marker={"color": "grey"})
        fig.add_scattergl(x=data["close_time"], y=data["buy"], mode="markers", marker={"color": "green"})
        fig.add_scattergl(x=data["close_time"], y=data["sell"], mode="markers", marker={"color": "red"})


    return fig

#callback for generating options in label dropdown
@app.callback(
    Output("inspection-label-dropdown", "options"),
    Input("inspection-kline-dropdown", "value"))
def update_labeling_dropdown(candlestick_interval):
    options=[]
    
    path = f"{DATABASE_PATH}/{candlestick_interval}/labels"
    try:
        labels = next(os.walk(path))[1]
        options = [{"label": label, "value": label} for label in labels]
    except:
        pass

    return options



"""
Create Labels
"""

create_layout = html.Div(children=[
    html.Div(id="create-title-wrapper", children=[html.H1(f"Working on DataBase: {name}")]),
    dcc.Dropdown(id='create-kline-dropdown', options=[{"label": kline, "value": kline} for kline in klines], value=klines[0]),
    dcc.Dropdown(id='create-labelingmethod-dropdown', options=[{"label": labelingmethod, "value": labelingmethod} for labelingmethod in LabelingMethods.labeling_methods], value=LabelingMethods.labeling_methods[0]),
    dcc.Graph(id="create-graph", config={"scrollZoom": True, "showAxisDragHandles": True}),
    html.Button(id="refresh-button"),
    html.Div(id="create-method-parameters"),
    html.Button(id="create-save"),
    dcc.Input(id="create-save-name", type="text", placeholder="name"),
    html.Div(id="create-save-confirmation")
])

#callback for generating the figure
@app.callback(
    Output("create-graph", "figure"),
    [Input("create-graph", "figure"),
    Input("create-kline-dropdown", "value"),
    Input("create-labelingmethod-dropdown", "value"),
    Input("refresh-button", "n_clicks"),
    State("window-length", "value"),
    State("poly-order", "value"),
    State("min-order", "value"),
    State("max-order", "value")])
def update_graph(figure, candlestick_interval, labeling_method, n_clicks, window_length, poly_order, min_order, max_order):
    #convert candlestick_interval into CandlestickInterval
    candlestick_interval = CandlestickInterval(candlestick_interval)
    
    #create the figure
    data = db[candlestick_interval, ["close_time", "close"]]
    fig = go.Figure()
    fig.add_scattergl(x=data["close_time"], y=data["close"])
    
    #add range if old figure is available
    if figure is not None:
        fig.update_xaxes(range=figure["layout"]["xaxis"]["range"])
        fig.update_yaxes(range=figure["layout"]["yaxis"]["range"])

    #calculate the labels
    labeler = getattr(LabelingMethods, labeling_method)
    labels = labeler(database_path=DATABASE_PATH, candlestick_interval=candlestick_interval, write=False, window_length=window_length, poly_order=poly_order, min_order=min_order, max_order=max_order)

    #plot the labels
    data.loc[labels == 0, "hold"] = data.loc[labels == 0, "close"]
    data.loc[labels == 1, "buy"] = data.loc[labels == 1, "close"]
    data.loc[labels == 2, "sell"] = data.loc[labels == 2, "close"]
        
    fig.add_scattergl(x=data["close_time"], y=data["hold"], mode="markers", marker={"color": "grey"})
    fig.add_scattergl(x=data["close_time"], y=data["buy"], mode="markers", marker={"color": "green"})
    fig.add_scattergl(x=data["close_time"], y=data["sell"], mode="markers", marker={"color": "red"})


    return fig


#callback for displaying the labeling methods parameters
@app.callback(
    Output("create-method-parameters", "children"),
    Input("create-labelingmethod-dropdown", "value"))
def render_parameters(labeling_method):

    if labeling_method == "smoothing_extrema_labeling":
        layout = html.Div([
            dcc.Input(id="window-length", type="number", placeholder="window length", value=11, min=11),
            dcc.Input(id="poly-order", type="number", placeholder="poly order", value=3, min=1),
            dcc.Input(id="min-order", type="number", placeholder="max order", value=5, min=1),
            dcc.Input(id="max-order", type="number", placeholder="max order", value=5, min=1)
        ])

        return layout
    else:
        return html.H1("your labeling method has not been implemented yet")

@app.callback(
    Output("create-save-confirmation", "children"),
    [Input("create-save", "n_clicks"),
    State("create-save-name", "value"),
    State("create-kline-dropdown", "value"),
    State("create-labelingmethod-dropdown", "value"),
    State("window-length", "value"),
    State("poly-order", "value"),
    State("min-order", "value"),
    State("max-order", "value")])
def save_the_labels(n_clicks, name, candlestick_interval, labeling_method, window_length, poly_order, min_order, max_order):
    #convert candlestick_interval into CandlestickInterval
    candlestick_interval = CandlestickInterval(candlestick_interval)
    
    try:
        #calculate the labels
        labeler = getattr(LabelingMethods, labeling_method)
        labeler(database_path=DATABASE_PATH, candlestick_interval=candlestick_interval, write=True, window_length=window_length, poly_order=poly_order, min_order=min_order, max_order=max_order, name=name)

        return html.H3("Saved!")
    except Exception as e:
        return html.H3(f"{e}")


#tab renderer
@app.callback(
    Output("content", "children"),
    Input("tabs", "value"))
def render_content(tab):
    if tab == "inspection":
        return inspection_layout
    elif tab == "create":
        return create_layout
    else:
        return html.H1("Something went wrong with your tabs")


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")