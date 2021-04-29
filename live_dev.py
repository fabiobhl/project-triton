#dash imports
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

import pandas as pd
import numpy as np

from database import LiveDataBase

ldb = LiveDataBase(symbol="ETHUSDT", info_path="./experiments/testeth/Run1/info.json")

"""
App Layout
"""
app = dash.Dash(__name__)
title = html.Div(id="title", children=[html.H1(f"Trading {ldb.symbol}, on the {ldb.market_endpoint} market")])
live_graph = html.Div(id="live-graph-wrapper")
interval = dcc.Interval(id='interval', interval=1*1000, n_intervals=0)

app.layout = html.Div(children=[title, live_graph, interval])

"""
App Callbacks
"""
@app.callback(Output('live-graph-wrapper', 'children'),
              Input('interval', 'n_intervals'))
def update_live_graph(n):
    fig = px.line(data_frame=ldb.data.iloc[:-1,:], x="close_time", y="close")
    return dcc.Graph(id="live-graph", figure=fig)


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")