#standard library imports
from database import DataBase
import pandas as pd

#dash imports
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

"""
Variables
"""
DATABASE_PATH = "./databases/ethtest"

"""
Database setup
"""
#create the database
db = DataBase(DATABASE_PATH)

#get the database name
name = DATABASE_PATH.split("/")[-1]

"""
Layout
"""
#title
title = html.Div(id="title-wrapper", children=[html.H1(f"Working on DataBase: {name}")])

#graph
figure = go.Figure()
figure.update_xaxes(rangeslider={"thickness": 0.1, "visible": True, "yaxis":{"rangemode": "auto"}})
figure.update_yaxes(autorange=True)
data = db["5m", ["close_time", "close"]]
figure.add_scatter(x=data["close_time"], y=data["close"])

def zoom(layout, xrange):
    in_view = data.loc[figure.layout.xaxis.range[0]:figure.layout.xaxis.range[1]]
    figure.layout.yaxis.range = [in_view.close.min() - 10, in_view.close.max() + 10]
    print("test")

figure.layout.on_change(zoom, 'xaxis.range')

graph = dcc.Graph(id="graph", figure=figure)
graph_wrapper = html.Div(id="graph-wrapper", children=[graph])



labeling = html.Div(id="labeling-wrapper")

menu = html.Div(id="menu-wrapper")

layout = html.Div(children=[title, graph, labeling, menu])

#create the app
app = dash.Dash(__name__)

#set the layout
app.layout = layout



if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")