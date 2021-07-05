from database import DataBase
import pandas as pd
import numpy as np
import hyperparameters
from plotly.subplots import make_subplots

#get the data
db = DataBase("./databases/ethtest")
data = db[hyperparameters.CandlestickInterval.M15, ["close_time", "close"]]

#create the derived data
data["close_derived"] = data["close"].pct_change()
data["close_derived_ma"] = data["close_derived"].rolling(30).mean()

#detect the dips
data.loc[data["close_derived_ma"] > 0,"dips"] = np.nan
data.loc[data["close_derived_ma"] <= 0,"dips"] = data.loc[data["close_derived_ma"] <= 0, "close_derived_ma"]
count = 0
for index, row in data.iterrows():
    if not pd.isnull(row["dips"]):
        count += 1
    
    elif count != 0 and count < 25:
        data.loc[index-count:index-1, "dips"] = np.nan
        count = 0

    else:
        count = 0

#create the labels
data["labels"] = 1
data.loc[pd.isna(data["dips"]),"labels"] = 0

data.loc[data["labels"] == 0, "normal"] = data.loc[data["labels"] == 0, "close"]
data.loc[data["labels"] == 1, "down"] = data.loc[data["labels"] == 1, "close"]


#create the figure
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0)
#set the theme
fig.update_layout(template="plotly_dark")
fig.update_yaxes(zeroline=True)
fig.update_yaxes(zerolinecolor="#723D46")

#add the price history
fig.add_scattergl(x=data["close_time"], y=data["close"], row=1, col=1, name="close price")
fig.add_scattergl(x=data["close_time"], y=data["normal"], mode="markers", marker={"color": "grey"}, row=1, col=1, name="normal")
fig.add_scattergl(x=data["close_time"], y=data["down"], mode="markers", marker={"color": "red"}, row=1, col=1, name="down")
#add the actions
fig.add_scattergl(x=data["close_time"], y=data["close_derived"], row=2, col=1, name="close price derived")
fig.add_scattergl(x=data["close_time"], y=data["close_derived_ma"], row=2, col=1, name="close price derived ma")
fig.add_scattergl(x=data["close_time"], y=data["dips"], row=2, col=1, name="close price derived ma")

fig.show()
