#imports
from pretrain import Network
import torch
import numpy as np
import pandas as pd
from database import DataBase, TrainDataBase
import datetime
from matplotlib import pyplot as plt
import json

EXPERIMENT_FOLDER = "./experiments/test"
DATABASE_FOLDER = "./databases/testdbbtc"
RUN = 1
EPOCH = 9

#load in experiment info
with open(f"{EXPERIMENT_FOLDER}/Run{RUN}/info.json") as json_file:
    run_info = json.load(json_file)

WINDOW_SIZE = run_info["window_size"]

#load in the model
model = Network(MHP=run_info)
model.load_state_dict(torch.load(f"{EXPERIMENT_FOLDER}/Run{RUN}/Epoch{EPOCH}"))
model.eval()

#load the database
db = DataBase(path=DATABASE_FOLDER)
close = db[run_info["candlestick_interval"], "close"].iloc[1:,:]
close["bsh"] = np.nan
close["hold"] = np.nan
close["buy"] = np.nan
close["sell"] = np.nan

#create a tdb
tdb = TrainDataBase(path=DATABASE_FOLDER, DHP=run_info)
data = tdb.train()

#plot
fig, ax = plt.subplots()
fig.show()

for index, batch in enumerate(data):
    samples = batch[0]
    labels = batch[1]



    for i in range(samples.shape[0]):
        #calculate global index
        ind = samples.shape[0]*index + i

        #get the action
        sample = samples[i].unsqueeze(dim=0)
        prediction = model(sample)
        prediction = torch.softmax(prediction, dim=1)
        action = prediction.argmax(dim=1).item()

        close.iloc[ind+WINDOW_SIZE-1, 1] = action
        price = close.iloc[ind+WINDOW_SIZE-1, 0]
        if action == 0:
            close.iloc[ind+WINDOW_SIZE-1, 2] = price
        elif action == 1:
            close.iloc[ind+WINDOW_SIZE-1, 3] = price
        elif action == 2:
            close.iloc[ind+WINDOW_SIZE-1, 4] = price
        

        #plotting
        window = close.iloc[ind:ind+WINDOW_SIZE]
        ax.cla()
        ax.plot(window["close"])
        ax.plot(window["hold"], marker="o", linestyle="", color="gray")
        ax.plot(window["buy"], marker="o", linestyle="", color="green")
        ax.plot(window["sell"], marker="o", linestyle="", color="red")

        fig.canvas.draw()
        plt.waitforbuttonpress()



    exit()