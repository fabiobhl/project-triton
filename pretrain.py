#standard python libraries
import datetime
import os
import json
import itertools
import collections
import gc

#external libraries
from tqdm import tqdm
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")
import joblib

#pytorch imports
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

#file imports
from database import TrainDataBase, dbid
from performance_analytics import PerformanceAnalytics


#definition of the network
class NetworkStateful(nn.Module):

    def __init__(self, MHP):
        super(NetworkStateful, self).__init__()

        #save values
        self.feature_size = len(MHP["features"])
        self.hidden_size = MHP["hidden_size"]
        self.num_layers = MHP["num_layers"]
        self.batch_size = MHP["batch_size"]

        #create the lstm layer
        self.lstm1 = nn.LSTM(input_size=self.feature_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        
        #create the intial states
        self.hidden = torch.rand(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.double)
        self.states = torch.rand(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.double)

        #create the relu
        self.relu = nn.ReLU()

        #create the linear layers
        self.linear = nn.Linear(self.hidden_size, 3)

        self.double()

    def forward(self, x):
        #lstm1 layer
        x, (hn, cn) = self.lstm1(x, (self.hidden, self.states))

        self.hidden = hn.data
        self.states = cn.data

        #relu
        x = self.relu(x)

        #linear layers
        x = x[:,-1,:]

        x = self.linear(x)

        return x

class Network(nn.Module):

    def __init__(self, MHP):
        super(Network, self).__init__()

        #save values
        self.feature_size = len(MHP["features"])
        self.hidden_size = MHP["hidden_size"]
        self.num_layers = MHP["num_layers"]

        #create the lstm layer
        self.lstm1 = nn.LSTM(input_size=self.feature_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)

        #create the relu
        self.activation = nn.Tanh()

        #create the linear layers
        self.linear = nn.Linear(self.hidden_size, 3)

        self.double()

    def forward(self, x):
        #lstm1 layer
        x, _ = self.lstm1(x, self._init_hidden_states(x.shape[0]))

        #relu
        x = self.activation(x)

        #linear layers
        x = x[:,-1,:]

        x = self.linear(x)

        return x

    def _init_hidden_states(self, batch_size):
        #create the intial states
        h_0 = torch.rand(self.num_layers, batch_size, self.hidden_size, dtype=torch.double)
        c_0 = torch.rand(self.num_layers, batch_size, self.hidden_size, dtype=torch.double)

class RunManager():
    """
    Description:
        Class for logging the results of one run to a tensorboard.
    Arguments:
        -path[string]:      The path to save the tensorboard data to.
    """
    
    def __init__(self, path, run, model, example_data, run_count):
        #save the variables
        self.path = path
        self.run = run

        """
        Epoch Variables
        """
        self.epoch_count = -1
        self.epoch_train_loss = 0
        self.epoch_test_loss = 0
        self.epoch_train_num_correct = 0
        self.epoch_test_num_correct = 0
        self.epoch_test_choices = torch.Tensor()

        """
        Run Variables
        """
        self.run_count = run_count
        self.run_best_test_accuracy = 0
        self.run_best_specific_profit_stability = 0

        #create the tb for the run and save the graph of the network
        self.tb = SummaryWriter(log_dir=f"{self.path}/Run{self.run_count}")
        self.tb.add_graph(model, input_to_model=example_data)

    def end_run(self):
        #save the hyperparameters
        metrics = {
            "ZMax Test Accuracy": self.run_best_test_accuracy,
            "ZMax Specific Profit Stability": self.run_best_specific_profit_stability
        }
        HPs = self.run
        HPs["features"] = str(HPs["features"])
        self.tb.add_hparams(hparam_dict=HPs, metric_dict=metrics)

        #close the tensorboard
        self.tb.close()

    def begin_epoch(self):
        #update epoch count
        self.epoch_count += 1

        #reset the variables
        self.epoch_train_loss = 0
        self.epoch_test_loss = 0
        self.epoch_train_num_correct = 0
        self.epoch_test_num_correct = 0
        self.epoch_test_choices = torch.Tensor()

    def log_training(self, num_train_samples):
        #calculate the metrics
        loss = (self.epoch_train_loss/num_train_samples)*100
        accuracy = (self.epoch_train_num_correct/num_train_samples)*100

        #add the metrics to the tensorboard
        self.tb.add_scalar('Train Loss', loss, self.epoch_count)
        self.tb.add_scalar('Train Accuracy', accuracy, self.epoch_count)

    def track_train_metrics(self, loss, preds, labels):
        #track train loss
        self.epoch_train_loss += loss

        #track train num correct
        self.epoch_train_num_correct += self._get_num_correct(preds, labels)

    def log_testing(self, num_test_samples, performance_data=None, trading_activity_interval=(500,560)): 
        #calculate the metrics
        loss = (self.epoch_test_loss/num_test_samples)*100
        accuracy = (self.epoch_test_num_correct/num_test_samples) * 100
        specific_profit = performance_data["specific_profit"]
        specific_profit_rate = performance_data["specific_profit_rate"]
        specific_profit_stability = performance_data["specific_profit_stability"]

        #update best variables
        self.run_best_test_accuracy = accuracy if (self.run_best_test_accuracy < accuracy) else self.run_best_test_accuracy
        self.run_best_specific_profit_stability = specific_profit_stability if (self.run_best_specific_profit_stability < specific_profit_stability) else self.run_best_specific_profit_stability

        #add the metrics to the tensorboard
        self.tb.add_scalar('Test Loss', loss, self.epoch_count)
        self.tb.add_scalar('Test Accuracy', accuracy, self.epoch_count)
        self.tb.add_histogram("Choices", self.epoch_test_choices, self.epoch_count)

        #add the performance data
        self.tb.add_scalar('Specific Profit', specific_profit, self.epoch_count)
        self.tb.add_scalar('Specific Profit Rate', specific_profit_rate, self.epoch_count)
        self.tb.add_scalar('Specific Profit Stability', specific_profit_stability, self.epoch_count)

        """
        Plots
        """
        #get the interval infos
        interval_info = performance_data["interval_info"]
        #get the trading frame
        trading_frame = performance_data["trading_frame"]
        
        #Specific Profit Figure
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        

        y = trading_frame.loc[:,"specific_profit"].to_numpy().astype(np.double)
        mask = np.isfinite(y)
        x = np.arange(0,len(y),1)
        ax1.plot(x[mask], y[mask])
        
        y = trading_frame.loc[:,"specific_profit_accumulated"].to_numpy().astype(np.double)
        mask = np.isfinite(y)
        x = np.arange(0,len(y),1)
        ax2.plot(x[mask], y[mask], drawstyle="steps")
        ax2.plot(trading_frame["specific_profit_accumulated"], drawstyle="steps", linestyle="--")

        ax1.set_title("Specific Profit", fontsize="7")
        ax2.set_title("Accumulated Specific Profit", fontsize="7")
        ax1.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
        ax1.set_xlim(left=0)
        ax2.set_xlim(left=0)
        fig.tight_layout()
        
        self.tb.add_figure("Specific Profit Graph", fig, self.epoch_count)

        #clear the figure
        fig.clear()
        del(fig)

        #trading Activity Figure
        fig, ax = plt.subplots()

        tas = trading_activity_interval[0]
        tae = trading_activity_interval[1]
        
        ax.plot(trading_frame.loc[tas:tae, "close"], color="black")
        ax.plot(trading_frame.loc[tas:tae, "hold"], marker='o', linestyle="", color="gray", markersize=4)
        ax.plot(trading_frame.loc[tas:tae, "buy"], marker='o', linestyle="", color="green", markersize=4)
        ax.plot(trading_frame.loc[tas:tae, "sell"], marker='o', linestyle="", color="red", markersize=4)

        title = f"Date: {interval_info['date_interval']}, Movement: {interval_info['movement']}"
        ax.set_title(title, fontsize="7")
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        
        self.tb.add_figure("Trading Activity", fig, self.epoch_count)

        #clear the figure
        fig.clear()
        del(fig)

        #close all the figures
        plt.close("all")

    def track_test_metrics(self, loss, preds, labels):
        #track test loss
        self.epoch_test_loss += loss

        #track test num correct
        self.epoch_test_num_correct += self._get_num_correct(preds, labels)

        #track choice distribution
        self.epoch_test_choices = torch.cat((self.epoch_test_choices, preds.argmax(dim=1).to('cpu')), dim=0)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

class Experiment():

    def __init__(self, path, MHP_space, DHP_space, train_database_path, performanceanalytics_database_path, network, checkpointing=False, device=None, identifier=None, torch_seed=None):
        #save the variables
        self.MHP_space = MHP_space
        self.DHP_space = DHP_space
        self.train_database_path = train_database_path
        self.performanceanalytics_database_path = performanceanalytics_database_path
        self.network = network
        self.checkpointing = checkpointing

        #create variables
        self.run_count = 0

        """
        File Setup
        """
        #presave the startdate
        start_date = datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")

        #create the id if not given, check if it already exists
        if identifier == None:
            self.id = start_date
        else:
            self.id = identifier

        if os.path.isdir(f"{path}/{self.id}"):
            raise Exception("Your chosen experiment-identifier already exists in your chosen path")

        #save path
        self.path = f"{path}/{self.id}"
        
        #create the path
        os.makedirs(self.path)

        """
        Device Setup
        """
        #save the device
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Working on {self.device}")

        """
        Seed Setup
        """
        #setup torch-seed
        if torch_seed != None:
            torch.manual_seed(torch_seed)
            self.torch_seed = torch_seed
        else:
            self.torch_seed = torch.initial_seed()
        
        print(f"Working with torch-seed: {torch.initial_seed()}")
        
        """
        Get all runs
        """
        self.runs = self._runs_generator()
        self.runs_amount = len(self.runs)
        print(f"Conducting {self.runs_amount} runs in this experiment")

        """
        Data Logging
        """
        #create the info dict where all the general information is saved
        self.info = {}
        self.info["start_date"] = start_date
        self.info["id"] = self.id
        self.info["device"] = str(self.device)
        self.info["MHP_space"] = self.MHP_space
        self.info["DHP_space"] = self.DHP_space
        self.info["torch_seed"] = self.torch_seed
        #train data dbid
        dbid_info = dbid(train_database_path)
        self.info["train_data"] = dbid_info.dbid
        #performance ana data dbid
        dbid_info = dbid(performanceanalytics_database_path)
        self.info["test_data"] = dbid_info.dbid

        #dump the info dict
        with open(f"{self.path}/info.json", "w") as info:
            json.dump(self.info, info, indent=4)
        
        print("="*100)

    def _runs_generator(self):
        """
        Description:
            Method for generating a list of all possible runs out of the hps. Generates a grid to test out --> Simple Grid Search
        Arguments:
            -none, accesses the hps over the instance (self)
        Return:
            -combinations[list]:   Returns a list containing all the runs. One run is represented by a named tuple.
        """
        #get a list of all hyperparameters
        HPs = list(self.MHP_space.keys()) + list(self.DHP_space.keys())
        #create a named tuple
        run = collections.namedtuple(typename="Run", field_names=HPs)

        #get a list of hp spaces
        values = list(self.MHP_space.values()) + list(self.DHP_space.values())

        #a list to save all runs
        runs = []

        for combination in itertools.product(*values):
            runs.append(dict(run(*combination)._asdict()))

        return runs

    def conduct_run(self, run):
        """
        Preparation
        """
        #update run_count
        self.run_count += 1

        #create traindatabase
        tdb = TrainDataBase(path=self.train_database_path, DHP=run, device=self.device, seed=self.torch_seed)

        #create the pa
        pa = PerformanceAnalytics(path=self.performanceanalytics_database_path, DHP=run, scaler=tdb.scaler, device=self.device)

        #create network
        model = self.network(MHP=run)
        model.to(self.device)

        #create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=run["lr"])

        #create the criterion (Loss Calculator)
        if run["balancing_method"] == "criterion_weights":
            #create the weight tensor
            weights = tdb.get_label_count()
            weights = weights / weights.sum()
            weights = 1.0 / weights
            weights = weights / weights.sum()

            #create the the criterion
            criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        #create the runmanager and start the run
        example_data = next(tdb.train())[0].data
        runman = RunManager(path=self.path, run=run, model=model, example_data=example_data, run_count=self.run_count)

        """
        Epoch Loop
        """
        #epoch loop
        for epoch in tqdm(range(run["epochs"]), leave=True, desc="Epochs", unit="Epoch"):
            #start the epoch
            runman.begin_epoch()

            #get the data
            train_data = tdb.train()
            test_data = tdb.test()
            
            """
            #Training
            """
            #set the network to trainmode
            model.train()

            #train loop
            with tqdm(total=tdb.train_batches_amount, desc="Training", unit="batches", leave=False, colour="green") as progressbar:
                for batch in train_data:
                    #get the the samples and labels
                    samples, labels = batch

                    #zero out the optimizer
                    optimizer.zero_grad()

                    #get the predictions
                    preds = model(samples)

                    #calculate the loss
                    loss = criterion(preds, labels)

                    #update the weights
                    loss.backward()
                    optimizer.step()

                    #track train metrics
                    runman.track_train_metrics(loss=loss.item(), preds=preds.data, labels=labels.data)

                    #update progressbar
                    progressbar.update(1)

            #log the train data
            number = tdb.train_batches_amount*run["batch_size"]  #number_of_batches * batch_size
            runman.log_training(num_train_samples=number)

            """
            #Testing
            """
            #set the network to evalutation mode
            model.eval()
            
            with tqdm(total=tdb.test_batches_amount, desc="Testing", unit="batches", leave=False, colour="blue") as progressbar:
                for batch in test_data:
                    #get the the samples and labels
                    samples, labels = batch

                    #get the predictions
                    preds = model(samples)

                    #calculate the loss
                    loss = criterion(preds, labels)

                    #track test metrics
                    runman.track_test_metrics(loss=loss.item(), preds=preds.data, labels=labels.data)

                    #update progressbar
                    progressbar.update(1)
            
            #pa evaluate the model
            performance = pa.evaluate_model(model=model) 

            #log the train data
            number = tdb.test_batches_amount*run["batch_size"]  #number_of_batches * batch_size
            runman.log_testing(num_test_samples=number, performance_data=performance)

            #checkpointing
            if self.checkpointing:
                torch.save(model.state_dict(), f"{self.path}/Run{self.run_count}/Epoch{epoch}")

        """
        Logging
        """
        #save run parameters
        parameters = run.copy()
        with open(f"{self.path}/Run{self.run_count}/info.json", "w") as info:
            json.dump(parameters, info, indent=4)

        #save the scaler
        joblib.dump(value=tdb.scaler, filename=f"{self.path}/Run{self.run_count}/scaler.joblib")

        #end the run in the runmanager
        runman.end_run()

    def start(self):
        for index, run in enumerate(self.runs):
            #named tuple for printing
            ntuple = collections.namedtuple("Run", run)
            
            print(f"Running: {index+1}/{self.runs_amount}", ntuple(**run))
            
            self.conduct_run(run=run)

            gc.collect()
            
            print("-"*100)

if __name__ == "__main__":
    MHP_total = {
        "hidden_size": [10, 100],
        "num_layers": [2],
        "lr": [0.01],
        "epochs": [10]
    }

    DHP_total = {
        "candlestick_interval": ["5m", "15m"],
        "derived": [True, False],
        "features": [["close", "open", "high", "low", "volume", "trend_macd", "trend_ema_slow", "trend_adx", "momentum_rsi", "momentum_kama"]],
        "batch_size": [50, 100],
        "window_size": [200],
        "labeling_method": ["test", "test2"],
        "scaling_method": ["global"],
        "test_percentage": [0.2],
        "balancing_method": ["criterion_weights", "oversampling", None],
        "shuffle": ["global", "local", None]
    }
    
    MHP_space = {
        "hidden_size": [10],
        "num_layers": [2],
        "lr": [0.01, 1e-4],
        "epochs": [10]
    }

    DHP_space = {
        "candlestick_interval": ["5m"],
        "derived": [True],
        "features": [["close", "open", "high", "low", "volume", "trend_macd", "trend_ema_slow", "trend_adx", "momentum_rsi", "momentum_kama"]],
        "batch_size": [50],
        "window_size": [200],
        "labeling_method": ["test", "test2"],
        "scaling_method": ["global"],
        "test_percentage": [0.2],
        "balancing_method": ["oversampling", "criterion_weights"],
        "shuffle": ["nothing", "global", "local"]
    }

    MHP_space2 = {
        "hidden_size": [3],
        "num_layers": [2],
        "lr": [0.01],
        "epochs": [3]
    }

    DHP_space2 = {
        "candlestick_interval": ["15m"],
        "derived": [True, False],
        "features": [["close", "open", "high", "low", "volume", "trend_macd", "trend_ema_slow", "trend_adx", "momentum_rsi", "momentum_kama"]],
        "batch_size": [50],
        "window_size": [200],
        "labeling_method": ["test"],
        "scaling_method": ["global", None],
        "test_percentage": [0.2],
        "balancing_method": ["criterion_weights", "oversampling", None],
        "shuffle": ["global", "local", None]
    }

    exp = Experiment(path="./experiments",
                     MHP_space=MHP_space2,
                     DHP_space=DHP_space2,
                     train_database_path="./databases/ethtest",
                     performanceanalytics_database_path="./databases/ethtest",
                     network=Network,
                     device=None,
                     identifier="testold",
                     torch_seed=1,
                     checkpointing=True)
    
    exp.start()