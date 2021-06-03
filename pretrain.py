#standard python libraries
import dataclasses
import datetime
import os
import json
import itertools
import gc
import pathlib
import re

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
import hyperparameters as hp
from architectures import LSTM

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
        run_dict = dataclasses.asdict(run)
        del run_dict["features"]
        del run_dict["epochs"]
        del run_dict["test_percentage"]
        run_string = json.dumps(run_dict).replace('"', "").replace(":", "=").replace(" ", "")
        directory = f"{self.path}/Run{self.run_count}{run_string}"
        self.tb = SummaryWriter(log_dir=directory)
        self.tb.add_graph(model, input_to_model=example_data)

        #save the directory
        self.log_directory = f"{self.path}/Run{self.run_count}{run_string}"

    def end_run(self):
        #save the hyperparameters
        metrics = {
            "ZMax Test Accuracy": self.run_best_test_accuracy,
            "ZMax Specific Profit Stability": self.run_best_specific_profit_stability
        }
        HPs = dataclasses.asdict(self.run)
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
    
    def checkpoint(self):
        return {"epoch": self.epoch_count,
                "best_test_accuracy": self.run_best_test_accuracy,
                "best_specific_profit_stability": self.run_best_specific_profit_stability}

    def load_checkpoint(self, checkpoint):
        self.epoch_count = checkpoint["epoch"]
        self.run_best_test_accuracy = checkpoint["best_test_accuracy"]
        self.run_best_specific_profit_stability = checkpoint["best_specific_profit_stability"]

class Experiment():

    def __init__(self, path, HP_space, train_database_path, performanceanalytics_database_path, network, checkpointing=False, device=None, identifier=None, torch_seed=None):
        #save the variables
        self.HP_space = HP_space
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
        self.info["HP_space"] = self.HP_space
        self.info["torch_seed"] = self.torch_seed
        #train data dbid
        dbid_info = dbid(train_database_path)
        self.info["train_data"] = dbid_info.dbid
        self.info["train_database_path"] = train_database_path
        #performance ana data dbid
        dbid_info = dbid(performanceanalytics_database_path)
        self.info["test_data"] = dbid_info.dbid
        self.info["performanceanalytics_database_path"] = performanceanalytics_database_path

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
        runs = []
        values = self.HP_space.values()
        for combination in itertools.product(*values):
            comb_dict = dict(zip(self.HP_space.keys(), combination))
            run = self.network.hyperparameter_type(**comb_dict)
            runs.append(run)

        return runs

    @staticmethod
    def conduct_run(run, experiment_path, train_database_path, performanceanalytics_database_path, network, checkpointing, device, torch_seed, run_count, checkpoint_path=None, checkpoint_epochs=10):
        """
        Preparation
        """
        #create traindatabase
        tdb = TrainDataBase(path=train_database_path, HP=run, device=device, seed=torch_seed)

        #create the pa
        pa = PerformanceAnalytics(path=performanceanalytics_database_path, HP=run, scaler=tdb.scaler, device=device)

        #create network
        model = network(HP=run, device=device)
        
        #create the optimizer
        if run.optimizer is hp.Optimizer.ADAM:
            optimizer = torch.optim.Adam(model.parameters(), lr=run.lr)
        elif run.optimizer is hp.Optimizer.SGD:
            optimizer = torch.optim.SGD(model.parameters(), lr=run.lr)

        #create the criterion (Loss Calculator)
        if run.balancing is hp.Balancing.CRITERION_WEIGHTS:
            #create the weight tensor
            weights = tdb.get_label_count()
            weights = weights / weights.sum()
            weights = 1.0 / weights
            weights = weights / weights.sum()

            #create the the criterion
            criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()

        #create the runmanager and start the run
        example_data = next(tdb.train())[0].data
        runman = RunManager(path=experiment_path, run=run, model=model, example_data=example_data, run_count=run_count)

        #load checkpoint if available
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            runman.load_checkpoint(checkpoint["runman_checkpoint"])

        """
        Epoch Loop
        """
        #get the epoch range
        if checkpoint_path is not None:
            epoch_range = range(runman.epoch_count+1, runman.epoch_count + 1 + checkpoint_epochs)
        else:
            epoch_range = range(run.epochs)

        #epoch loop
        for _ in tqdm(epoch_range, leave=True, desc="Epochs", unit="Epoch"):
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
            number = tdb.train_batches_amount*run.batch_size  #number_of_batches * batch_size
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
            number = tdb.test_batches_amount*run.batch_size  #number_of_batches * batch_size
            runman.log_testing(num_test_samples=number, performance_data=performance)

            #checkpointing
            if checkpointing:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "runman_checkpoint": runman.checkpoint() 
                    }, 
                    f"{runman.log_directory}/checkpoint_epoch{runman.epoch_count}")

        """
        Logging
        """
        #save run parameters
        with open(f"{runman.log_directory}/hyperparameters.json", "w") as info:
            json.dump(dataclasses.asdict(run), info, indent=4)

        #save the scaler
        joblib.dump(value=tdb.scaler, filename=f"{runman.log_directory}/scaler.joblib")

        #end the run in the runmanager
        runman.end_run()

    def start(self):
        for index, run in enumerate(self.runs):
            print(f"Running: {index+1}/{self.runs_amount}", run)
            
            #update the runcount
            self.run_count += 1

            #conduct the run
            self.conduct_run(run=run,
                             experiment_path=self.path,
                             train_database_path=self.train_database_path,
                             performanceanalytics_database_path=self.performanceanalytics_database_path,
                             network=self.network,
                             checkpointing=self.checkpointing,
                             device=self.device,
                             torch_seed=self.torch_seed,
                             run_count=self.run_count)

            #garbage memory
            gc.collect()
            
            print("-"*100)

    @staticmethod
    def continue_run(path, network, device, epochs):
        run_count = int(re.search('Run(\d+)', path).group(1))
        path = pathlib.Path(path)
        experiment_path = path.parent.parent.as_posix()

        #load in the hyperparameters
        HP = hp.HyperParameters.load(f"{path.parent.as_posix()}/hyperparameters.json")
        #load in the info dict
        with open(f"{experiment_path}/info.json", "r") as json_file:
            info = json.load(json_file)

        #continue the run
        Experiment.conduct_run(run=HP,
                               experiment_path=experiment_path,
                               train_database_path=info["train_database_path"],
                               performanceanalytics_database_path=info["performanceanalytics_database_path"],
                               network=network,
                               checkpointing=True,
                               device=device,
                               torch_seed=info["torch_seed"],
                               run_count=run_count,
                               checkpoint_path=path.as_posix(),
                               checkpoint_epochs=epochs)

if __name__ == "__main__":
    HP_space = {
        "hidden_size": [3],
        "num_layers": [2],
        "lr": [0.001],
        "epochs": [20],
        "dropout": [0.2],
        "candlestick_interval": [hp.CandlestickInterval.M15],
        "derivation": [hp.Derivation.TRUE],
        "features": [["close", "open", "high", "low", "volume"]],
        "batch_size": [100],
        "window_size": [10, 25],
        "labeling": ["test2"],
        "scaling": [hp.Scaling.GLOBAL],
        "scaler_type": [hp.ScalerType.STANDARD],
        "test_percentage": [0.2],
        "balancing": [hp.Balancing.OVERSAMPLING],
        "shuffle": [hp.Shuffle.GLOBAL],
        "activation": [hp.Activation.RELU],
        "optimizer": [hp.Optimizer.ADAM]
    }

    exp = Experiment(path="./experiments",
                     HP_space=HP_space,
                     train_database_path="./databases/eth",
                     performanceanalytics_database_path="./databases/ethtest",
                     network=LSTM,
                     device=None,
                     identifier="ts",
                     torch_seed=None,
                     checkpointing=True)
    
    exp.start()