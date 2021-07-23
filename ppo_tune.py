from os import close
import gym
from gym.spaces import Discrete, Box

import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models import ModelCatalog

import torch
import torch.nn as nn

from database import DataBase, TrainDataBase
import hyperparameters as hp

import numpy as np
import joblib
from sklearn.base import BaseEstimator

import warnings
import random

class MarketEnvironment(gym.Env):
    
    def __init__(self, env_config):
        #save the variables
        self.env_config = env_config
        self.db = DataBase(env_config["database_path"])
        self.hp = env_config["hyperparameters"]
        self.episode_length = env_config["episode_length"]
        self.trading_fee_dez = env_config["trading_fee"]/100
        
        #specify action and observation spaces
        self.action_space = Discrete(3)
        self.observation_space = Box(low=-np.finfo(np.float32).max,
                                     high=np.finfo(np.float32).max,
                                     shape=(self.hp.window_size, len(self.hp.features)))

        #handling of the scaler
        if env_config["preloaded_scaler"] is None:
            self.scaler = None
            warnings.warn("You did not specify a preloaded scaler!")
        elif isinstance(env_config["preloaded_scaler"], BaseEstimator):
            #save preloaded scaler instance
            self.scaler = env_config["preloaded_scaler"]
        elif type(env_config["preloaded_scaler"]) == str:
            #load in the scaler
            self.scaler = joblib.load(filename=env_config["preloaded_scaler"])
        else:
            raise Exception("You have to either specify a scaler_location, scaler_instance or None for config_parameter: scaler")

        #prepare the data
        self.data, self.close_data = self._prepare_data()

        #local episode variables
        self.current_step = 0
        self.start_index = 0
        self.done = True
        self.mode = "buy"
        self.specific_profit = 0

    def step(self, action):
        #check that environment is ready so get stepped in
        if self.done:
            raise Exception("You have to reset the environment before you can step through it")
        
        #check that action is in actionspace
        if not self.action_space.contains(action):
            raise Exception(f"Your chosen action: {action} is not valid!")

        """
        Caculate the specific_profit (reward)
        """
        #calculate the index
        index = self.start_index + self.current_step + self.hp.window_size - 1
        #get the price
        price = self.close_data.iloc[index, 1]
        #set reward to 0
        reward = 0
        
        #buy assets
        if self.mode == "buy" and action == 1:
            #save the buyprice
            self.buy_price = price
            #set mode to buy
            self.mode = "sell"

        #sell assets
        elif self.mode == "sell" and action == 2:
            #calculate profit
            local_specific_profit = price/self.buy_price * (1-self.trading_fee_dez)**2 - 1
            #update episode profit
            self.specific_profit += local_specific_profit
            #set mode to buy
            self.mode = "buy"
            #set reward
            reward = local_specific_profit
        
        
        """
        Get new Observation
        """
        #update the current_step
        self.current_step += 1 
        
        #if we are not at the end of this episode yet:
        if self.current_step < self.episode_length:            
            #get the new observation
            observation = self.data.iloc[self.start_index + self.current_step:self.start_index + self.current_step + self.hp.window_size].to_numpy()

        #if we reached the end of this episode:
        else:
            #update the doneflag
            self.done = True
            observation = self.data.iloc[self.start_index + self.current_step-1:self.start_index + self.current_step-1 + self.hp.window_size].to_numpy()

        return observation, reward, self.done, {}

    def reset(self):
        #reset the local run variables
        self.current_step = 0
        self.done = False
        self.mode = "buy"
        self.specific_profit = 0

        #get random startingpoint
        self.start_index = random.randrange(0, self.data.shape[0]-self.episode_length-self.hp.window_size + 1, 1)

        #get the start_state
        start_state = self.data.iloc[self.start_index + self.current_step:self.start_index + self.current_step + self.hp.window_size].to_numpy()
        
        return start_state

    def _prepare_data(self):
        #select the features
        data = self.db[self.hp.candlestick_interval, self.hp.features]

        #data operations that can be made on the whole dataset
        data, scaler = TrainDataBase._raw_data_prep(data=data,
                                                    derive=self.hp.derivation,
                                                    scaling_method=self.hp.scaling,
                                                    scaler_type=self.hp.scaler_type,
                                                    preloaded_scaler=self.scaler)

        #reset the index
        data.reset_index(inplace=True, drop=True)

        #create the close_data
        close_data = self.db[self.hp.candlestick_interval, ["close_time", "close"]]
        #remove first row
        close_data = close_data.iloc[1:,:]
        #reset index
        close_data.reset_index(inplace=True, drop=True)

        return data, close_data

class LSTMModel(TorchModelV2, nn.Module):

    #set the hyperparametertype for this architecture
    hyperparameter_type = hp.LSTMHyperParameters

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **custom_model_config):
        #call inheritance
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        HP = custom_model_config["hyperparameters"]

        #check if HP is of type LSTMHyperParameters
        if not isinstance(HP, self.hyperparameter_type):
            raise Exception("Make sure to pass the correct HyperParameter-Type for your model")
        
        #save values
        self.feature_size = len(HP.features)
        self.hidden_size = HP.hidden_size
        self.num_layers = HP.num_layers
        self.dropout = HP.dropout

        #create the lstm layer
        self.lstm1 = nn.LSTM(input_size=self.feature_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers, dropout=self.dropout)

        #create the linear layers
        self.linear = nn.Linear(self.hidden_size, 3)

        #create the activation
        if HP.activation is hp.Activation.TANH:
            self.activation = nn.Tanh()
        elif HP.activation is hp.Activation.RELU:
            self.activation = nn.ReLU()

        self.last_activation = nn.Softmax(dim=1)

        #create the valuefuntion layers
        self.vf_lstm1 = nn.LSTM(input_size=self.feature_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers, dropout=self.dropout)
        self.vf_linear = nn.Linear(self.hidden_size, 1)
    
    def forward(self, input_dict, state, seq_lens):
        #get the observation
        self.observation = input_dict["obs"]

        
        #lstm1 layer
        x, _ = self.lstm1(self.observation, self._init_hidden_states(self.observation.shape[0]))

        #activation
        x = self.activation(x)

        #linear layers
        x = x[:,-1,:]

        x = self.linear(x)

        x = self.last_activation(x)

        return x, []

    def _init_hidden_states(self, batch_size):
        #create the intial states
        h_0 = torch.rand(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.rand(self.num_layers, batch_size, self.hidden_size)

        return h_0, c_0

    def value_function(self):
        #lstm1 layer
        x, _ = self.vf_lstm1(self.observation, self._init_hidden_states(self.observation.shape[0]))

        #activation
        x = self.activation(x)

        #linear layers
        x = x[:,-1,:]

        x = self.vf_linear(x)

        x = self.activation(x)

        return torch.squeeze(x, dim=1)

    def import_from_h5(self, h5_file):
        #load in the state dict
        state_dict = torch.load(h5_file, map_location=torch.device("cpu"))["model_state_dict"]
        #load in the statedict
        self.load_state_dict(state_dict, strict=False)

if __name__ == "__main__":
    """
    ToDo:
        - Hyperparameter Search
        - Softmax at end of network? (probability distribution or not?)
        - Design different reward functions (exponetial, ...)
        - Model Based (Have Model predict price actions (pretrain regression network))
        - Off policy learning from (human trading, or from labeled data)
    """
    #get ray started
    ray.init()

    #config for this experiment
    tune_config = {
        "trainer_config": {
            "num_gpus": 1,
            "num_workers": 4,
            "framework": "torch",
            "env": MarketEnvironment,
            "env_config": {
                "database_path": "/Users/fabio/Desktop/project-triton/databases/eth",
                "episode_length": 700,
                "trading_fee": 0.075
            },
            "log_level": "DEBUG",
            "batch_mode": "complete_episodes",
            "model": {
                "custom_model": LSTMModel,
                "custom_model_config": {}
            }
        },

        "pretrained_weights": {
            "path": "/Users/fabio/Desktop/project-triton/experiments/15m2/Run1",
            "epoch": 12
        },

        "train_iterations": 1000
    }

    #trainable function
    def train_function(config):
        #handling pretrained weights
        pretrained_weights = config["pretrained_weights"]
        if pretrained_weights is not None:
            path = pretrained_weights["path"]
            hps = hp.LSTMHyperParameters.load(f"{path}/hyperparameters.json")
            scaler = joblib.load(filename=f"{path}/scaler.joblib")

            #add them to the config
            config["trainer_config"]["env_config"]["hyperparameters"] = hps
            config["trainer_config"]["env_config"]["preloaded_scaler"] = scaler
            config["trainer_config"]["model"]["custom_model_config"]["hyperparameters"] = hps
        else:
            raise Exception("You have to specify a pretrained path!")
        
        #create the trainer
        trainer_config = config["trainer_config"]
        environment = trainer_config["env"]
        trainer = ppo.PPOTrainer(env=environment, config=trainer_config)

        #import pretrained weights
        epoch = pretrained_weights["epoch"]
        trainer.import_model(f"{path}/checkpoint_epoch{epoch}")

        #train the model n times
        n = config["train_iterations"]
        for _ in range(n): 
            result = trainer.train()
            tune.report(result)

    tune.run(run_or_experiment=train_function, config=tune_config, local_dir="./experiments_rl", name="test", resources_per_trial={"gpu": 1})
    

    
    
        

