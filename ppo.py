from os import close
import gym
from gym.spaces import Discrete, Box

from database import DataBase, TrainDataBase
from hyperparameters import CandlestickInterval as CI
from hyperparameters import Derivation, Scaling, ScalerType
from hyperparameters import HyperParameters

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
        self.data, self.action_log = self._prepare_data()

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
        price = self.action_log.iloc[index, 1]
        
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
        
        
        """
        Reward definition and new Observation
        """
        if self.mode == "sell" and action == 2:
            #define the reward
            reward = local_specific_profit
        else:
            reward = 0

        #update the current_step
        self.current_step += 1 
        
        #if we are not at the end of this episode yet:
        if self.current_step < self.episode_length:            
            #get the new observation
            observation = self.data.iloc[self.start_index + self.current_step:self.start_index + self.current_step + self.hp.window_size]
            #create numpy array from dataframe
            observation = observation.to_numpy()

        #if we reached the end of this episode:
        else:
            #update the doneflag
            self.done = True
            observation = None

        return observation, reward, self.done

    def reset(self):
        #reset the local run variables
        self.current_step = 0
        self.done = False
        self.mode = "buy"
        self.specific_profit = 0

        #get random startingpoint
        self.start_index = random.randrange(0, self.data.shape[0]-self.episode_length-self.hp.window_size + 1, 1)

        #get the start_state
        start_state = self.data.iloc[self.start_index + self.current_step:self.start_index + self.current_step + self.hp.window_size]
        
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


if __name__ == "__main__":
    from hyperparameters import LSTMHyperParameters
    hps = LSTMHyperParameters.load("./experiments/15m/Run5/hyperparameters.json")
    scaler = joblib.load(filename="./experiments/15m/Run5/scaler.joblib")
     
    
    myenv = MarketEnvironment(env_config={
        "database_path": "./databases/ethsmall",
        "hyperparameters": hps,
        "preloaded_scaler": scaler,
        "episode_length": 3,
        "trading_fee": 0.075
    })

    done = False

    obs = myenv.reset()
    
    while not done:
        obs, reward, done = myenv.step(myenv.action_space.sample())
        

