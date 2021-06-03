#pytorch imports
import torch
import torch.nn as nn

#file imports
from hyperparameters import HyperParameters, CandlestickInterval, Derivation, ScalerType, Scaling, Balancing, Shuffle, Activation, Optimizer

class LSTM(nn.Module):

    def __init__(self, HP, device):
        super(LSTM, self).__init__()

        #save values
        self.feature_size = len(HP.features)
        self.hidden_size = HP.hidden_size
        self.num_layers = HP.num_layers
        self.dropout = HP.dropout

        #save the device
        self.device = device

        #create the lstm layer
        self.lstm1 = nn.LSTM(input_size=self.feature_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers, dropout=self.dropout)

        #create the activation
        if HP.activation is Activation.TANH:
            self.activation = nn.Tanh()
        elif HP.activation is Activation.RELU:
            self.activation = nn.ReLU()

        #create the linear layers
        self.linear = nn.Linear(self.hidden_size, 3)

        #get double precision
        self.double()

        #move to device
        self.to(self.device)

    def forward(self, x):
        #lstm1 layer
        x, _ = self.lstm1(x, self._init_hidden_states(x.shape[0]))

        #activation
        x = self.activation(x)

        #linear layers
        x = x[:,-1,:]

        x = self.linear(x)

        return x

    def _init_hidden_states(self, batch_size):
        #create the intial states
        h_0 = torch.rand(self.num_layers, batch_size, self.hidden_size, dtype=torch.double, device=self.device)
        c_0 = torch.rand(self.num_layers, batch_size, self.hidden_size, dtype=torch.double, device=self.device)

        return h_0, c_0
