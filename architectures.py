#pytorch imports
import torch
import torch.nn as nn

#file imports
import hyperparameters as hp

class LSTM(nn.Module):

    #set the hyperparametertype for this architecture
    hyperparameter_type = hp.LSTMHyperParameters

    def __init__(self, HP, device):
        super(LSTM, self).__init__()

        
        #check if HP is of type LSTMHyperParameters
        if not isinstance(HP, self.hyperparameter_type):
            raise Exception("Make sure to pass the correct HyperParameter-Type for your model")
        
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
        if HP.activation is hp.Activation.TANH:
            self.activation = nn.Tanh()
        elif HP.activation is hp.Activation.RELU:
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

class MCNN(nn.Module):
    """
    Description:
        Implementation of a Multi-Scale Convolutional Neural Network according to "Cui et al." (arXiv: 1603.06995v4)
    """

    def __init__(self, HP, device=None):
        super(LSTM, self).__init__()

        """
        Device Setup
        """
        #save the device
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        #save the HyperParameters
        self.HP = HP

        #get double precision
        self.double()
        #move to device
        self.to(self.device)

class TCN(nn.Module):
    pass