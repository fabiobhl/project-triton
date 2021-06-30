#pytorch imports
import torch
import torch.nn as nn

#file imports
import hyperparameters as hp

#external libraries imports
import numpy as np

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

    #set the hyperparametertype for this architecture
    hyperparameter_type = hp.MCNNHyperParameters


    def __init__(self, HP, device=None):
        super(MCNN, self).__init__()

        #check if HP is of the correct HyperParameters type
        if not isinstance(HP, self.hyperparameter_type):
            raise Exception("Make sure to pass the correct HyperParameter-Type for your model")

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

        #save important variables
        self.in_channels = len(self.HP.features)

        """
        Local Convolution
        """

        #create local convolution layers
        self.local_convolutions = torch.nn.ModuleList()
        for _ in range(len(self.HP.downsamplig_rates) + len(self.HP.ma_window_sizes) + 1):
            self.local_convolutions.append(torch.nn.Conv1d(in_channels=self.in_channels, out_channels=1, kernel_size=self.HP.local_convolution_size))

        #create local maxpooling layers
        self.local_maxpoolings = torch.nn.ModuleList()
        for pooling_factor in self.HP.pooling_factors:
            self.local_maxpoolings.append(torch.nn.MaxPool1d(kernel_size=round(self.HP.window_size/pooling_factor), ceil_mode=True))

        """
        Full Convolution
        """
        #convolution
        self.full_convolution = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.HP.full_convolution_size)
        self.full_maxpool = torch.nn.MaxPool1d(kernel_size=self.HP.full_convolution_pooling_size)

        #linear layers
        L_in = len(self.local_convolutions) * sum(self.HP.pooling_factors)
        L_conv = self.calc_output_length_conv1d(L_in, self.full_convolution)
        in_features = self.calc_output_length_maxpool1d(L_conv, self.full_maxpool)
        self.linear = torch.nn.Linear(in_features=10, out_features=3)
        

        #get double precision
        #self.double()
        #move to device
        self.to(self.device)


    def forward(self, x):
        """
        Prepare Batch
        """
        x = x.permute(0, 2, 1)

        """
        Transformation Stage
        """
        input_branches = []

        #identity mapping
        input_branches.append(x)

        #downsamplings
        for downsampling_rate in self.HP.downsamplig_rates:
            input_branches.append(torch.nn.functional.interpolate(x, scale_factor=1/downsampling_rate, recompute_scale_factor=False))

        """
        Local Convolution Stage
        """
        local_features = []
        for index, convolution in enumerate(self.local_convolutions):
            convoluted = convolution(input_branches[index])

            for maxpool in self.local_maxpoolings:
                local_features.append(maxpool(convoluted))
        
        #deep concatenation of extracted features
        concatted = torch.cat(local_features, dim=2)

        """
        Full Convolution
        """
        x = self.full_convolution(concatted)
        x = self.full_maxpool(x)

        """
        Fully Connected
        """
        x = self.linear(x)

        return x
    
    def calc_output_length_conv1d(self, input_length, conv1d):
        a = 2 * conv1d.padding[0]
        b = conv1d.dilation[0] * (conv1d.kernel_size[0] - 1)

        return (input_length + a - b -1)/conv1d.stride[0] + 1

    def calc_output_length_maxpool1d(self, input_length, maxpool):
        a = 2 * maxpool.padding
        b = maxpool.dilation * (maxpool.kernel_size - 1)

        return (input_length + a - b -1)/maxpool.stride + 1

if __name__ == "__main__":
    """
    conv = torch.nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, bias=False)
    maxpool = torch.nn.MaxPool1d(kernel_size=2, stride=1)
    tens = torch.randn(1, 7, 3)
    print(tens)
    tens = tens.permute(0, 2, 1)
    print(tens)
    result = conv(tens)
    print(result)
    pooled = maxpool(result)
    print(pooled)
    """

    hps = hp.MCNNHyperParameters(
        candlestick_interval=hp.CandlestickInterval.M15,
        features=["close", "open", "volume"],
        derivation=hp.Derivation.TRUE,
        batch_size=10,
        window_size=200,
        labeling="test2",
        scaling=hp.Scaling.GLOBAL,
        scaler_type=hp.ScalerType.STANDARD,
        test_percentage=0.2,
        balancing=hp.Balancing.OVERSAMPLING,
        shuffle=hp.Shuffle.NONE,
        activation=hp.Activation.RELU,
        optimizer=hp.Optimizer.ADAM,
        downsamplig_rates=[2, 3],
        ma_window_sizes=[],
        local_convolution_size=5,
        pooling_factors=[2,3,5],
        full_convolution_size=10,
        full_convolution_pooling_size=4
    )

    model = MCNN(HP=hps, device="cpu")

    tens = torch.randn(1, hps.window_size, len(hps.features))

    tens = model(tens)

    print(tens)