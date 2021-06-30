import dataclasses
from enum import Enum, unique
import json


@unique
class CandlestickInterval(str, Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"

@unique
class Derivation(str, Enum):
    TRUE = "true"
    FALSE = "false"

@unique
class Scaling(str, Enum):
    GLOBAL = "global"
    NONE = "none"

@unique
class ScalerType(str, Enum):
    MAXABS = "maxabs"
    STANDARD = "standard"

@unique
class Balancing(str, Enum):
    CRITERION_WEIGHTS = "criterion_weights"
    OVERSAMPLING = "oversampling"
    NONE = "none"

@unique
class Shuffle(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    NONE = "none"

@unique
class Activation(str, Enum):
    TANH = "tanh"
    RELU = "relu"

@unique
class Optimizer(str, Enum):
    ADAM = "adam"
    SGD = "sgd"

@dataclasses.dataclass(frozen=True)
class HyperParameters:
    def __post_init__(self):
        #enforce datatypes
        for element in dataclasses.fields(self):
            name = element.name
            field_type = element.type

            #skip lists
            if type(self.__dict__[name]) is list:
                continue
            
            if not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                raise TypeError(f"The Hyperparameter `{name}` was assigned with `{current_type}` instead of `{field_type}`")

    def dump(self, path):
        #get dictionary
        dic = dataclasses.asdict(self)

        #dump the dictionary
        with open(path, "w") as hps:
            json.dump(dic, hps, indent=4)

    @classmethod
    def load(cls, path):
        #load the dictionary
        with open(path, "r") as hps:
            dic = json.load(hps)

        #convert strings to enums
        for element in dataclasses.fields(cls):
            name = element.name
            field_type = element.type

            if issubclass(field_type, Enum):
                dic[name] = field_type(dic[name])

        return cls(**dic)


@dataclasses.dataclass(frozen=True)
class DataHyperParameters(HyperParameters):
    candlestick_interval: CandlestickInterval
    features: list[int] = dataclasses.field(repr=False)
    derivation: Derivation
    batch_size: int
    window_size: int
    labeling: str
    scaling: Scaling
    scaler_type: ScalerType
    test_percentage: float
    balancing: Balancing
    shuffle: Shuffle
    activation: Activation
    optimizer: Optimizer

@dataclasses.dataclass(frozen=True)
class LSTMHyperParameters(DataHyperParameters):
    hidden_size: int
    num_layers: int
    lr: float
    epochs: int
    dropout: float

@dataclasses.dataclass(frozen=True)
class MCNNHyperParameters(DataHyperParameters):
        downsamplig_rates: list[int]
        ma_window_sizes: list[int]
        local_convolution_size: int
        pooling_factors: list[int]
        full_convolution_size: int
        full_convolution_pooling_size: int


if __name__ == "__main__":
    pass