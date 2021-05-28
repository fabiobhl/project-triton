import dataclasses
from enum import Enum, unique
from dataclasses import dataclass, field, fields
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
class Balancing(str, Enum):
    CRITERION_WEIGHTS = "criterion_weights"
    OVERSAMPLING = "oversampling"
    NONE = "none"

@unique
class Shuffle(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    NONE = "none"


@dataclass(frozen=True)
class HyperParameters:
    #model hyperparameters
    hidden_size: int
    num_layers: int
    lr: float
    epochs: int

    #data hyperparameters
    candlestick_interval: CandlestickInterval
    features: list[int] = field(repr=False)
    derivation: Derivation
    batch_size: int
    window_size: int
    labeling: str
    scaling: Scaling
    test_percentage: float
    balancing: Balancing
    shuffle: Shuffle

    def __post_init__(self):
        #enforce datatypes
        for element in dataclasses.fields(HyperParameters):
            name = element.name
            field_type = element.type
            
            #skip features
            if name == "features":
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

        

if __name__ == "__main__":
    HPS = HyperParameters(
        hidden_size=10,
        num_layers=4,
        lr=1e-4,
        epochs=50,
        candlestick_interval=CandlestickInterval.M5,
        features=["close", "open", "high", "low", "volume"],
        derivation=Derivation.TRUE,
        batch_size=100,
        window_size=400,
        labeling="test",
        scaling=Scaling.GLOBAL,
        test_percentage=0.2,
        balancing=Balancing.OVERSAMPLING,
        shuffle=Shuffle.GLOBAL
    )