import torch
import json

class BaseActor():

    def __init__(self):
        pass


class NNActor():

    def __init__(self, neural_network, load_path, epoch):
        #load in info dict
        with open(f"{load_path}/info.json") as json_file:
            info = json.load(json_file)

        #load in the pretrained weights
        state_dict = torch.load(f"{load_path}/Epoch{epoch}", map_location=torch.device("cpu"))
        
        #create the neural network
        self.model = neural_network(MHP=info)
        self.model.load_state_dict(state_dict)

        #set it to eval mode
        self.model.eval()

    def get_action(self, state):

        preds = self.model(state)

        action = preds.argmax(dim=1).item()

        return action

if __name__ == "__main__":
    pass