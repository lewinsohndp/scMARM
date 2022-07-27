import json
from utilities import EdgeConv
import torch

class Model(torch.nn.Module):
    """class to describe any model added to project"""

    def __init__(self):
        super().__init__()
        self.layers = []
        self.config_file = None
    
    def get_layers(self):
        return self.layers

    def load(self, file_path):
        """loads model from json format"""
        
        self.config_file = file_path
        f = open(file_path)
        data = json.load(f)
        f.close()
        final_layers = []

        for layer in data['layers']:
            if layer['type'] == 'EdgeConv':
                new_layer = EdgeConv(layer['input'],layer['output'])
            
            elif layer['type'] == 'Linear':
                new_layer = torch.nn.Linear(layer['input'], layer['output'])
            
            else:
                raise Exception("Unrecognizable layer type")
            
            final_layers.append(new_layer)
        
        self.layers = final_layers

    def save(self, file_out):
        """saves model in *blank* format"""
        pass

    def forward(self, X, arm, edge_index=None):
        """forward pass through model"""

        for layer in self.layers:
            #if on linear layer, no need to construct graph
            if type(layer) == torch.nn.modules.linear.Linear:
                X = layer(X)
            else:
                if edge_index==None:
                    edge_index = arm.construct_graph(X)
                X = layer(X, edge_index)

        return X

    def reset(self):
        """reset weights"""
        self.load(self.config_file)

    def to_device(self, device):
        """send layers to device"""
        for layer in self.layers:
            layer.to(device)
