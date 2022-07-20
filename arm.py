import torch

class Arm:
    """Class representing arm of project"""

    def __init__(self, model):
        self.model = model
        self.opt = torch.optim.Adam([weight for item in self.model.get_layers() for weight in list(item.parameters())],0.0001)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

    def train(self, trainloader, epochs, ignore_index_input = -1):
        """train arm"""
        pass

    def predict(self):
        """predict with arm"""
        pass

    def construct_graph(self, X):
        """construct graph from data"""
        pass

    def validation_metrics(self, preds, y):
        """returns validation metrics (ROC, prC, accuracy, etc)"""
        pass

    def reset(self):
        """resets weights in model"""
        self.model.reset()
        self.opt = torch.optim.Adam([weight for item in self.model.get_layers() for weight in list(item.parameters())],0.0001)
