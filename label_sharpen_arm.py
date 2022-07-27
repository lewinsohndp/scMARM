from arm import Arm
from model import Model
from torch_cluster import knn_graph
import torch
from sklearn.metrics import confusion_matrix

class LabelSharpen(Arm):
    """implementation of GCN label propagation arm"""

    def __init__(self, config_file, neighbors):
        gcn = Model()
        gcn.load(config_file)
        self.neighbors = neighbors
        super().__init__(gcn)
    
    def train(self, trainloader, epochs, verbose=True, ignore_index_input = -1):
        """train arm"""
        self.model.to_device(self.device)
        for epoch in range(epochs):
            epoch_loss = 0

            for features, X, y in trainloader:

                features, X, y = features.to(self.device), X.to(self.device), y.to(self.device)

                self.opt.zero_grad()

                # construct graph
                edge_index = self.construct_graph(features)

                # feed data through the model - "forward pass"
                current = X.float()
                current = self.model(current, self, edge_index=edge_index)
                
                Yt = y.long()
            
                # get loss value 
                loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index_input)
                loss = loss_func(current, Yt.squeeze()) # in your training for-loop    

                epoch_loss += loss
                loss.backward()
                self.opt.step()
            
            if verbose and epoch % 10 == 0:
                print("Loss in epoch %d = %f" % (epoch, epoch_loss))

    def predict(self, testloader):
        """predict with arm"""
        self.model.to_device(self.device)
        all_preds = []
        all_labels = []

        for features, X, y in testloader:
            
            features, X, y = features.to(self.device), X.to(self.device), y.to(self.device)

            # construct graph
            edge_index = self.construct_graph(features)

            # feed data through the model - "forward pass"
            current = X.float()
            current = self.model(current, self, edge_index=edge_index)
            all_preds.append(current)
            all_labels.append(y)
        
        combined = torch.cat(all_preds, dim=0)
        softmax = torch.nn.Softmax(dim=1)
        combined = softmax(combined)

        combined_labels = torch.cat(all_labels, dim=0)

        return combined, combined_labels

    def validation_metrics(self, testloader):
        """returns validation metrics (ROC, prC, accuracy, etc)"""
        preds, real_y = self.predict(testloader)
        #all_labels = []
        #for features, X, y in testloader:
        #    all_labels.append(y)
        #real_y = torch.cat(all_labels, dim=0)
        final_pred = preds.max(dim=1)[1]
        equality = (real_y.cpu() == final_pred.cpu())
        accuracy = equality.type(torch.FloatTensor).mean()

        cm = confusion_matrix(real_y.cpu(), final_pred.cpu())
        return float(accuracy), cm

    def construct_graph(self, X):
        """construct graph from data"""
        
        return knn_graph(X, k=self.neighbors, loop=True)