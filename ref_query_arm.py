from arm import Arm
from model import Model
from torch_cluster import knn_graph, knn
import torch
from sklearn.metrics import confusion_matrix

class RefQueryArm(Arm):
    """implementation of GCN ref to query label propagation arm"""

    def __init__(self, config_file, in_neighbors, out_neighbors, ref_batch_size=None):
        gcn = Model()
        gcn.load(config_file)
        self.in_neighbors = in_neighbors
        self.out_neighbors = out_neighbors
        self.ref_batch_len = ref_batch_size
        super().__init__(gcn)
    
    def train(self, ref_loader, query_loader, epochs, verbose=True, ignore_index_input = -1):
        """train arm"""

        if len(ref_loader) != len(query_loader): raise Exception("Reference and Query Loaders have different amount of batches")

        self.model.to_device(self.device)
        for epoch in range(epochs):
            epoch_loss = 0

            query_iter = iter(query_loader)
            for ref_X, ref_y in ref_loader:
                self.ref_batch_len = len(ref_y)
                
                query_X, query_y = next(query_iter)
                
                query_X, query_y = query_X.to(self.device), query_y.to(self.device)
                ref_X, ref_y = ref_X.to(self.device), ref_y.to(self.device)

                self.opt.zero_grad()

                #combine ref and query
                current = torch.cat([ref_X, query_X], dim=0).float()
                Yt = torch.cat([ref_y, query_y]).long()
                
                # feed data through the model - "forward pass"
                current = self.model(current, self)
            
                # get loss value 
                loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index_input)
                loss = loss_func(current, Yt.squeeze()) # in your training for-loop    

                epoch_loss += loss
                loss.backward()
                self.opt.step()
            
            if verbose and epoch % 10 == 0:
                print("Loss in epoch %d = %f" % (epoch, epoch_loss))

    def predict(self, ref_loader, query_loader):
        """predict with arm"""
        self.model.to_device(self.device)
        ref_preds = []
        query_preds = []
        true_ref = []
        true_query = []
        query_iter = iter(query_loader)
        for ref_X, ref_y in ref_loader:
            self.ref_batch_len = len(ref_y)

            query_X, query_y = next(query_iter)
                
            query_X, query_y = query_X.to(self.device), query_y.to(self.device)
            ref_X, ref_y = ref_X.to(self.device), ref_y.to(self.device)
            
            #save true labels
            true_ref.append(ref_y.long())
            true_query.append(query_y.long())

            #combine ref and query
            current = torch.cat([ref_X, query_X], dim=0).float()

            # feed data through the model - "forward pass"
            current = self.model(current, self)

            ref_preds.append(current[:self.ref_batch_len,:])
            query_preds.append(current[self.ref_batch_len:,:])


        ref_combined = torch.cat(ref_preds, dim=0)
        query_combined = torch.cat(query_preds, dim=0)
        softmax = torch.nn.Softmax(dim=1)
        ref_combined = softmax(ref_combined)
        query_combined = softmax(query_combined)
        
        ref_y = torch.cat(true_ref, dim=0)
        query_y = torch.cat(true_query, dim=0)
        return ref_combined, query_combined, ref_y, query_y

    def validation_metrics(self, ref_loader, query_loader):
        """returns validation metrics (ROC, prC, accuracy, etc)"""
        ref_preds, query_preds, ref_y, query_y = self.predict(ref_loader, query_loader)
        
        """all_labels = []
        iter_query = iter(query_loader)
        for data, ref_y in ref_loader:
            nah, query_y = next(iter_query)
            labels = torch.cat([ref_y, query_y]).long()
            all_labels.append(labels)"""
        
        #real_y = torch.cat(all_labels, dim=0)

        final_pred_ref = ref_preds.max(dim=1)[1]
        final_pred_query = query_preds.max(dim=1)[1]

        ref_equality = (ref_y.cpu() == final_pred_ref.cpu())
        ref_accuracy = ref_equality.type(torch.FloatTensor).mean()
        ref_cm = confusion_matrix(ref_y.cpu(), final_pred_ref.cpu())

        query_equality = (query_y.cpu() == final_pred_query.cpu())
        query_accuracy = query_equality.type(torch.FloatTensor).mean()
        query_cm = confusion_matrix(query_y.cpu(), final_pred_query.cpu())

        return float(ref_accuracy), ref_cm, float(query_accuracy), query_cm

    def validation_metrics_test(self, ref_preds, query_preds, ref_y, query_y):
        """returns validation metrics (ROC, prC, accuracy, etc)"""
        
        """all_labels = []
        iter_query = iter(query_loader)
        for data, ref_y in ref_loader:
            nah, query_y = next(iter_query)
            labels = torch.cat([ref_y, query_y]).long()
            all_labels.append(labels)"""
        
        #real_y = torch.cat(all_labels, dim=0)

        final_pred_ref = ref_preds.max(dim=1)[1]
        final_pred_query = query_preds.max(dim=1)[1]

        print(final_pred_ref)
        ref_equality = (ref_y.cpu() == final_pred_ref.cpu())
        ref_accuracy = ref_equality.type(torch.FloatTensor).mean()
        ref_cm = confusion_matrix(ref_y.cpu(), final_pred_ref.cpu())

        query_equality = (query_y.cpu() == final_pred_query.cpu())
        query_accuracy = query_equality.type(torch.FloatTensor).mean()
        query_cm = confusion_matrix(query_y.cpu(), final_pred_query.cpu())

        return float(ref_accuracy), ref_cm, float(query_accuracy), query_cm

    def construct_graph(self, X):
        """construct graph from data"""
        
        # knn graph with self.ref_neighbors and self.query_neighbors for each ref node
        # don't think we want nodes from query to ref
        # but nodes within query should connect to each other
        
        # split X in middle of rows to get back ref and query
        split = X.split(self.ref_batch_len, dim=0)
        ref = split[0]
        query = split[1]

        # knn graph within ref
        ref_graph = knn_graph(ref, k=self.in_neighbors, loop=True)

        # knn graph within query
        query_graph = knn_graph(query, k=self.in_neighbors, loop=True)
        query_graph = query_graph + self.ref_batch_len

        inter_edges = torch.tensor([])
        if self.out_neighbors != 0:
            # edges from ref to query
            inter_edges = knn(query, ref, k=self.out_neighbors)
            inter_edges = inter_edges + torch.tensor([[0],[self.ref_batch_len]])

        #combine graphs
        final_edges = torch.cat([ref_graph, query_graph, inter_edges], dim=1).long()

        return final_edges