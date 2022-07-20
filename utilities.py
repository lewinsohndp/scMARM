import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, SiLU
import anndata as ad
import scanpy as sc
from sklearn.decomposition import PCA
import numpy as np
import random
import math

"""General functions and definitions"""

class EdgeConv(MessagePassing):
    """Edge convolutional layer definition"""

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       SiLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

def preprocess(data, scale=True, targetsum=1e4, run_pca=True, comps=500):
    """method for preprocessing raw counts matrix"""

    adata = ad.AnnData(data, dtype=data.dtype)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=targetsum, layer=None)
    new_data = adata.X
    new_data = sc.pp.log1p(new_data)
    
    if scale:
        new_data = sc.pp.scale(new_data)
    
    if run_pca:
        pca = PCA(n_components=comps, random_state=8)
        new_data = pca.fit_transform(new_data)
    
    return new_data

def mask_labels(labels, masking_pct):
    """method for masking labels"""
    random.seed(8)
    subset = random.sample(range(len(labels)), math.floor((len(labels)*masking_pct)))
    masked_labels = np.copy(labels)
    masked_labels[subset] = -1 

    return labels, masked_labels  


    
