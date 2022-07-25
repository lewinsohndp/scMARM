import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, SiLU

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

if __name__ == "__main__":
    nodes = torch.tensor([[1,1], [2,2]]).float()
    edge_index = torch.tensor([[1],[0]])

    layer = EdgeConv(2,2)

    print(layer(nodes,edge_index))