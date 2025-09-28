from torch import nn
import torch
import math
from torch.nn.parameter import Parameter
import numpy as np

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

      

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self):
     
        self.get_edge()
        self.get_adjacency()
        

    def __str__(self):
        return self.A

    def get_edge(self):
       
        self.num_nodes = 46
        self_link = [(i, i) for i in range(self.num_nodes)]
        neighbor_link = [
            (0, 2),(1, 3),(2, 4),(3, 5),(0, 1),(4, 6),(6, 7),(7, 8),(8, 9),(4, 10),(10, 11),(11, 12),(12, 13),(10, 14),(14, 15),(15, 16),
            (16, 17),(14, 18),(18, 19),(19, 20),(20, 21),(18, 22),(4, 22),(22, 23),(23, 24),(24, 25),(5, 26),(26, 27),(27, 28),(28, 29),
            (5, 30),(30, 31),(31, 32),(32, 33),(30, 34),(34, 35),(35, 36), (36, 37),(34, 38),(38, 39),(39, 40),(40, 41),(38, 42),(5, 42),(42, 43),(43, 44),(44, 45)
        ]
        self.edge = self_link + neighbor_link
       
    def get_adjacency(self):
        self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        for edge in self.edge:
            self.adjacency_matrix[edge[0], edge[1]] = 1
            self.adjacency_matrix[edge[1], edge[0]] = 1  # because the graph is undirected

def calculate_laplacian_with_self_loop(matrix):
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, adj,in_features, out_features,gcn_dropout = 0.1):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.projection = nn.Linear(in_features,out_features)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(gcn_dropout)
        self.skip_projection = nn.Linear(in_features,out_features)
        self.adj = calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        self.norm = nn.LayerNorm(out_features)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


    def forward(self, input):
     
        output = torch.einsum('bnc,nn->bnc',input,self.adj.to(input))
        output = self.projection(output)
        output = self.tanh(output)
        output = self.dropout(output)
        output = self.norm(output)
        output = output + self.skip_projection(input) # skip connection
        return output