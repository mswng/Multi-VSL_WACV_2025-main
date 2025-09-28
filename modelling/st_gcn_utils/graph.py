from torch import nn
import torch
import math
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
from torch.nn.functional import normalize
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

    def __init__(self,
                 strategy='uniform',
                 max_hop=1,
                 dilation=1,
                 norm = True):
        self.max_hop = max_hop
        self.dilation = dilation
        self.norm = norm
        self.get_edge()
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)
        

    def __str__(self):
        return self.A

    def get_edge(self):
       
        self.num_node = 46
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            (0, 2),(1, 3),(2, 4),(3, 5),(0, 1),(4, 6),(6, 7),(7, 8),(8, 9),(4, 10),(10, 11),(11, 12),(12, 13),(10, 14),(14, 15),(15, 16),
            (16, 17),(14, 18),(18, 19),(19, 20),(20, 21),(18, 22),(4, 22),(22, 23),(23, 24),(24, 25),(5, 26),(26, 27),(27, 28),(28, 29),
            (5, 30),(30, 31),(31, 32),(32, 33),(30, 34),(34, 35),(35, 36), (36, 37),(34, 38),(38, 39),(39, 40),(40, 41),(38, 42),(5, 42),(42, 43),(43, 44),(44, 45)
        ]
        self.edge = self_link + neighbor_link
        self.center = 0
       
      
    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        if self.norm:
            adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = adjacency[j, i]
                            else:
                                a_further[j, i] = adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,gcn_dropout = 0, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.projection = nn.Linear(in_features,out_features)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(p=gcn_dropout)
        self.skip_projection = nn.Linear(in_features,out_features)
        self.norm = nn.BatchNorm1d(out_features)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features) * 0.1)
        else:
            self.register_parameter('bias', None)
        


    def forward(self, input, adj):
     
        support =  self.projection(input) # bt,n,k
        output = torch.einsum('nn,bnc->bnc',adj.squeeze(),support)
        if self.bias is not None:
           output =  output + self.bias
        output = self.relu(output)
        output = self.dropout(output)
        output = output + self.skip_projection(input) # skip connection
        output = self.norm(output.permute(0,2,1)).permute(0,2,1)
        return output
    
def calculate_laplacian_with_self_loop(matrix):
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian



class GCN(nn.Module):
    def __init__(self, adj, input_dim: int, output_dim: int, **kwargs):
        super(GCN, self).__init__()
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim  # seq_len for prediction
        self._output_dim = output_dim  # hidden_dim for prediction
        self.weights = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, inputs):
        bs,t,n,c = inputs.shape
        inputs = inputs.permute(0,1,3,2).contiguous().view(bs,t*c,n)
        # (batch_size, seq_len*c, num_nodes)
        batch_size = inputs.shape[0]
        # (num_nodes, batch_size, seq_len)
        inputs = inputs.transpose(0, 2).transpose(1, 2)
        # (num_nodes, batch_size * seq_len)
        inputs = inputs.view((self._num_nodes, batch_size*t * self._input_dim))
        # AX (num_nodes, batch_size * seq_len)
        ax = self.laplacian @ inputs
        # (num_nodes, batch_size, seq_len,c)
        ax = ax.view((self._num_nodes, batch_size,t, self._input_dim))
        # (num_nodes * batch_size, seq_len)
        ax = ax.contiguous().view((self._num_nodes * batch_size*t, self._input_dim))
        # act(AXW) (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.weights)
        # (num_nodes, batch_size,t, output_dim)
        outputs = outputs.view((self._num_nodes, batch_size,t, self._output_dim))
        # (batch_size,t, num_nodes, output_dim)
        outputs = outputs.contiguous().permute(1,2,0,3)
        return outputs


    @property
    def hyperparameters(self):
        return {
            "num_nodes": self._num_nodes,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
        }