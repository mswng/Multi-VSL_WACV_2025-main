from modelling.graph_utils import *
from modelling.vtn_utils import *


class MMTensorNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim).unsqueeze(self.dim)
        std = torch.std(x, dim=self.dim).unsqueeze(self.dim)
        return (x - mean) / std

class GCN_BERT(nn.Module):

    def __init__(self,num_layers = 4,embed_size = 512,num_heads = 8,sequence_length = 16,num_classes = 199,dropout = 0.1,**kwargs):
        super(GCN_BERT, self).__init__()
        adj = Graph().adjacency_matrix
        gcn_dropout = 0.5
        self.graph_conv = nn.Sequential(
            GraphConvolutionLayer(adj,2,16,gcn_dropout=gcn_dropout),
            GraphConvolutionLayer(adj,16,16,gcn_dropout=gcn_dropout),
            GraphConvolutionLayer(adj,16,32,gcn_dropout=gcn_dropout),
            GraphConvolutionLayer(adj,32,32,gcn_dropout=gcn_dropout),
            GraphConvolutionLayer(adj,32,32,gcn_dropout=gcn_dropout),
            GraphConvolutionLayer(adj,32,32,gcn_dropout=gcn_dropout),
            GraphConvolutionLayer(adj,32,32,gcn_dropout=gcn_dropout),
        )
        self.norm = MMTensorNorm(-1)
        self.bottle_mm = nn.Linear(32*46, embed_size)
        self.relu = F.relu
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_size))
        self.self_attention_encoder = SelfAttention(embed_size, embed_size,
                                                    [num_heads] * num_layers,
                                                    sequence_length+1, layer_norm=True)
       
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.classifier = nn.Sequential(
            nn.Linear(embed_size,embed_size),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(embed_size,num_classes)
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(46*32,embed_size),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(embed_size,num_classes)
        )
        

    def forward(self, keypoints):
        bs,t,n,d = keypoints.shape
        graph_features = self.graph_conv(keypoints.view(bs*t,n,d)).view(bs,t,-1)
       
        attn_inputs = self.relu(self.bottle_mm(self.norm(graph_features)))
        zp = self.self_attention_encoder(torch.cat([self.cls_token.expand(bs,-1,-1).to(attn_inputs),attn_inputs],dim = 1),cls_token_encodings = True)
       
        cls_token = zp[:,0] # bs, embed_size

        y = self.classifier(cls_token) + self.classifier1(graph_features.mean(1))

        return {
            'logits':y
        }
