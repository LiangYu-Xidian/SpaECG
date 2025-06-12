import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GATConv



class GraphAttentionNet(nn.Module):

    def __init__(self, in_features, out_features, hidden_layers=[128, 64, 32], num_head=1, dropout=0.1,
                 alpha=1.0, concat=True):
        super(GraphAttentionNet, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden = hidden_layers
        self.num_head = num_head

        self.concat = concat

        if self.concat:
            hidden1_dim = self.num_head * self.hidden[0]
            hidden2_dim = self.num_head * self.hidden[2]
        else:
            hidden1_dim = self.hidden[0]
            hidden2_dim = self.hidden[2]

        # 使用多头注意力机制

        self.attr1 = GATConv(self.in_features, self.hidden[0], self.num_head, self.concat)
        self.attr2 = GATConv(self.hidden[1], self.hidden[2], self.num_head, self.concat)

        self.lin1 = nn.Sequential(nn.Linear(hidden1_dim, self.hidden[1]),
                                  nn.BatchNorm1d(self.hidden[1]),
                                  nn.ELU(alpha),
                                  nn.Dropout(dropout))

        self.lin2 = nn.Sequential(nn.Linear(hidden2_dim, self.out_features),
                                  nn.BatchNorm1d(self.out_features),
                                  nn.ELU(alpha),
                                  nn.Dropout(dropout))


    def forward(self, input, adj):
        
        x1 = self.attr1(input, adj)

        x2 = self.lin1(x1)
        x2 = self.attr2(x2, adj)
        out = self.lin2(x2)
        return out
