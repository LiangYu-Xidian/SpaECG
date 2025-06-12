import numpy as np
from torch_geometric.nn import TransformerConv, LayerNorm, Sequential
import torch.nn.functional as F
import torch.nn as nn
import torch
from model.gat import GraphAttentionNet
from torch_geometric.utils import negative_sampling
from model.torch_utils import *
class GTrans(torch.nn.Module):
    def __init__(self, hidden_dims, decoder = False, num_heads=1, alpha=1.0):
        super().__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.activate = nn.ELU(alpha)
        self.conv1 = Sequential('x, edge_index', [(
                            TransformerConv(in_dim, num_hidden, heads=num_heads), 'x, edge_index -> x'),
                            LayerNorm(num_hidden),
                        ]) 

        self.conv2 = Sequential('x, edge_index', [(
                            TransformerConv(num_hidden, out_dim, heads=num_heads), 'x, edge_index -> x'),
                            LayerNorm(out_dim)
                        ])

    def forward(self, x, adj):
        a1 = self.conv1(x, adj)
        h1 = self.activate(a1)
        h2 = self.conv2(h1, adj)
        return h2
    
    
class Gene2VecPositionalEmbedding(nn.Module):
    def __init__(self, gene_emb_path = '../data/gene2vec_16906.npy'):
        super().__init__()
        gene_emb= np.load(gene_emb_path)
        gene_emb = torch.from_numpy(gene_emb)
        self.emb = nn.Embedding.from_pretrained(gene_emb)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)
    
class Identity(torch.nn.Module):
    def __init__(self, in_dim, num_class, dropout = .1, h_dim = [64,32]):
        super(Identity, self).__init__()
        # self.out_dim = out_dim
        
            
        self.mlp1 = nn.Sequential(nn.Linear(in_features=in_dim, out_features=h_dim[0], bias=True),
                                nn.ReLU(),
                                nn.Dropout(dropout)
                        )
        
        self.mlp2 = nn.Sequential(nn.Linear(in_features=h_dim[0], out_features=h_dim[1], bias=True),
                                nn.ReLU(),
                                nn.Dropout(dropout)
                        )
        
            
        self.to_out = nn.Linear(in_features=h_dim[1], out_features=num_class, bias=True)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.to_out(x)
        return x
    
class CGNet(nn.Module):
    def __init__(self, 
            cell_feature_dim,  # max length of sequence
            gene_feature_dim,
            embed_dim,  # encoder dim of tokens
            num_class,
            weights=None,
            lambda_recon=1,
            lambda_gene=0.5,
            lambda_cell=0.1,
            lambda_pred=0.05,
            path = "./gene_interaction_graph.npy"
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.lambda_recon = lambda_recon
        self.lambda_gene = lambda_gene
        self.lambda_cell = lambda_cell
        self.lambda_pred = lambda_pred
        
        self.gene_emb = Gene2VecPositionalEmbedding(path)
        self.gene_enc = GraphAttentionNet(gene_feature_dim, embed_dim)
        self.cell_enc = GraphAttentionNet(cell_feature_dim, embed_dim)
        self.to_out = Identity(embed_dim, num_class)
        self.weights = weights

    def forward(self, x, cell_adj, path_adj, return_cell_emb = False, return_gene_emb = False):
         '''
         x: b*m
         path_adj: m*m
         cell_adj: b*b
         '''
         gene_feature = self.gene_emb(x)
         gene_emb = self.gene_enc(gene_feature, path_adj)
         cell_emb = self.cell_enc(x, cell_adj)
         
         pred = self.to_out(cell_emb)  
         
         recX = torch.mm(cell_emb, gene_emb.T)
         
         out = {"pred":pred}
         out['recX'] = recX
         if return_cell_emb:
             out['cell_emb'] = cell_emb
         if return_gene_emb:
             out['gene_emb'] = gene_emb
         return out

     
    def get_gene_attn_weights(self, x):
        ##  Tudo 返回基因之间attn_weights
        return None
    
    def get_gene_embedding(self, x, path_adj):
        gene_feature = self.gene_emb(x)
        gene_emb = self.gene_enc(gene_feature, path_adj)
        return gene_feature, gene_emb
    
    def get_graph_from_Emd(self, emd, edge_index):
        value = (emd[edge_index[0]] * emd[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)
    
    def loss_adj(self, edge_index, emb, eps=1e-4):
        
        pos_loss = -torch.log(
            self.get_graph_from_Emd(emb, edge_index) + eps).mean()

        neg_edge_index = negative_sampling(edge_index, emb.size(0))
        neg_loss = -torch.log(1 - self.get_graph_from_Emd(emb, neg_edge_index) + eps).mean()
        # adj = torch.zeros(n, n)
        loss = (pos_loss + neg_loss)/2
        # adj[edge_index[0], edge_index[1]] = 1
        # rec_A = torch.sigmoid(torch.matmul(emb, emb.T))
        # loss = F.mse_loss(rec_A, adj, reduction='mean')
        # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        # loss = norm *  F.binary_cross_entropy_with_logits(adj, rec_A)
        return loss  
    
    def cos(self, emb):
        mat = torch.matmul(emb, emb.T)
        norm = torch.norm(emb, p=2, dim=1).reshape(emb.shape[0], 1)
        mat = torch.div(mat, torch.matmul(norm, norm.T))
        if(torch.any(torch.isnan(mat))):
            mat = torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)
        mat = mat-torch.diag_embed(torch.diag(mat))
        return mat
    
    def loss_cell(self, edge_index, emb):
        n = emb.shape[0]
        rec_A = torch.sigmoid(torch.matmul(emb, emb.T))
        rec_A = rec_A * (1 - torch.eye(rec_A.shape[0], device=rec_A.device)) # test
        
        adj = torch.eye(n).to(rec_A.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1
        
        adj = adj * (1 - torch.eye(adj.shape[0], device=rec_A.device))
        # rec_A = torch.sigmoid(self.cos(emb))
        loss = F.binary_cross_entropy(rec_A.view(-1), adj.view(-1), reduction='mean')
        # loss = F.mse_loss(rec_A, adj, reduction='mean')
        # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        # loss = norm *  F.binary_cross_entropy_with_logits(adj.view(-1), rec_A.view(-1))
        # n = emb.shape[0]
        # adj = torch.zeros(n, n)
        # adj[edge_index[0], edge_index[1]] = 1
        # mat = torch.sigmoid(self.cos(emb))
        # loss = -(torch.mul(adj, torch.log(mat)).mean() + torch.mul(1-adj, torch.log(1-mat)).mean())/2
        return loss
        
        
    def loss(self, out, x, y, cell_adj, gene_adj):
        cell_emb = out['cell_emb']
        gene_emb = out['gene_emb']
        loss_gene_adj = self.loss_adj(gene_adj, gene_emb)
        loss_cell_adj = self.loss_cell(cell_adj, cell_emb)
        loss_pred = F.cross_entropy(out['pred'], y, self.weights)  
        loss_rec = F.mse_loss(out['recX'], x, reduction = 'mean' )
        loss = self.lambda_gene * loss_gene_adj +\
               self.lambda_pred * loss_pred +\
               self.lambda_cell * loss_cell_adj +\
               self.lambda_recon * loss_rec
        return loss, loss_gene_adj, loss_cell_adj, loss_pred, loss_rec
