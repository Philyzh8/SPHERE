import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from SPHERE.layer import *
    
class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.act(torch.mm(z, z.t())) 
        return adj


class AttnAE(Module):
      
    def __init__(self, in_feat, hid_feat, out_feat, device, dropout=0.1, add_act=False, inte=False):
        super(AttnAE, self).__init__()
        self.in_feat = in_feat
        self.hid_feat = hid_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.add_act = add_act
        self.inte = inte
        self.device = device
    
        self.encoder = Encoder(self.in_feat, self.hid_feat, self.out_feat, add_act=self.add_act)
        self.decoder = Decoder(self.out_feat, self.hid_feat, self.in_feat, add_act=self.add_act)
        self.atten = MultiHeadCombinedAttentionLayer(self.out_feat, 8, self.device, dropout=self.dropout, inte=self.inte)
        
        
    def forward(self, features, adj_spatial, adj_feature, adj_combined):
            
        latent_spatial, spatial_rec = self.encoder(features, adj_spatial)  
        latent_feature, feature_rec = self.encoder(features, adj_feature)
        
        latent = self.atten(latent_spatial, latent_feature, adj_combined)

        recon = self.decoder(latent, adj_feature)

        results = {'latent':latent,
                   'recon':recon,
                   'latent_spatial':latent_spatial,
                   'latent_feature':latent_feature,
                   'spatial_rec':spatial_rec,
                   'feature_rec':feature_rec,
                   }
        
        return results     
    
class Encoder_decon(Module):
      
    def __init__(self, in_feat, hid_feat, out_feat, celltype_dims, device, dropout=0.6):
        super(Encoder_decon, self).__init__()
        self.in_feat = in_feat
        self.hid_feat = hid_feat
        self.out_feat = out_feat
        self.celltype_dims = celltype_dims
        self.dropout = dropout
        self.device = device
    
        self.encoder = Encoder(self.in_feat, self.hid_feat, self.out_feat)
        self.predmodel = PredictionModel(self.in_feat, self.hid_feat, self.out_feat, 
                                         celltype_dims=self.celltype_dims, dropout=self.dropout)
        
        
    def forward(self, features, features_sc, adj_spatial, adj_feature):
            
        latent_spatial, spatial_graph_rec = self.encoder(features, adj_spatial) 
        pred_st, spatial_rec = self.predmodel(latent_spatial) 
        latent_feature, feature_graph_rec = self.encoder(features_sc, adj_feature)
        pred_sc, feature_rec = self.predmodel(latent_feature)

        results = {'latent_spatial':latent_spatial,
                   'latent_feature':latent_feature,
                   'spatial_rec':spatial_rec,
                   'feature_rec':feature_rec,
                   'spatial_graph_rec':spatial_graph_rec,
                   'feature_graph_rec':feature_graph_rec,
                   'pred_st': pred_st,
                   'pred_sc': pred_sc
                   }
        
        return results     


class Encoder(Module): 
    
    def __init__(self, in_feat, hid_feat, out_feat, dropout=0.0, add_act=False, act=nn.ReLU()):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.add_act = add_act
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_feat, hid_feat))
        self.weight2 = Parameter(torch.FloatTensor(hid_feat, self.out_feat))
        self.dc = InnerProductDecoder(0.0, act=nn.Sigmoid())
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight1)
        x = torch.spmm(adj, x)

        if self.add_act:
            x = self.act(x)
            
        x = torch.mm(x, self.weight2)
        x = torch.spmm(adj, x)

        A_rec = self.dc(x)
        
        return x, A_rec
    
class Decoder(Module):
    
    def __init__(self, out_feat, hid_feat, in_feat, dropout=0.0, add_act=False, act=nn.ReLU()):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.add_act = add_act
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.out_feat, hid_feat))
        self.weight2 = Parameter(torch.FloatTensor(hid_feat, self.in_feat))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight1)
        x = torch.spmm(adj, x)
        
        if self.add_act:
            x = self.act(x)
            
        x = torch.mm(x, self.weight2)
        x = torch.spmm(adj, x)

        return x   


def align(A1, A2, X1, X2, H, sepRwrIter, prodRwrIter, alpha, beta, gamma, inIter, outIter, l1, l2, l3, l4, device='cpu'):

    A1 = A1.to(device)
    A2 = A2.to(device)
    if X1 is not None:
        X1 = X1.to(device)
    if X2 is not None:
        X2 = X2.to(device)
    H = H.to(device)

    nx, ny = H.T.shape

    if torch.sum(A1.sum(1) == 0) != 0:
        A1[torch.where(A1.sum(1) == 0)] = torch.ones(nx).int().to(device)
    if torch.sum(A2.sum(1) == 0) != 0:
        A2[torch.where(A2.sum(1) == 0)] = torch.ones(ny).int().to(device)

    L1 = A1 / A1.sum(1, keepdim=True).to(torch.float64)
    L2 = A2 / A2.sum(1, keepdim=True).to(torch.float64)

    crossC, intraC1, intraC2 = get_cost(A1, A2, X1, X2, H, sepRwrIter, prodRwrIter, alpha, beta, gamma)
    T, W, res = cpot(L1, L2, crossC, intraC1, intraC2, inIter, outIter, H, l1, l2, l3, l4)
    print(intraC1.shape,intraC2.shape)
    return T.cpu(), W, res           



