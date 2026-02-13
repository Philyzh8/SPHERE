import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module
import time
from tqdm import tqdm


class MultiHeadCombinedAttentionLayer(Module):
    def __init__(self, in_feat, num_heads, device, dropout=0.1, inte=False):
        super(MultiHeadCombinedAttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.num_heads = num_heads
        self.head_dim = in_feat // num_heads
        assert in_feat % num_heads == 0, "Input feature size must be divisible by the number of heads."
        self.device = device
        self.inte = inte

        self.w_q = Parameter(torch.FloatTensor(in_feat, num_heads * self.head_dim))
        self.w_k = Parameter(torch.FloatTensor(in_feat, num_heads * self.head_dim))
        self.w_v = Parameter(torch.FloatTensor(in_feat, num_heads * self.head_dim))
        self.out_proj = nn.Linear(in_feat, in_feat)  # 投影层

        self.gamma = Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout)

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.in_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_q)
        torch.nn.init.xavier_uniform_(self.w_k)
        torch.nn.init.xavier_uniform_(self.w_v)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)

        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, emb1, emb2, adj):

        N = emb1.size(0)  
        emb = torch.stack([emb1, emb2], dim=1)  # (N, 2, in_feat)

        q = torch.einsum("bif,fm->bim", emb, self.w_q)  # (N, 2, num_heads * head_dim)
        k = torch.einsum("bif,fm->bim", emb, self.w_k)
        v = torch.einsum("bif,fm->bim", emb, self.w_v)

        q = q.view(N, 2, self.num_heads, self.head_dim)  # (N, 2, num_heads, head_dim)
        k = k.view(N, 2, self.num_heads, self.head_dim)
        v = v.view(N, 2, self.num_heads, self.head_dim)

        attn_scores = torch.einsum("bnhd,bmhd->bnhm", q, k) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )  # (N, num_heads, 2, 2)

        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        attn_out = torch.einsum("bnhm,bmhd->bnhd", attn_weights, v)  # (N, num_heads, 2, head_dim)
        attn_out = attn_out.flatten(start_dim=2)  # (N, 2, num_heads * head_dim)
        emb_combined = self.out_proj(attn_out[:, 0, :])  # (N, in_feat)

        if self.inte:
            z_tilde = emb_combined
            return z_tilde

        else:
            s = F.softmax(torch.mm(emb_combined, emb_combined.t()), dim=1)  # (N, N)
            z_g = torch.mm(s, emb_combined)  # (N, in_feat)
            z_tilde = self.gamma * z_g + emb_combined  

            z_tilde = torch.mm(z_tilde, self.weight)
            z_tilde = torch.spmm(adj, z_tilde)
            return z_tilde



class PredictionModel(Module):
    def __init__(self, in_feat, hid_feat, out_feat, celltype_dims, dropout):
        super().__init__()

        self.pred = nn.Sequential(
            nn.Linear(out_feat, hid_feat),
            nn.LeakyReLU(),
            nn.LayerNorm(hid_feat),
            nn.Dropout(dropout),
            nn.Linear(hid_feat, hid_feat),
            nn.LeakyReLU(),
            nn.LayerNorm(hid_feat),
            nn.Dropout(dropout),
            nn.Linear(hid_feat, celltype_dims),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(celltype_dims, hid_feat),
            nn.LeakyReLU(),
            nn.Linear(hid_feat, hid_feat),
            nn.LeakyReLU(),
            nn.Linear(hid_feat, in_feat)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.pred[0].weight)
        nn.init.xavier_uniform_(self.pred[4].weight)
        nn.init.xavier_uniform_(self.pred[-2].weight)
        nn.init.kaiming_normal_(self.decoder[0].weight)
        nn.init.xavier_uniform_(self.decoder[2].weight)
        nn.init.xavier_uniform_(self.decoder[-1].weight)
 
    def forward(self, z):

        pred = self.pred(z)
        decoded = self.decoder(pred)

        return pred, decoded
    

def cpot(L1, L2, crossC, intraC1, intraC2, inIter, outIter, H, l1, l2, l3, l4):

    nx, ny = crossC.shape
    l4 = l4 * nx * ny
    device = crossC.device

    # define initial matrix values
    a = torch.ones((nx, 1)).to(torch.float64).to(device) / nx
    b = torch.ones((1, ny)).to(torch.float64).to(device) / ny
    r = torch.ones((nx, 1)).to(torch.float64).to(device) / nx
    c = torch.ones((1, ny)).to(torch.float64).to(device) / ny
    l_total = l1 + l2 + l3

    T = torch.ones((nx, ny)).to(torch.float64).to(device) / (nx * ny)
    H = H.T + torch.ones((nx, ny)).to(torch.float64).to(device) / ny

    # functions for OT
    def mina(H_in, epsilon):
        in_a = torch.ones((nx, 1)).to(torch.float64).to(device) / nx
        return -epsilon * torch.log(torch.sum(in_a * torch.exp(-H_in / epsilon), dim=0, keepdim=True))

    def minb(H_in, epsilon):
        in_b = torch.ones((1, ny)).to(torch.float64).to(device) / ny
        return -epsilon * torch.log(torch.sum(in_b * torch.exp(-H_in / epsilon), dim=1, keepdim=True))

    def minaa(H_in, epsilon):
        return mina(H_in - torch.min(H_in, dim=0).values.view(1, -1), epsilon) + torch.min(H_in, dim=0).values.view(1, -1)

    def minbb(H_in, epsilon):
        return minb(H_in - torch.min(H_in, dim=1).values.view(-1, 1), epsilon) + torch.min(H_in, dim=1).values.view(-1, 1)

    temp1 = 0.5 * (intraC1 ** 2) @ r @ torch.ones((1, ny)).to(torch.float64).to(device) + 0.5 * torch.ones((nx, 1)).to(torch.float64).to(device) @ c @ (intraC2 ** 2).T

    resRecord = []
    WRecord = []
    start_time = time.time()
    for i in tqdm(range(outIter), desc="Computing constraint proximal point iteration"):
        T_old = torch.clone(T)
        CGW = temp1 - intraC1 @ T @ intraC2.T
        C = crossC - l2 * torch.log(L1 @ T @ L2.T) - l3 * torch.log(H) + l4 * CGW

        if i == 0:
            C_old = C
        else:
            W_old = torch.sum(T * C_old)
            W = torch.sum(T * C)
            if W <= W_old:
                C_old = C
            else:
                C = C_old

        Q = C - l1 * torch.log(T)
        for j in range(inIter):
            a = minaa(Q - b, l_total)
            b = minbb(Q - a, l_total)
            pass

        T = 0.05 * T_old + 0.95 * r * torch.exp((a + b - Q) / l_total) * c
        res = torch.sum(torch.abs(T - T_old))
        resRecord.append(res)
        WRecord.append(torch.sum(T * C))

    end_time = time.time()
    print(f"Time for optimization: {end_time - start_time:.2f}s")

    return T, WRecord, resRecord


def get_cost(A1, A2, X1, X2, H, rwrIter, rwIter, alpha, beta, gamma):

    start_time = time.time()

    # calculate RWR
    T1 = cal_trans(A1, None)
    T2 = cal_trans(A2, None)
    rwr1, rwr2 = get_sep_rwr(T1, T2, H, beta, rwrIter)
    rwrCost = get_cross_cost(rwr1, rwr2, H)

    # cross/intra-graph cost based on node attributes
    if X1 is None or X2 is None:
        X1 = rwr1
        X2 = rwr2

    intraC1 = get_intra_cost(X1) * A1
    intraC2 = get_intra_cost(X2) * A2
    crossC = get_cross_cost(X1, X2, H)

    # rwr on the product graph
    crossC = crossC + alpha * rwrCost
    L1 = A1 / A1.sum(1, keepdim=True).to(torch.float64)
    L2 = A2 / A2.sum(1, keepdim=True).to(torch.float64)

    crossC = get_prod_rwr(L1, L2, crossC, H, beta, gamma, rwIter)

    end_time = time.time()
    print(f"Time for cost matrix: {end_time - start_time:.2f}s")

    # if not os.path.exists(f"datasets/rwr"):
    #     os.makedirs(f"datasets/rwr")
    # np.savez(f"datasets/rwr/rwr_cost.npz",
    #          rwr1=rwr1.cpu().numpy(), 
    #          rwr2=rwr2.cpu().numpy(),
    #          cross_rwr=rwrCost.cpu().numpy())

    return crossC, intraC1, intraC2

def cal_trans(A, X=None):

    n = A.shape[0]

    if X is None:
        X = torch.ones((n, 1)).to(torch.float64).to(A.device)
    X = X / torch.linalg.norm(X, dim=1, ord=2, keepdim=True)
    sim = X @ X.T
    T = sim * A
    for i in range(n):
        T[i, torch.where(T[i] != 0)[0]] = F.softmax(T[i, torch.where(T[i] != 0)[0]], dim=0)

    return T


def get_sep_rwr(T1, T2, H, beta, sepRwrIter):
    eps = 1e-5

    anchors1, anchors2 = torch.where(H.T == 1)
    n1, n2 = T1.shape[0], T2.shape[0]
    num_anchors = anchors1.shape[0]

    e1 = torch.zeros((n1, num_anchors)).to(torch.float64).to(T1.device)
    e2 = torch.zeros((n2, num_anchors)).to(torch.float64).to(T2.device)
    e1[(anchors1, torch.arange(num_anchors))] = 1
    e2[(anchors2, torch.arange(num_anchors))] = 1

    r1 = torch.zeros((n1, num_anchors)).to(torch.float64).to(T1.device)
    r2 = torch.zeros((n2, num_anchors)).to(torch.float64).to(T2.device)

    for i in tqdm(range(sepRwrIter), desc="Computing separate RWR scores"):
        r1_old = torch.clone(r1)
        r2_old = torch.clone(r2)
        r1 = (1 - beta) * T1 @ r1 + beta * e1
        r2 = (1 - beta) * T2 @ r2 + beta * e2
        diff = torch.max(torch.max(torch.abs(r1 - r1_old)), torch.max(torch.abs(r2 - r2_old)))
        if diff < eps:
            break

    return r1, r2


def get_cross_cost(X1, X2, H):

    _, d = X1.shape
    X1_zero_pos = torch.where(X1.abs().sum(1) == 0)
    X2_zero_pos = torch.where(X2.abs().sum(1) == 0)
    if X1_zero_pos[0].shape[0] != 0:
        X1[X1_zero_pos] = torch.ones(d).to(torch.float64).to(X1.device)
    if X2_zero_pos[0].shape[0] != 0:
        X2[X2_zero_pos] = torch.ones(d).to(torch.float64).to(X2.device)

    X1 = X1 / torch.linalg.norm(X1, dim=1, ord=2, keepdim=True)
    X2 = X2 / torch.linalg.norm(X2, dim=1, ord=2, keepdim=True)

    crossCost = torch.exp(-(X1 @ X2.T))
    crossCost[torch.where(H.T == 1)] = 0

    return crossCost


def get_intra_cost(X):

    _, d = X.shape
    X_zero_pos = torch.where(X.abs().sum(1) == 0)
    if X_zero_pos[0].shape[0] != 0:
        X[X_zero_pos] = torch.ones(d).to(torch.float64)
    X = X / torch.linalg.norm(X, dim=1, ord=2, keepdim=True)
    intraCost = torch.exp(-(X @ X.T))

    return intraCost



def get_prod_rwr(L1, L2, nodeCost, H, beta, gamma, prodRwrIter):

    eps = 1e-2
    nx, ny = H.T.shape
    HInd = torch.where(H.T == 1)
    crossCost = torch.zeros((nx, ny)).to(torch.float64).to(L1.device)
    for i in tqdm(range(prodRwrIter), desc="Computing product RWR scores"):
        rwCost_old = torch.clone(crossCost)
        crossCost = (1 + gamma * beta) * nodeCost + (1 - beta) * gamma * L1 @ crossCost @ L2.T
        crossCost[HInd] = 0
        if torch.max(torch.abs(crossCost - rwCost_old)) < eps:
            break
    crossCost = (1 - gamma) * crossCost
    crossCost[HInd] = 0

    return crossCost
