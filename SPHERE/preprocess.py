import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 


def construct_combined_graph(adata, n_neighbors=3, k=10, mode="connectivity", metric="correlation", include_self=False):
    
    # spatial graph
    cell_position = adata.obsm['spatial']
    spatial_graph = construct_graph_by_coordinate(cell_position, n_neighbors=n_neighbors)

    # feature graph 
    feature_graph = construct_graph_by_feature(adata, k=k, mode=mode, metric=metric, include_self=include_self)
    if not isinstance(feature_graph, coo_matrix):
        feature_graph = feature_graph.tocoo()
    spatial_graph_coo = coo_matrix(
        (spatial_graph['value'], (spatial_graph['x'], spatial_graph['y'])),
        shape=(adata.obsm['spatial'].shape[0], adata.obsm['spatial'].shape[0])
    )

    # combine
    combined_graph = (spatial_graph_coo + feature_graph).astype(bool).astype(int)
    combined_graph = combined_graph.tocsr()

    return combined_graph


def construct_neighbor_graph(adata, loc_neighbors=6, gene_neighbors=14, com_neighbors=5): 
    
    # spatial graph
    cell_position = adata.obsm['spatial']
    adj_spatial = construct_graph_by_coordinate(cell_position, n_neighbors=loc_neighbors)
    adata.uns['adj_spatial'] = adj_spatial
    
    # feature graph 
    adj_feature = construct_graph_by_feature(adata,k=gene_neighbors)
    adata.obsm['adj_feature'] = adj_feature

    adata.obsm['adj_combined'] = construct_combined_graph(adata, n_neighbors=loc_neighbors, k=com_neighbors)# 6 5
    
    return adata

def construct_neighbor_graph_inte(adata, slice_name_list, loc_neighbors=6, gene_neighbors=20): #6

    spatial_graphs = []
    spatial_offsets = [0]

    for slice in slice_name_list:

        cell_position = adata[adata.obs['slice_name'] == slice].obsm['spatial']
        spatial_graph = construct_graph_by_coordinate(cell_position, n_neighbors=loc_neighbors)
        spatial_graph['x'] += spatial_offsets[-1]
        spatial_graph['y'] += spatial_offsets[-1]
        spatial_graphs.append(spatial_graph)
        spatial_offsets.append(spatial_offsets[-1] + cell_position.shape[0])

    combined_spatial_graph = pd.concat(spatial_graphs, ignore_index=True)
    adata.uns['adj_spatial'] = combined_spatial_graph
    
    adj_feature = construct_graph_by_feature(adata,k=gene_neighbors)#20
    adata.obsm['adj_feature'] = adj_feature
    adata.obsm['adj_combined'] = adj_feature
    
    return adata



def construct_neighbor_graph_inte_scale(adata, slice_name_list, loc_neighbors=6, gene_neighbors=20):
    from scipy.sparse import csr_matrix, coo_matrix
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph
    import numpy as np

    # --- 1. 空间图 (Spatial Graph) ---
    spatial_rows = []
    spatial_cols = []
    current_offset = 0
    
    for slice_name in slice_name_list:
        coords = adata[adata.obs['slice_name'] == slice_name].obsm['spatial']
        # 空间坐标通常只有2-3维，精确搜索非常快，没必要用近似
        nbrs = NearestNeighbors(n_neighbors=loc_neighbors+1, algorithm='kd_tree', n_jobs=-1).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        # 构建 COO 格式所需的行列索引
        row = np.repeat(np.arange(coords.shape[0]), loc_neighbors)
        col = indices[:, 1:].flatten()
        
        spatial_rows.append(row + current_offset)
        spatial_cols.append(col + current_offset)
        current_offset += coords.shape[0]

    # 合并并构建初始稀疏矩阵
    adj_spatial = coo_matrix((np.ones(len(np.concatenate(spatial_rows))), 
                             (np.concatenate(spatial_rows), np.concatenate(spatial_cols))), 
                             shape=(adata.n_obs, adata.n_obs)).tocsr()
    
    # 【关键：对称化与二值化】 对应原版 adj = adj + adj.T 且 np.where > 1
    adj_spatial = adj_spatial + adj_spatial.transpose()
    adj_spatial.data = np.ones_like(adj_spatial.data) # 快速二值化
    
    adata.uns['adj_spatial_sparse'] = adj_spatial 

    # --- 2. 特征图 (Feature Graph) ---
    print(f"Constructing feature graph for {adata.n_obs} spots...")
    data_feat = adata.obsm['feat']
    
    # 自动切换：小规模用精确搜索，大规模用 pynndescent
    if adata.n_obs < 20000:
        adj_feature = kneighbors_graph(data_feat, gene_neighbors, mode='connectivity', 
                                       metric='correlation', include_self=False, n_jobs=-1)
    else:
        from pynndescent import NNDescent
        index = NNDescent(data_feat, metric='correlation', n_neighbors=gene_neighbors + 1, n_jobs=-1)
        indices, _ = index.query(data_feat, k=gene_neighbors + 1)
        
        f_rows = np.repeat(np.arange(adata.n_obs), gene_neighbors)
        f_cols = indices[:, 1:].flatten()
        adj_feature = coo_matrix((np.ones(f_rows.size), (f_rows, f_cols)), 
                                 shape=(adata.n_obs, adata.n_obs)).tocsr()
    
    # 【关键：对称化与二值化】
    adj_feature = adj_feature + adj_feature.transpose()
    adj_feature.data = np.ones_like(adj_feature.data)
    
    adata.obsm['adj_feature'] = adj_feature
    adata.obsm['adj_combined'] = adj_feature
    
    return adata

def adjacent_matrix_preprocessing_scale(adata):
    import scipy.sparse as sp
    import torch
    
    def sparse_preprocess(indices_mat):
        # 1. 添加自连接 (A + I)
        adj_i = indices_mat + sp.eye(indices_mat.shape[0])
        
        # 2. 计算对称度归一化 D^-0.5 * A * D^-0.5
        rowsum = np.array(adj_i.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        adj_normalized = d_mat_inv_sqrt.dot(adj_i).dot(d_mat_inv_sqrt).tocoo()
        
        # 3. 转为 torch 稀疏张量
        indices = torch.from_numpy(np.vstack((adj_normalized.row, adj_normalized.col)).astype(np.int64))
        values = torch.from_numpy(adj_normalized.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, torch.Size(adj_normalized.shape))

    return {
        'adj_spatial': sparse_preprocess(adata.uns['adj_spatial_sparse']),
        'adj_feature': sparse_preprocess(adata.obsm['adj_feature']),
        'adj_combined': sparse_preprocess(adata.obsm['adj_combined'])
    }

def construct_neighbor_graph_decon(adata_st, adata_sc, loc_neighbors=6, gene_neighbors=14): 
    
    # spatial graph
    cell_position = adata_st.obsm['spatial']
    adj_spatial = construct_graph_by_coordinate(cell_position, n_neighbors=loc_neighbors)
    adata_st.uns['adj_spatial'] = adj_spatial
    
    # feature graph 
    adj_feature = construct_graph_by_feature(adata_sc,k=gene_neighbors)#14
    adata_sc.obsm['adj_feature'] = adj_feature
    
    return adata_st, adata_sc

def pca(adata, use_reps=None, n_comps=10):
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    return feat_pca

def pca_feat(adata, use_reps=None, n_comps=10):
    
    X = tfidf(adata.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comps)

    input_matrix = pca.fit_transform(X_norm.toarray())
    adata.obsm["X_pca"] = input_matrix
 

def construct_graph_by_feature(adata, k=10, mode= "connectivity", metric="correlation", include_self=False):
    feature_graph=kneighbors_graph(adata.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    return feature_graph

def construct_graph_by_coordinate(cell_position, n_neighbors=3):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)  
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)

    return adj

def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # adj_normalized = sp.coo_matrix(adj)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def adjacent_matrix_preprocessing(adata):
    """Converting dense adjacent matrix to sparse adjacent matrix"""
    
    # spatial graph
    adj_spatial = adata.uns['adj_spatial']
    adj_spatial = transform_adjacent_matrix(adj_spatial)
    adj_spatial = adj_spatial.toarray()   
    adj_spatial = adj_spatial + adj_spatial.T
    adj_spatial = np.where(adj_spatial>1, 1, adj_spatial)
    adj_spatial = preprocess_graph(adj_spatial) 
    
    # feature graph
    adj_feature = torch.FloatTensor(adata.obsm['adj_feature'].copy().toarray())
    adj_feature = adj_feature + adj_feature.T
    adj_feature = np.where(adj_feature>1, 1, adj_feature)
    adj_feature = preprocess_graph(adj_feature) 

    # combine graph
    adj_combined = torch.FloatTensor(adata.obsm['adj_combined'].copy().toarray())
    adj_combined = adj_combined + adj_combined.T
    adj_combined = np.where(adj_combined>1, 1, adj_combined)
    adj_combined = preprocess_graph(adj_combined) 
    
    adj = {'adj_spatial': adj_spatial,
           'adj_feature': adj_feature,
           'adj_combined':adj_combined}
    
    return adj

def adjacent_matrix_preprocessing_decon(adata_st, adata_sc):
    """Converting dense adjacent matrix to sparse adjacent matrix"""
    
    # spatial graph
    adj_spatial = adata_st.uns['adj_spatial']
    adj_spatial = transform_adjacent_matrix(adj_spatial)
    adj_spatial = adj_spatial.toarray()
    adj_spatial = adj_spatial + adj_spatial.T
    adj_spatial = np.where(adj_spatial>1, 1, adj_spatial)
    adj_spatial = preprocess_graph(adj_spatial)
    
    # feature graph
    adj_feature = torch.FloatTensor(adata_sc.obsm['adj_feature'].copy().toarray())
    adj_feature = adj_feature + adj_feature.T
    adj_feature = np.where(adj_feature>1, 1, adj_feature)
    adj_feature = preprocess_graph(adj_feature) 

    adj = {'adj_spatial': adj_spatial,
           'adj_feature': adj_feature}
    
    return adj

def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def lsi_scale(adata, n_components=201, use_highly_variable=None, **kwargs):
    import sklearn.utils.extmath
    from scipy.sparse import issparse
    
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    
    X = adata_use.X
    
    if issparse(X):
        row_sums = np.array(X.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0 
        X_tf = X.multiply(1.0 / row_sums[:, np.newaxis])
        
        col_sums = np.array(X.sum(axis=0)).flatten()
        col_sums[col_sums == 0] = 1.0
        idf = np.log(1.0 + X.shape[0] / col_sums)
        
        X_tfidf = X_tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        idf = np.log(1.0 + X.shape[0] / X.sum(axis=0))
        X_tfidf = tf * idf

    if issparse(X_tfidf):
        scales = np.array(X_tfidf.sum(axis=1)).flatten()
        scales[scales == 0] = 1.0
        X_norm = X_tfidf.multiply(1e4 / scales[:, np.newaxis])
        X_norm.data = np.log1p(X_norm.data) 
    else:
        X_norm = np.log1p((X_tfidf / X_tfidf.sum(axis=1, keepdims=True)) * 1e4)

    U, Sigma, VT = sklearn.utils.extmath.randomized_svd(
        X_norm, 
        n_components=n_components, 
        n_iter=5, 
        random_state=42
    )
    
    X_lsi = U * Sigma 
    X_lsi -= X_lsi.mean(axis=0)
    X_lsi /= X_lsi.std(axis=0, ddof=1)
    adata.obsm["X_lsi"] = X_lsi[:, 1:].astype(np.float32)

    
def tfidf(X):
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   
    
def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'    


from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected
from typing import Optional, Union
def Cal_Spatial_Net(
    adata,
    rad_cutoff: Optional[Union[None, int]] = None,
    k_cutoff: Optional[Union[None, int]] = None,
    model: Optional[str] = "Radius",
    return_data: Optional[bool] = False,
    verbose: Optional[bool] = True,
) -> None:
    assert model in ["Radius", "KNN"]
    if verbose:
        print("Calculating spatial neighbor graph ...")

    if model == "KNN":
        edge_index = knn_graph(
            x=torch.tensor(adata.obsm["spatial"]),
            flow="target_to_source",
            k=k_cutoff,
            loop=True,
            num_workers=8,
        )
        edge_index = to_undirected(
            edge_index, num_nodes=adata.shape[0]
        )  # ensure the graph is undirected
    elif model == "Radius":
        edge_index = radius_graph(
            x=torch.tensor(adata.obsm["spatial"]),
            flow="target_to_source",
            r=rad_cutoff,
            loop=True,
            num_workers=8,
        )

    graph_df = pd.DataFrame(edge_index.numpy().T, columns=["Cell1", "Cell2"])
    id_cell_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    graph_df["Cell1"] = graph_df["Cell1"].map(id_cell_trans)
    graph_df["Cell2"] = graph_df["Cell2"].map(id_cell_trans)
    adata.uns["Spatial_Net"] = graph_df

    if verbose:
        print(f"The graph contains {graph_df.shape[0]} edges, {adata.n_obs} cells.")
        print(f"{graph_df.shape[0]/adata.n_obs} neighbors per cell on average.")

    if return_data:
        return adata

import ot
def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    adata.obsm['adj'] = adj
    return adj

from scipy.sparse import coo_matrix, issparse
from torch_geometric.data import Data
def Preprocess_without_SVD(adatas,
    dim: Optional[int] = 50,
    used_obsmkey:[str] = 'emb_latent',
    self_loop: Optional[bool] = False,
    SVD = True,
    join: Optional[str] = "inner",
    backend: Optional[str] = "sklearn",
    mincells_ratio : Optional[float] = 0.01,
    use_highly_variable: Optional[bool]=True,
    singular: Optional[bool] = True,
    check_order: Optional[bool] = True,
    n_top_genes: Optional[int] = 2500,
    device: Optional[str] = "cpu"):

    gpu_flag = True if torch.cuda.is_available() else False

    edgeLists = []
    for adata in adatas:
        G_df = adata.uns["Spatial_Net"].copy()
        cells = np.array(adata.obs_names)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        G_df["Cell1"] = G_df["Cell1"].map(cells_id_tran)
        G_df["Cell2"] = G_df["Cell2"].map(cells_id_tran)

        # build adjacent matrix
        G = scipy.sparse.coo_matrix(
            (np.ones(G_df.shape[0]), (G_df["Cell1"], G_df["Cell2"])),
            shape=(adata.n_obs, adata.n_obs),
        )
        if self_loop:
            G = G + scipy.sparse.eye(G.shape[0])
        edgeList = np.nonzero(G)
        edgeLists.append(edgeList)

    if issparse(adatas[0].X):
        adatas[0].X = adatas[0].X.todense()
    if issparse(adatas[1].X):
        adatas[1].X = adatas[1].X.todense()

    data_x = Data(
        edge_index=torch.LongTensor(np.array([edgeLists[0][0], edgeLists[0][1]])), x=adatas[0].X
    )
    data_y = Data(
        edge_index=torch.LongTensor(np.array([edgeLists[1][0], edgeLists[1][1]])), x=adatas[1].X
    )
    datas = [data_x, data_y]    

    edges = [dataset.edge_index for dataset in datas]
    features = [dataset.x for dataset in datas]

    return edges, features

def find_rigid_transform(A, B):
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)


    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t
