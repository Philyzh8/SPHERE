import numpy as np
import scanpy as sc
import pandas as pd
import ot
import anndata as ad

def refine_label(adata, radius=50, key='label', vertical_weight=3, min_support=0.5):

    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # Calculate spatial positions
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')  # Full distance matrix
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()  # Get sorted indices by distance
        neigh_type = []
        for j in range(1, n_neigh + 1):  # Exclude itself (index[0])
            neighbor_idx = index[j]
            if abs(position[i, 1] - position[neighbor_idx, 1]) > abs(position[i, 0] - position[neighbor_idx, 0]):
                # Neighbor is more vertically aligned, apply vertical weight
                neigh_type.extend([old_type[neighbor_idx]] * vertical_weight)
            else:
                # Horizontal or diagonal neighbor, normal weight
                neigh_type.append(old_type[neighbor_idx])

        label_counts = {label: neigh_type.count(label) for label in set(neigh_type)}
        max_label = max(label_counts, key=label_counts.get)

        if label_counts[max_label] / len(neigh_type) >= min_support:
            new_type.append(max_label)
        else:
            new_type.append(old_type[i])
    
    new_type = [str(i) for i in list(new_type)]
    return new_type

def refine_label1(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    
    return new_type



def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     

def create_adata_from_latent(adata_ref, latent_key="X_lsi"):

    new_X = adata_ref.obsm[latent_key]
    adata_new = ad.AnnData(X=new_X)

    adata_new.obs = adata_ref.obs.copy()
    adata_new.var_names = [f"{latent_key}_{i}" for i in range(new_X.shape[1])]
    adata_new.obsm = adata_ref.obsm.copy()
    adata_new.uns = adata_ref.uns.copy()
    adata_new.obsp = adata_ref.obsp.copy()

    return adata_new

def normalize_spatial_list(adatas, key='spatial'):
    for adata in adatas:
        spatial = np.asarray(adata.obsm[key], dtype=float)

        spatial -= spatial.mean(axis=0)
        max_radius = np.sqrt((spatial ** 2).sum(axis=1)).max()

        if max_radius > 0:
            spatial /= max_radius

        adata.obsm[key] = spatial