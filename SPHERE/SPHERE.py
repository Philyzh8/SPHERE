import torch
from tqdm import tqdm
import torch.nn.functional as F
from SPHERE.model import *
from SPHERE.preprocess import *
import numpy as np
import matplotlib.pyplot as plt


class SPHERE:
    def __init__(self, 
        data,
        device= torch.device('cpu'),
        random_seed = 2022,
        learning_rate=0.0008,
        pre_learning_rate=0.0001,
        weight_decay=0.00,
        epochs=1000, 
        pre_epochs=1000,
        dim_input=200,
        dim_hid = 128,
        dim_output=32,
        data_sc=None,
        deconvolution = False,
        integrate = False,
        alignment = False,
        scale = False,
        pretrain = True,
        batch_label = None,
        slice_name_list = None,
        lambda_fea_recon = None,
        lambda_spa_recon = None,
        lambda_recon = None,
        lambda_con = None,
        lambda_recon_pre = None,
        lambda_con_pre = None,
        lambda_kl = None,
        lambda_align = None,
        lambda_latent = None,
        rwrIter = None,
        rwIter = None,
        inIter = None,
        outIter = None,
        alpha = None,
        beta = None,
        gamma = None,
        l1 = None,
        l2 = None,        
        l3 = None,
        l4 = None,
        ):

        self.data = data.copy()
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.pre_learning_rate=pre_learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.pre_epochs=pre_epochs
        self.dim_input = dim_input
        self.dim_hid = dim_hid
        self.dim_output = dim_output
        self.pretrain = pretrain
        
        # parameter
        self.lambda_fea_recon = lambda_fea_recon
        self.lambda_spa_recon = lambda_spa_recon
        self.lambda_recon = lambda_recon
        self.lambda_con = lambda_con
        self.lambda_recon_pre = lambda_recon_pre
        self.lambda_con_pre = lambda_con_pre
        self.lambda_kl = lambda_kl
        self.lambda_align = lambda_align
        self.lambda_latent = lambda_latent

        # align
        self.rwrIter = rwrIter
        self.rwIter = rwIter
        self.inIter = inIter
        self.outIter = outIter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1 = l1
        self.l2 = l2      
        self.l3 = l3
        self.l4 = l4

        # dataset
        self.adata = self.data

        if scale:
            self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
            self.dim_input = self.features.shape[1]
            self.n_cell = self.adata.n_obs
            adj_data = adjacent_matrix_preprocessing_scale(self.adata)
            self.adj_spatial = adj_data['adj_spatial'].to(self.device)
            self.adj_feature = adj_data['adj_feature'].to(self.device)
            self.adj_combined = adj_data['adj_combined'].to(self.device)

            self.batch_labels = torch.FloatTensor(batch_label).to(self.device)
            self.n_cls = batch_label.shape[1]
            self.slice_index = np.zeros((self.n_cell, ))
            slice_name_list = slice_name_list
            self.indices = [0]
            idx = 0
            for i in range(self.n_cls):
                idx += self.adata[self.adata.obs['slice_name']==slice_name_list[i]].shape[0]
                self.indices.append(idx)
                self.slice_index[self.indices[-2]:self.indices[-1]] = i

        else:
        
            if not alignment:
                self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
                self.n_cell = self.adata.n_obs
                self.dim_input = self.features.shape[1]


                # task
                if not deconvolution:
                    self.adj = adjacent_matrix_preprocessing(self.adata)
                    self.adj_spatial = self.adj['adj_spatial'].to(self.device)
                    self.adj_feature = self.adj['adj_feature'].to(self.device)
                    self.adj_combined = self.adj['adj_combined'].to(self.device)

                    if integrate:
                        self.batch_labels = torch.FloatTensor(batch_label).to(self.device)
                        self.n_cls = batch_label.shape[1]
                        self.slice_index = np.zeros((self.n_cell, ))
                        slice_name_list = slice_name_list
                        self.indices = [0]
                        idx = 0
                        for i in range(self.n_cls):
                            idx += self.adata[self.adata.obs['slice_name']==slice_name_list[i]].shape[0]
                            self.indices.append(idx)
                            self.slice_index[self.indices[-2]:self.indices[-1]] = i

                else:
                    self.adata_sc = data_sc.copy()
                    self.adj = adjacent_matrix_preprocessing_decon(self.adata, self.adata_sc)
                    self.adj_spatial = self.adj['adj_spatial'].to(self.device)
                    self.adj_feature = self.adj['adj_feature'].to(self.device)

                    self.features_sc = torch.FloatTensor(self.adata_sc.obsm['feat'].copy()).to(self.device)
                    self.labels_sc = torch.FloatTensor(self.adata_sc.obsm['label'].values.astype(np.float32)).to(self.device)
                    self.clusters = np.array(self.adata_sc.obsm['label'].columns)
                    self.celltype_dims = len(self.clusters)

    def train(self):
        self.model = AttnAE(self.dim_input, self.dim_hid, self.dim_output, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(self.features, self.adj_spatial, self.adj_feature,self.adj_combined)
            
            self.loss_recon = F.mse_loss(self.features, results['recon'])
            self.loss_recon += self.lambda_fea_recon * F.mse_loss(self.adj_feature.to_dense(), results['feature_rec']) + self.lambda_spa_recon * F.mse_loss(self.adj_spatial.to_dense(), results['spatial_rec']) 
            self.loss_con = self.correlation_reduction_loss(self.cross_correlation(results['latent_spatial'], results['latent_feature']))
            loss = self.lambda_recon*self.loss_recon + self.lambda_con*self.loss_con

            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Model training finished!\n")    
    
        with torch.no_grad():
            self.model.eval()
            results = self.model(self.features, self.adj_spatial, self.adj_feature,self.adj_combined) 
        emb_latent = F.normalize(results['latent'], p=2, eps=1e-12, dim=1)        
        output = {'latent': emb_latent.detach().cpu().numpy()}
        
        return output
    

    def train_inte(self):
        self.model = AttnAE(self.dim_input, self.dim_hid, self.dim_output, self.device, dropout=0.0, add_act=True, inte=True).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        self.optimizer_pre = torch.optim.Adam(self.model.parameters(), self.pre_learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model.train()

        if self.pretrain:
            for epoch in tqdm(range(self.pre_epochs)):
                self.model.train()
                results = self.model(self.features, self.adj_spatial, self.adj_feature, self.adj_combined)
                self.loss_recon = F.mse_loss(self.features, results['recon'])
                self.loss_recon += self.lambda_fea_recon*F.mse_loss(self.adj_feature.to_dense(), results['feature_rec']) + self.lambda_spa_recon * F.mse_loss(self.adj_spatial.to_dense(), results['spatial_rec'])
                self.loss_con = self.correlation_reduction_loss(self.cross_correlation(results['latent_spatial'], results['latent_feature']))
                self.loss_align = self.mmd_loss(results['latent'], self.batch_labels)
                loss = self.lambda_recon_pre*self.loss_recon + self.lambda_con_pre*self.loss_con + 0.5*self.loss_align
                self.optimizer_pre.zero_grad()
                loss.backward() 
                self.optimizer_pre.step()

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            cls_for_slices = [random.choice(random.sample([i for i in range(self.n_cls) if i != j], self.n_cls - 1))
                    for j in range(self.n_cls)]
            results = self.model(self.features, self.adj_spatial, self.adj_feature,self.adj_combined)

            with torch.no_grad():
                knn_indices_list, random_indices_list = self.find_knn(self.indices, cls_for_slices, results['latent'])
            dis = self.calculate_distance(self.indices, cls_for_slices, results['latent'], knn_indices_list, random_indices_list)
   
            self.loss_recon = F.mse_loss(self.features, results['recon'])
            self.loss_recon += self.lambda_fea_recon * F.mse_loss(self.adj_feature.to_dense(), results['feature_rec'])+ self.lambda_spa_recon * F.mse_loss(self.adj_spatial.to_dense(), results['spatial_rec'])
            self.loss_con = self.correlation_reduction_loss(self.cross_correlation(results['latent_spatial'], results['latent_feature']))
            self.loss_latent = self.latent_loss(dis, 0, margin=None)
            self.loss_align = self.mmd_loss(results['latent'], self.batch_labels)
            loss = self.lambda_recon*self.loss_recon + self.lambda_con*self.loss_con + self.lambda_align*self.loss_align + self.lambda_latent*self.loss_latent

            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Model training finished!\n")    

        with torch.no_grad():
            self.model.eval()
            results = self.model(self.features, self.adj_spatial, self.adj_feature,self.adj_combined) 
        emb_latent = F.normalize(results['latent'], p=2, eps=1e-12, dim=1)        
        output = {'latent': emb_latent.detach().cpu().numpy()}
        
        return output



    def train_decon(self):
        self.model = Encoder_decon(self.dim_input, self.dim_hid, self.dim_output, self.celltype_dims, self.device, dropout=0.6).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                    weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            self.optimizer.zero_grad()
            results = self.model(self.features, self.features_sc, self.adj_spatial, self.adj_feature)

            self.infer_loss = self.lambda_recon*F.mse_loss(results['pred_sc'], self.labels_sc) + self.lambda_kl*F.kl_div(self.labels_sc, results['pred_sc'])
            self.rec_loss  = self.lambda_spa_recon*F.mse_loss(results['spatial_rec'], self.features) 
            loss = self.infer_loss + self.rec_loss 
            loss += self.lambda_fea_recon*F.mse_loss(self.adj_spatial.to_dense(), results['spatial_graph_rec'])

            A_feat = self.adj_feature.to_dense()            
            D_feat = torch.diag(A_feat.sum(1))
            L_feat = D_feat - A_feat                      
            smooth_loss = torch.trace(results['pred_sc'].T @ L_feat @ results['pred_sc']) / results['pred_sc'].size(0)
            loss += 0.4*smooth_loss 

            loss.backward()
            self.optimizer.step()

        print("Model training finished!\n")    
    
        with torch.no_grad():
            self.model.eval()
            latent_spatial, _ = self.model.encoder(self.features, self.adj_spatial)
            pre, _ = self.model.predmodel(latent_spatial)
            pre = pre.cpu().detach().numpy()
            pre[pre < 0.01] = 0
            pre = pd.DataFrame(pre,columns=self.clusters,index=self.adata.obs_names)
            self.adata.obs[pre.columns] = pre.values
            self.adata.uns['celltypes'] = list(pre.columns)

        output = {'pre': pre}
        
        return output
    
    def train_align(self):
        R_list, T_list = [], []

        for i in range(len(self.adata) - 1):
            print(f'------ slice{i} and slice{i+1} -------')

            tgt_adata = self.adata[i].copy()
            src_adata = self.adata[i + 1].copy()

            Cal_Spatial_Net(tgt_adata, k_cutoff=1, model='KNN')
            Cal_Spatial_Net(src_adata, k_cutoff=1, model='KNN')

            adj1 = construct_interaction(tgt_adata, n_neighbors=300)
            adj2 = construct_interaction(src_adata, n_neighbors=300)

            _, features = Preprocess_without_SVD([tgt_adata, src_adata])
            tgt_adata.obsm['emb'], src_adata.obsm['emb'] = features

            H = self.find_best_matching(tgt_adata, src_adata)
            
            adj1 = torch.from_numpy(adj1)
            adj2 = torch.from_numpy(adj2)
            x1 = torch.from_numpy(features[0]).double()
            x2 = torch.from_numpy(features[1]).double()

            S, _, _ = align(
                adj1, adj2, x1, x2, H,
                self.rwrIter, self.rwIter,
                self.alpha, self.beta, self.gamma,
                self.inIter, self.outIter,
                self.l1, self.l2, self.l3, self.l4,
                device=self.device
            )

            best = np.argmax(S, axis=1)
            matching = np.vstack([np.arange(x1.shape[0]), best])

            R, T = find_rigid_transform(
                src_adata.obsm['spatial'][matching[1]],
                tgt_adata.obsm['spatial']
            )

            self.adata[i + 1].obsm['spatial'] = (
                self.adata[i + 1].obsm['spatial'] @ R.T + T
            )

            R_list.append(R)
            T_list.append(T)

            plt.figure(figsize=(5, 4))
            plt.scatter(
                self.adata[i + 1].obsm['spatial'][:, 0],
                -self.adata[i + 1].obsm['spatial'][:, 1],
                c='r', s=5, alpha=1
            )
            plt.scatter(
                self.adata[i].obsm['spatial'][:, 0],
                -self.adata[i].obsm['spatial'][:, 1],
                c='b', s=5, alpha=1
            )
            plt.axis('off')
            plt.show()

        return self.adata


    def cross_correlation(self, Z_v1, Z_v2):
        return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t()) 

    def correlation_reduction_loss(self, S):
        return torch.diagonal(S).add(-1).pow(2).mean() + self.off_diagonal(S).pow(2).mean()
    
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def find_knn(self, spot_idx_list, target_slice_list, latent_features, k=50):

        knn_indices_list = []
        random_indices_list = []

        for i in range(len(target_slice_list)):

            start_idx = spot_idx_list[i]
            end_idx = spot_idx_list[i + 1]
            target_start_idx = spot_idx_list[target_slice_list[i]]
            target_end_idx = spot_idx_list[target_slice_list[i] + 1]

            _, knn_indices = F.cosine_similarity(latent_features[start_idx:end_idx].unsqueeze(1),
                                                    latent_features[target_start_idx:target_end_idx].unsqueeze(0),
                                                    dim=2).topk(k, dim=1, largest=True)

            knn_indices_list.append(knn_indices)

            random_indices = torch.tensor(np.array([random.choices(range(0, end_idx - start_idx))
                                                    for _ in range(k)]).T[0]).long().to(self.device)
            random_indices_list.append(random_indices)

        return knn_indices_list, random_indices_list
    
    def calculate_distance(self, spot_idx_list, target_slice_list, latent_features,
                           knn_indices_list, random_indices_list):
        dis = []
        for i in range(len(target_slice_list)):

            start_idx = spot_idx_list[i]
            end_idx = spot_idx_list[i + 1]
            target_start_idx = spot_idx_list[target_slice_list[i]]
            target_end_idx = spot_idx_list[target_slice_list[i] + 1]

            pos = torch.norm(latent_features[start_idx:end_idx].unsqueeze(1) -
                               latent_features[target_start_idx:target_end_idx][knn_indices_list[i]].unsqueeze(0),
                               p=2, dim=3).squeeze(0)

            neg = torch.norm(latent_features[start_idx:end_idx].unsqueeze(1) -
                               latent_features[start_idx:end_idx][random_indices_list[i]].unsqueeze(0),
                               p=2, dim=2)

            dis.append(pos - neg)

        dis = torch.concat(dis, dim=0)
        return dis


    def latent_loss(self, dis, alpha, margin=None):
        if margin:
            loss = torch.mean(torch.relu(torch.clamp_max(dis, margin) + alpha))
        else:
            loss = torch.mean(torch.relu(dis + alpha))
        return loss

    def compute_pairwise_mmd_linear(self, x, y):
        def linear_kernel(x, y):
            return torch.mm(x, y.t())  

        k_xx = linear_kernel(x, x).mean()
        k_yy = linear_kernel(y, y).mean()
        k_xy = linear_kernel(x, y).mean()
        
        return k_xx + k_yy - 2 * k_xy  
    def mmd_loss(self, z, batch_labels):
        unique_batches = torch.arange(batch_labels.size(1), device=batch_labels.device)  
        batch_masks = batch_labels.bool()  

        total_mmd_loss = 0.0
        count = 0

        for i in range(len(unique_batches)):
            for j in range(i + 1, len(unique_batches)):
                z_i = z[batch_masks[:, i]]
                z_j = z[batch_masks[:, j]]
                mmd_ij = self.compute_pairwise_mmd_linear(z_i, z_j)
                
                total_mmd_loss += mmd_ij
                count += 1

        return total_mmd_loss / count if count > 0 else 0.0
    
    
    def find_best_matching(self, src, tgt, k_list=[3, 10, 40]):
        from scipy.sparse import issparse
        from scipy.spatial import cKDTree
        import ruptures as rpt

        kd_tree = cKDTree(src.obsm['spatial'])
        knn_src_exp_base = src.obsm['emb'].copy()
        knn_src_exp = src.obsm['emb'].copy()
        if issparse(knn_src_exp_base):
            knn_src_exp_base = knn_src_exp_base.todense()
        if issparse(knn_src_exp):
            knn_src_exp = knn_src_exp.todense()
        if len(k_list) != 0:
            for k in k_list:
                distances, indices = kd_tree.query(src.obsm['spatial'], k=k)  # (source_num_points, k)
                knn_src_exp = knn_src_exp + np.array(np.mean(knn_src_exp_base[indices, :], axis=1))

        kd_tree = cKDTree(tgt.obsm['spatial'])
        knn_tgt_exp = tgt.obsm['emb'].copy()
        knn_tgt_exp_base = tgt.obsm['emb'].copy()
        if issparse(knn_tgt_exp_base):
            knn_tgt_exp_base = knn_tgt_exp_base.todense()
        if issparse(knn_tgt_exp):
            knn_tgt_exp = knn_tgt_exp.todense()
        if len(k_list) != 0:
            for k in k_list:
                distances, indices = kd_tree.query(tgt.obsm['spatial'], k=k)  # (source_num_points, k)
                knn_tgt_exp = knn_tgt_exp + np.array(np.mean(knn_tgt_exp_base[indices, :], axis=1))

        corr = np.corrcoef(knn_src_exp, knn_tgt_exp)[:knn_src_exp.shape[0],
            knn_src_exp.shape[0]:]  # (src_points, tgt_points)

        src.obsm['emb'] = knn_src_exp
        tgt.obsm['emb'] = knn_tgt_exp

        ''' find the spots which are possibly in the overlap region by L1 changepoint detection '''
        y = np.sort(np.max(corr, axis=0))[::-1]
        data = np.array(y).reshape(-1, 1)
        algo = rpt.Dynp(model="l1").fit(data)
        result = algo.predict(n_bkps=1)
        first_inflection_point = result[0]

        ### set1: For each of point in tgt, the corresponding best matched point in src
        set1 = np.array([[index, value]for index, value in enumerate(np.argmax(corr, axis=0))])
        set1 = np.column_stack((set1,np.max(corr, axis=0)))
        set1 = pd.DataFrame(set1,columns = ['tgt_index','src_index','corr'])
        set1.sort_values(by='corr',ascending=False,inplace=True)
        set1 = set1.iloc[:first_inflection_point,:]


        y = np.sort(np.max(corr, axis=1))[::-1]
        data = np.array(y).reshape(-1, 1)
        algo = rpt.Dynp(model="l1").fit(data)
        result = algo.predict(n_bkps=1)
        first_inflection_point = result[0]

        ### set2: For each of point in src, the corresponding best matched point in tgt
        set2 = np.array([[index, value]for index, value in enumerate(np.argmax(corr, axis=1))])
        set2 = np.column_stack((set2,np.max(corr, axis=1)))
        set2 = pd.DataFrame(set2,columns = ['src_index','tgt_index','corr'])
        set2.sort_values(by='corr',ascending=False,inplace=True)
        set2 = set2.iloc[:first_inflection_point,:]


        result = pd.merge(set1, set2, left_on=['tgt_index', 'src_index'], right_on=['tgt_index', 'src_index'], how='inner')
        src_sub = src[result['src_index'].to_numpy().astype(int), :]
        tgt_sub = tgt[result['tgt_index'].to_numpy().astype(int), :]

        slice1_index =  result['src_index'].values.flatten().astype(int)
        slice2_index =  result['tgt_index'].values.flatten().astype(int)
        pair_index = [slice1_index,slice2_index]
        H = torch.zeros([src.shape[0],tgt.shape[0]],dtype=torch.int32)
        for i, j in zip(pair_index[0], pair_index[1]):
            H[i, j] = 1
        return H.T

    
    def train_inte_scale(self, batch_size=2048):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        import scipy.sparse as sp
        import numpy as np
        import gc

        def normalize_sub_adj(adj_tensor):
            adj = adj_tensor + torch.eye(adj_tensor.size(0)).to(adj_tensor.device)
            rowsum = adj.sum(1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        if not hasattr(self, 'model'):
            self.model = AttnAE_scale(self.dim_input, self.dim_hid, self.dim_output, self.device, 
                                      dropout=0.1, add_act=True, inte=True, scale=True).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        def to_scipy_cpu(adj):
            if torch.is_tensor(adj) and adj.is_sparse:
                adj = adj.coalesce().cpu()
                return sp.csr_matrix((adj.values().numpy(), adj.indices().numpy()), shape=adj.shape)
            return adj

        adj_s_csr = to_scipy_cpu(self.adj_spatial)
        adj_f_csr = to_scipy_cpu(self.adj_feature)
        
        sampler = StratifiedBatchSampler(self.indices, batch_size)
        loader = DataLoader(range(self.n_cell), batch_sampler=sampler)


        for epoch in range(self.epochs):
            self.model.train()
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch_indices in pbar:
                batch_indices_cpu = np.array(batch_indices)
                batch_tensor = torch.tensor(batch_indices_cpu).long().to(self.device)
                
                x_b_raw = self.features[batch_indices_cpu]
                if torch.is_tensor(x_b_raw):
                    x_b = x_b_raw.to(self.device).float()
                elif sp.issparse(x_b_raw):
                    x_b = torch.from_numpy(x_b_raw.toarray()).float().to(self.device)
                else:
                    x_b = torch.from_numpy(x_b_raw).float().to(self.device)

                sub_s_raw = torch.from_numpy(adj_s_csr[batch_indices_cpu, :][:, batch_indices_cpu].toarray()).float()
                sub_f_raw = torch.from_numpy(adj_f_csr[batch_indices_cpu, :][:, batch_indices_cpu].toarray()).float()
                sub_s_norm = normalize_sub_adj(sub_s_raw).to(self.device)
                sub_f_norm = normalize_sub_adj(sub_f_raw).to(self.device)
                
                z_s_b = self.model.encoder(x_b, sub_s_norm)
                z_f_b = self.model.encoder(x_b, sub_f_norm)
                z_b = self.model.atten(z_s_b, z_f_b, None)
                recon_b = self.model.decoder(z_b, sub_f_norm)

                labels_b = self.batch_labels[batch_tensor].argmax(dim=1) if self.batch_labels.dim() > 1 else self.batch_labels[batch_tensor]
                
                l_align = self.mmd_loss_scale(z_b, labels_b)
                
                with torch.no_grad():
                    z_b_norm = F.normalize(z_b, p=2, dim=1)
                    sim_m = torch.mm(z_b_norm, z_b_norm.t())
                    mask = (labels_b.unsqueeze(0) != labels_b.unsqueeze(1)).float()
                    sim_m = sim_m * mask - (1 - mask) * 1e9
                    _, knn_idx = sim_m.topk(min(20, sim_m.size(1)), dim=1)
                    rand_idx = torch.randint(0, z_b.size(0), (z_b.size(0), min(20, z_b.size(0)))).to(self.device)

                l_latent = torch.mean(F.relu(torch.norm(z_b.unsqueeze(1)-z_b[knn_idx], p=2, dim=2) - 
                                            torch.norm(z_b.unsqueeze(1)-z_b[rand_idx], p=2, dim=2) + 0.5))
                l_recon_feat = F.mse_loss(recon_b, x_b)
                rec_s, rec_f = self.model.encoder.dc(z_s_b), self.model.encoder.dc(z_f_b)
                l_graph = self.lambda_spa_recon * F.mse_loss(rec_s, sub_s_raw.to(self.device)) + \
                          self.lambda_fea_recon * F.mse_loss(rec_f, sub_f_raw.to(self.device))
                l_con = self.correlation_reduction_loss(self.cross_correlation(z_s_b, z_f_b))

                loss = self.lambda_recon * (l_recon_feat + l_graph) + self.lambda_con * l_con + \
                       self.lambda_align * l_align + self.lambda_latent * l_latent

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                del x_b, z_b, z_s_b, z_f_b, recon_b, sub_s_norm, sub_f_norm, loss
            
            torch.cuda.empty_cache()
            gc.collect()

        self.model.eval()
        all_l = []
        with torch.no_grad():
            for idxs in DataLoader(range(self.n_cell), batch_size=batch_size * 4):
                xi = self.features[idxs].to(self.device).float()
                ii_adj = torch.eye(xi.size(0)).to(self.device).to_sparse()
                zi = self.model.encoder(xi, ii_adj)
                if isinstance(zi, tuple): zi = zi[0]
                all_l.append(zi.cpu())
        
        return {'latent': F.normalize(torch.cat(all_l), p=2, dim=1).numpy()}
    
    def mmd_loss_scale(self, x, labels):
        unique_batches = torch.unique(labels)
        if len(unique_batches) < 2:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        loss = 0.0
        pair_count = 0
        centroids = []
        for b_id in unique_batches:
            mask = (labels == b_id)
            if mask.sum() > 1:
                centroids.append(x[mask].mean(0))
        
        if len(centroids) < 2:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
            
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                loss += torch.norm(centroids[i] - centroids[j], p=2)
                pair_count += 1
                
        return loss / pair_count
    
    def train_query_frozen(self, batch_size=2048, query_epochs=10):
        from torch.utils.data import DataLoader
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True 
                
        active_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(active_params, lr=self.learning_rate * 0.1)
        
        def to_scipy_local(adj):
            if torch.is_tensor(adj) and adj.is_sparse:
                adj = adj.coalesce()
                return sp.csr_matrix((adj.values().cpu().numpy(), adj.indices().cpu().numpy()), shape=adj.shape)
            return adj

        adj_s_csr = to_scipy_local(self.adj_spatial)
        adj_f_csr = to_scipy_local(self.adj_feature)

        sampler = StratifiedBatchSampler(self.indices, batch_size)
        loader = DataLoader(range(self.n_cell), batch_sampler=sampler)

        print(f"🔒 Architecture Lock active. Mapping query seamlessly...")
        for epoch in range(query_epochs):
            self.model.train()
            for name, param in self.model.named_parameters():
                if 'encoder' in name: param.requires_grad = False
                
            pbar = tqdm(loader, desc=f"Query Epoch {epoch+1}/{query_epochs}")
            for batch_indices in pbar:
                batch_indices_cpu = np.array(batch_indices)
                sub_s = adj_s_csr[batch_indices_cpu, :][:, batch_indices_cpu].toarray()
                sub_f = adj_f_csr[batch_indices_cpu, :][:, batch_indices_cpu].toarray()
                
                adj_s_true = torch.from_numpy(sub_s).float().to(self.device)
                adj_f_true = torch.from_numpy(sub_f).float().to(self.device)
                batch_tensor = torch.tensor(batch_indices_cpu).long().to(self.device)
                
                
                z_s_b = self.model.encoder(self.features[batch_tensor].to(self.device).float(), adj_s_true)
                z_f_b = self.model.encoder(self.features[batch_tensor].to(self.device).float(), adj_f_true)
                z_b = self.model.atten(z_s_b, z_f_b, None)

                recon_b = self.model.decoder(z_b, adj_f_true)
                
                labels_b = self.batch_labels[batch_tensor].argmax(dim=1) if self.batch_labels.dim() > 1 else self.batch_labels[batch_tensor]

                with torch.no_grad():
                    z_b_norm = F.normalize(z_b, p=2, dim=1)
                    sim_matrix = torch.mm(z_b_norm, z_b_norm.t())
                    mask_cross = (labels_b.unsqueeze(0) != labels_b.unsqueeze(1)).float()
                    sim_matrix_cross = sim_matrix * mask_cross - (1 - mask_cross) * 1e9
                    k_batch = min(20, sim_matrix_cross.size(1))
                    _, knn_idx_b = sim_matrix_cross.topk(k_batch, dim=1)
                    rand_idx_b = torch.randint(0, z_b.size(0), (z_b.size(0), k_batch)).to(self.device)

                l_latent = torch.mean(F.relu(torch.norm(z_b.unsqueeze(1) - z_b[knn_idx_b], p=2, dim=2) - 
                                        torch.norm(z_b.unsqueeze(1) - z_b[rand_idx_b], p=2, dim=2) + 0.5))
                # l_recon_feat = F.mse_loss(results['recon'][batch_tensor], self.features[batch_tensor])
                l_recon_feat = F.mse_loss(recon_b, self.features[batch_tensor].to(self.device).float())
                rec_s, rec_f = self.model.encoder.dc(z_s_b), self.model.encoder.dc(z_f_b)
                l_graph = self.lambda_spa_recon * F.mse_loss(rec_s, adj_s_true) + \
                        self.lambda_fea_recon * F.mse_loss(rec_f, adj_f_true)
                l_con = self.correlation_reduction_loss(self.cross_correlation(z_s_b, z_f_b))
                l_align = self.mmd_loss_scale(z_b, labels_b)

                loss = (self.lambda_recon * (l_recon_feat + l_graph) + self.lambda_con * l_con + 
                        self.lambda_align * l_align + self.lambda_latent * l_latent)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                del loss, adj_s_true, adj_f_true
                
            
        self.model.eval()
        all_l = []
        with torch.no_grad():
            for idxs in DataLoader(range(self.n_cell), batch_size=batch_size * 4):
                xi = self.features[idxs].to(self.device).float()
                ii_adj = torch.eye(xi.size(0)).to(self.device).to_sparse()
                zi = self.model.encoder(xi, ii_adj)
                if isinstance(zi, tuple): zi = zi[0]
                all_l.append(zi.cpu())
        
        return {'latent': F.normalize(torch.cat(all_l), p=2, dim=1).numpy()}

    
