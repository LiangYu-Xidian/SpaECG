import scipy
import pandas as pd
from scipy.spatial.distance import cdist,squareform, pdist
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
def null_test_lr(gene_emb_df, embedding_dist_df, pval=0.05,faked_pairs=200):

    dist_test = embedding_dist_df
    for index, row in dist_test.iterrows():
        ligand = dist_test.loc[index,"ligand"]
        receptor = dist_test.loc[index,"receptor"]
        dist = dist_test.loc[index,"dist"]
        
        ligand_embedding = gene_emb_df.loc[ligand].to_numpy().reshape(1,-1)
        filtered_df = gene_emb_df[gene_emb_df.index != receptor]
        filtered_df = filtered_df.sample(n=faked_pairs, replace=False)
        pair=[]

        for index2, row2 in filtered_df.iterrows():
            tmp2_embedding = gene_emb_df.loc[index2].to_numpy().reshape(1,-1)
            pair.append(cosine_distances(ligand_embedding, tmp2_embedding)[0][0])
            pair.append(cdist(ligand_embedding, tmp2_embedding, metric='cosine')[0][0])
        dist_test.loc[index,"p_val"]=scipy.stats.percentileofscore(pair, dist) / 100
            
    df_enriched = dist_test[dist_test['p_val'] < pval].sort_values(by=['p_val'])

    return df_enriched


def find_sig_lr(gene_embedding_df, lr_db):
    df = pd.DataFrame(columns=['ligand', 'receptor', 'dist'])
    #計算篩選之後的LR的欧式距离
    for i in lr_db.index:
        gene1 = lr_db.loc[i,"ligand"]
        gene2 = lr_db.loc[i,"receptor"]
        
        gene_embedding1 = gene_embedding_df.loc[gene1,:].to_numpy().reshape(1,-1)
        gene_embedding2 = gene_embedding_df.loc[gene2,:].to_numpy().reshape(1,-1)
        dist =cdist(gene_embedding1, gene_embedding2, metric='cosine')[0][0]
        df.loc[gene1 + '_' + gene2]=[gene1, gene2, dist]

    lr_enriched = null_test_lr(gene_embedding_df, df, pval=0.05,faked_pairs=200)
    
    return lr_enriched


def default_w_visium(adata,
                     min_cell_distance: int = 100,
                     cover_distance: int = 255,
                     obsm_spatial_slot: str = 'spatial',
                     ):

    position_mat = adata.obsm[obsm_spatial_slot]
    dist_mat = squareform(pdist(position_mat, metric='euclidean'))

    # ligands cover 255 micrometers by default, and the min value of distance between spot is 100 micrometers
    w_best = cover_distance * (dist_mat[dist_mat > 0].min() / min_cell_distance) / np.sqrt(np.pi)

    return w_best

def dist_factor_calculate(adata, w_best, obsm_spatial_slot = 'spatial'):

    position_mat = adata.obsm[obsm_spatial_slot]
    dist_mat = squareform(pdist(position_mat, metric='euclidean'))
    dist_factor = dist_mat / w_best

    dist_factor = np.exp((-1) * dist_factor * dist_factor)
    dist_factor[dist_factor<1e-5] = 0
    return dist_factor

def cal_cci_score(adata, gene_importance, lr_enriched, maps):
    w_best = default_w_visium(adata)
    dist_factor_mat = dist_factor_calculate(adata, w_best=w_best )

    
    lr_genes = np.unique(lr_enriched[['ligand', 'receptor']].values)

    sig_score = np.zeros(shape=(adata.shape[0], len(lr_genes)))
    
    labels = np.array(adata.obs['celltype'])
    clusters = np.unique(labels)
    
    lr_genes_idx = np.vectorize(maps.get)(lr_genes)
    for cluster in clusters:
        target_indices = (labels == cluster)
        gene_importance_score = gene_importance[cluster][target_indices, :]
        gene_importance_score = gene_importance_score[:, lr_genes_idx]
        sig_score[target_indices, :] = gene_importance_score
        
    sig_score = pd.DataFrame(sig_score, columns=lr_genes)
    
    cci_matrix = np.zeros(( adata.shape[0],adata.shape[0],lr_enriched.shape[0]),dtype=np.float16)
    for index, (_, row) in enumerate(lr_enriched.iterrows()):
        l_data = sig_score.loc[:, row['ligand']].values
        r_data = sig_score.loc[:, row['receptor']].values
        cci_matrix[:,:, index] = (np.outer(l_data, r_data)*dist_factor_mat)
    return cci_matrix


def spatial_cell2cell(X, source_type, target_type, labels, lr_db, dist, shuffle=False):
    if shuffle:
        X = X.sample(frac=1)
    cci_matrix = np.zeros((len(source_type)* len(target_type), np.sum(len(lr_db))))

            
    l_data = X[lr_db['ligand']]
    r_data = X[lr_db['receptor']]
    for i in range(len(source_type)):
        gi = source_type[i]
        cells_in_gi = (labels==gi)
        t_l_data = l_data.loc[cells_in_gi, :].values
        for j in range(len(target_type)):
            gj = target_type[j]
            cells_in_gj = (labels==gj)
            t_r_data = r_data.loc[cells_in_gj, :].values

            t_dist = dist[cells_in_gj,:][:, cells_in_gi]
            tmp = np.sum(t_dist)
            if tmp == 0:
                break
            tmp = np.sum(np.dot(t_dist, t_l_data)*t_r_data, axis=0)/(t_dist.shape[0]*t_dist.shape[1])
           
            cci_matrix[i*len(target_type)+j, :] = tmp
    return cci_matrix


def getCelltypeInteraction(X, lr_db, source_types, target_types, labels, dist, iter_num=1000,thre_p=0.05):

    celltype_lr_interaction = spatial_cell2cell(X, source_types, target_types, labels, lr_db, dist)
    p_celltype_interaction = [spatial_cell2cell(X, source_types, target_types, labels, lr_db, dist, shuffle=True)
            for i in range(iter_num)]
    p_celltype_interaction = np.stack(p_celltype_interaction, axis=0)
    p_celltype_interaction = np.sum(p_celltype_interaction>celltype_lr_interaction,axis=0)/iter_num
    
    cellpairs = [s +"--"+t for s in source_types for t in target_types]
    celltype_lr_interaction = pd.DataFrame(celltype_lr_interaction, index=cellpairs, columns=lr_db.index)
    p_celltype_interaction = pd.DataFrame(p_celltype_interaction, index=cellpairs, columns=lr_db.index)
    ix, iy = np.where((p_celltype_interaction <= thre_p) & (celltype_lr_interaction > 0))
    celltype_interaction = pd.DataFrame({
        "interaction": np.array(cellpairs)[ix],
        "lr": lr_db.index[iy],
        "cci_score": celltype_lr_interaction.values[ix,iy],
        "p_val": p_celltype_interaction.values[ix,iy] 
    })
    return celltype_interaction