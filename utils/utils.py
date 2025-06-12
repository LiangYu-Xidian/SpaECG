import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as sm


def calCorr(data, method='pearson'):
    # corr = data.corr(method=method)
    X = data.values

    X = (X-np.mean(X,axis=0))/np.std(X,axis=0)

    r = np.dot(X.T,X)/X.shape[0]
    r = pd.DataFrame(r, columns = data.columns, index = data.columns)
    return r

def getTopCorrPairs(corr, cutoff=None, thre_cor = 0.1):

    if cutoff==None:
        thre_cor = thre_cor
    else:
        values = corr.abs().values
        values = values[values > 0]
        values = np.sort(values)[::-1]
        thre_cor = values[int(len(values) * cutoff)]
    print(thre_cor)
    row, col = np.where(corr.abs().values > thre_cor)
    
    nodes = corr.columns.values
    topCorrPairs = pd.DataFrame({"src": nodes[row], "dest": nodes[col]})
    return topCorrPairs


def findSigGenes(gene_importance, labels, genes_interest):
    
    num_genes = len(genes_interest)
    important_genes_cluster = dict()

    # 遍历每种细胞类型
    for cluster in gene_importance.keys():
        # 1. 筛选出目标细胞类型和其他细胞类型的索引
        gene_important_df = gene_importance[cluster]
        target_indices = (labels == cluster)
        other_indices = ~target_indices

        # 2. 提取目标细胞类型和其他细胞类型的基因重要性分数
        target_scores = gene_important_df[target_indices, :]
        other_scores = gene_important_df[other_indices, :]

        # 4. 统计检验法
        p_values = []
        for gene_idx in range(num_genes):
            # 对目标细胞类型和其他细胞类型的基因分数进行统计检验
            # t_stat, p_val = ttest_ind(target_scores[:, gene_idx], other_scores[:, gene_idx], equal_var=False, alternative='two-sided')
            # 使用 Mann-Whitney U 检验：
            u_stat, p_val = mannwhitneyu(target_scores[:, gene_idx], other_scores[:, gene_idx], alternative='greater')
            p_values.append(p_val)
        reject, pvals_adj = sm.fdrcorrection(p_values, alpha=0.05, method='indep', is_sorted=False)

        # 筛选显著基因
        significant_gene_indices = np.where(pvals_adj < 0.05)[0]  # 选择 p 值小于 0.05 的基因
        important_genes_cluster[cluster] = pd.DataFrame({
            "significant_genes": genes_interest[significant_gene_indices],  # 显著基因索引
            "p_values": pvals_adj[significant_gene_indices]  # 显著基因对应的校正 p 值
        }).sort_values(by='p_values')
    
    return important_genes_cluster
