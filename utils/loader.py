import torch  
import random
from torch.utils.data import Dataset  
import numpy as np  
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
import numpy as np  

def build_knn_adj(X, k):  
    # 初始化一个NearestNeighbors对象  
    knn = NearestNeighbors(n_neighbors=k+1,  # 包含自己本身作为最近邻之一  
                           metric='euclidean',  # 使用欧几里得距离  
                           algorithm='auto')  
      
    # 拟合模型并找到每个点的k个最近邻  
    knn.fit(X)  
    distances, indices = knn.kneighbors(X)  
    return distances, indices

def data_collate_fn(examples: list):
    

    # Group by tensor type
    idx, x_list, adj_list, y_list = zip(*examples)
    id1, index = np.unique(np.concatenate(adj_list), return_index=True)
    mapping = dict(zip(id1, range(len(id1))))
    batch_id = [mapping[i] for i in idx]
    batch_id = torch.tensor(batch_id).long()
    print(batch_id)
    batch_x = torch.cat(x_list, dim=0)[index, :]
    
    batch_y = np.vectorize(mapping.get)(np.vstack(adj_list)) 
    edge_x = torch.repeat_interleave(batch_id, batch_y.shape[1])
    edge_y = torch.from_numpy(batch_y).long().view(-1)
    batch_adj = torch.vstack([edge_x, edge_y])

    batch_y = torch.cat(y_list)[index]
    return batch_id, batch_x, batch_adj, batch_y

class StDataset(Dataset):  
    def __init__(self, adata, graph, labels):  
        """  
        初始化Dataset，设置整数范围和样本大小  
        """  
        self.x = torch.from_numpy(adata.X.A).float()
        self.adj = graph
        clusters, labels = np.unique(labels, return_inverse=True)
        self.out = torch.tensor(labels).long()
        
    def __len__(self):  
        """  
        返回数据集的大小  
        """  
        return self.x.shape[0]
  
    def __getitem__(self, idx):  
        """  
        根据索引返回单个样本  
        """  
        return idx, self.x[self.adj[idx, :],:], self.adj[idx, :], self.out[self.adj[idx, :]]

    def get_batch(self, index):
        """
        Get a batch of data according to given index
        """
        data_batch = self.x[index,:]
        knn_batch = self.adj[index][:, index]
        y_batch = self.out[index]
        return data_batch, knn_batch, y_batch

def Ripplewalk_sampler(graph, r=0.5, batchsize=64, total_times=10):
    """
    Training stategy of subgraph segmentation based on random walk, enabling mini-batch training on large datasets
    
    Parameters
    ------
    graph
        graph indicating connectivity between spots, usually K-NN graph constructed from spatial coordinates
    r
        expansion ratio for sampling subgraph
    batchsize
        number of samples for a mini-batch
    total_times
        decide the number of subgraph to sample, number of subgraph = total_times * dataset_size / batchsize
        
    Returns
    ------
    list
        a list containing index of sampled sub-graphs
    """
    if not isinstance(graph, sp.coo_matrix):
        graph = sp.coo_matrix(graph)
    num_nodes = graph.shape[0]
    number_subgraph = (num_nodes * total_times) // batchsize + 1

    if batchsize >= num_nodes:
        print("This dataset is smaller than batchsize so that ripple walk sampler is not used!")
        subgraph_set = []
        for i in range(total_times):
            index_list = [j for j in range(num_nodes)]
            random.shuffle(index_list)
            subgraph_set.append(index_list)
        return subgraph_set

    # transform adj to index
    final = []
    for i in range(num_nodes):
        final.append(graph.col[graph.row == i].tolist())
    graph = final

    # Ripplewalk sampling
    subgraph_set = []
    for i in range(number_subgraph):
        # select initial node, and store it in the index_subgraph list
        index_subgraph = [np.random.randint(0, num_nodes)]
        # the neighbor node set of the initial nodes
        neighbors = graph[index_subgraph[0]]
        len_subgraph = 1
        while (1):
            len_neighbors = len(neighbors)
            if (len_neighbors == 0):  # getting stuck in the inconnected graph, select restart node
                while (1):
                    restart_node = np.random.randint(0, num_nodes)
                    if (restart_node not in index_subgraph):
                        break
                index_subgraph.append(restart_node)
                neighbors = neighbors + graph[restart_node]
                neighbors = list(set(neighbors) - set(index_subgraph))
                len_subgraph = len(index_subgraph)
            else:
                # select part (half) of the neighbor nodes and insert them into the current subgraph
                if ((batchsize - len_subgraph) > (len_neighbors * r)):  # judge if we need to select that much neighbors
                    neig_random = random.sample(neighbors, max(1, int(r * len_neighbors)))
                    neighbors = list(set(neighbors) - set(neig_random))

                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))
                    for i in neig_random:
                        neighbors = neighbors + graph[i]
                    neighbors = list(set(neighbors) - set(index_subgraph))
                    len_subgraph = len(index_subgraph)
                else:
                    neig_random = random.sample(neighbors, (batchsize - len_subgraph))
                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))
                    break
        subgraph_set.append(index_subgraph)
    return subgraph_set

def Ripplewalk_prediction(graph, r=0.5, batchsize=6400):
    """
    Prediction stategy of subgraph segmentation based on random walk, enabling mini-batch prediction on large datasets
    
    Parameters
    ------
    graph
        graph indicating connectivity between spots, usually K-NN graph constructed from spatial coordinates
    r
        expansion ratio for sampling subgraph
    batchsize
        number of samples for a mini-batch
        
    Returns
    ------
    list
        a list containing index of sampled sub-graphs
    """
    
    if not isinstance(graph, sp.coo_matrix):
        graph = sp.coo_matrix(graph)
    num_nodes = graph.shape[0]
    
    if batchsize >= num_nodes:
        subgraph_set = []
        index_list = np.array([j for j in range(num_nodes)])
        subgraph_set.append(index_list)
        return subgraph_set
    
    # transform adj to index
    final = []
    for i in range(num_nodes):
        final.append(graph.col[graph.row==i].tolist())
    graph = final
    
    all_nodes = [j for j in range(num_nodes)]
    sampled_nodes = []
    
    # Ripplewalk sampling
    subgraph_set = []
    
    while len(all_nodes)>len(sampled_nodes):
        # select initial node from non-sampled nodes
        nonsampled_nodes = np.setdiff1d(all_nodes, sampled_nodes)
        random.shuffle(nonsampled_nodes)
        cur_node = nonsampled_nodes[0]
        index_subgraph = [cur_node]
        
        #the neighbor node set of the initial nodes
        neighbors = graph[index_subgraph[0]]
        sampled_nodes.append(cur_node)
        len_subgraph = 1
        
        while(1):
            len_neighbors = len(neighbors)
            if(len_neighbors == 0): # getting stuck in the inconnected graph, select restart node
                nonsampled_nodes = np.setdiff1d(all_nodes, sampled_nodes)
                random.shuffle(nonsampled_nodes)
                restart_node = nonsampled_nodes[0]
                index_subgraph.append(restart_node)
                
                neighbors = neighbors + graph[restart_node]
                neighbors = list(set(neighbors) - set(index_subgraph))
                sampled_nodes.append(restart_node)
                len_subgraph = len(index_subgraph)
            else: # select part (half) of the neighbor nodes and insert them into the current subgraph
                if ((batchsize - len_subgraph) > (len_neighbors*r)): # judge if we need to select that much neighbors
                    neig_random = random.sample(neighbors, max(1, int(r*len_neighbors)))
                    neighbors = list(set(neighbors) - set(neig_random))

                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))
                    
                    for i in neig_random:
                        neighbors = neighbors + graph[i]
                    neighbors = list(set(neighbors) - set(index_subgraph))
                    sampled_nodes = np.union1d(sampled_nodes, neig_random).tolist()
                    len_subgraph = len(index_subgraph)
                else:
                    neig_random = random.sample(neighbors, (batchsize - len_subgraph))
                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))
                    sampled_nodes = np.union1d(sampled_nodes, neig_random).tolist()
                    break
        subgraph_set.append(np.array(index_subgraph))
    return subgraph_set