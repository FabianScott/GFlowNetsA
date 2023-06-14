import os
import numpy as np
import torch
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import sys
from GraphClustering import GraphNet
from GraphClustering import IRM_graph, clusterIndex
from GraphClustering import Cmatrix_to_array, torch_posterior
# from GraphClustering.IRM_post import Gibbs_sample_np, Gibbs_sample_torch
from GraphClustering.Network.Full_Distribution import get_clustering_matrix, get_clustering_list, allPermutations, allPosteriors, create_graph, scramble_graph, fix_net_clusters #, clusters_all_index


if __name__ == '__main__':
    N = 3
    a, b, A_alpha = 1, 1, 1
    log = False
    A_adj = np.array([[0,0,1],[0,0,1],[1,1,0]])
    A_adj_torch = torch.from_numpy(A_adj).float()
    plt.figure()
    plt.imshow(A_adj)
    plt.show()

    clusters_all = allPermutations(N)
    cluster_post = allPosteriors(A_adj_torch, a, b, A_alpha, log, joint = False)
    
    sort_idx = np.argsort(cluster_post)
    # Results
    top = 10
    if top:
        print("Total possible clusterings: "+str(len(sort_idx)))
        # print("Ground truth: "+str((cluster_random-1).tolist()))
        print("Top clusterings:")
        for i, idx in enumerate(np.flip(sort_idx)[:top]):
            print(str(i+1)+": "+str(clusters_all[idx])+" - " + str(cluster_post[idx]))