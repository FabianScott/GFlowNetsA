import os
# print(os.getcwd())  # You should be running this from the GFlowNetsA directory.
import numpy as np
import torch
from scipy.special import logsumexp
import matplotlib.pyplot as plt

try:
    from GraphClustering import GraphNet  # I created a launch.json to make this work for debugging sessions. It is infuriating that it doesn't work regularly.
except:  # Do not change this if it is unnecessary for you. Directly picking the cwd for jupyter notebooks can be a massive hassle in VSCode.
    import sys

    print("Appending to sys path")
    sys.path.append(os.getcwd())  # This is really ugly
    from GraphClustering import GraphNet

from GraphClustering import IRM_graph, clusterIndex
from GraphClustering import Cmatrix_to_array, torch_posterior
from GraphClustering.IRM_post import Gibbs_sample_np, Gibbs_sample_torch
from GraphClustering.Network.Full_Distribution import get_clustering_matrix, get_clustering_list, allPermutations, allPosteriors, create_graph, scramble_graph, fix_net_clusters #, clusters_all_index

def clusters_all_index_np(clusters_all, specific_cluster_list):
    cluster_ind = np.argwhere(np.all(np.equal(clusters_all, specific_cluster_list), axis=1) == 1)[0][0]
    return cluster_ind

if __name__ == '__main__':
    N = 5
    a, b, A_alpha = 1, 1, 5  # 10000
    log = True
    seed = 54
    plot = True
    adjacency_matrix, cluster_idxs, clusters = create_graph(N, a, b, A_alpha, log, seed)

    A_random, idxs, cluster_random, cluster_num = scramble_graph(adjacency_matrix, clustering_list=cluster_idxs,
                                                                 seed=42)
    inv_idx = np.argsort(idxs)

    clusters_all = allPermutations(N)
    cluster_post = allPosteriors(A_random, a, b, A_alpha, log, joint=False)
    sort_idx = np.argsort(cluster_post)

    # Sample Gibbs
    N_samples = 2000
    A_random_numpy = A_random.numpy()
    clusters_sampled = Gibbs_sample_torch(A_random, T = N_samples, burn_in_buffer = None, sample_interval = None, seed = seed, a = a, b = b, A_alpha = A_alpha, return_clustering_matrix = True)
    clusters_unscrambled_lists = [get_clustering_list(clustering)[0] for clustering in clusters_sampled]
    clusters_unscrambled_tensor = torch.vstack(tuple(clusters_unscrambled_lists))
    clusters_unscrambled_tensor_numpy = clusters_unscrambled_tensor.numpy()
    clusters_unique, clusters_count = np.unique(clusters_unscrambled_tensor_numpy, axis=0, return_counts=True)

    clusters_count_ordered = np.zeros(len(clusters_all))
    for c, n_k in zip(clusters_unique, clusters_count):
        clusters_count_ordered[clusters_all_index_np(clusters_all+1, c)] = n_k
    # assert np.all(clusters_count == clusters_count_ordered) # Often false now. Depends on when we spot the different clusterings.

    gibbs_post = np.log(clusters_count_ordered / N_samples)

    plt.plot(cluster_post[sort_idx], "bo")
    plt.plot(gibbs_post[sort_idx], "rx")
    plt.show()





