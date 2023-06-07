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
from GraphClustering.Network.Gibbs import gibbsSampler
from GraphClustering.Network.Full_Distribution import get_clustering_matrix, get_clustering_list, allPermutations, allPosteriors, create_graph, scramble_graph, fix_net_clusters, clusters_all_index

if __name__ == '__main__':
    N = 4
    a, b, A_alpha = 1, 1, 3  # 10000
    log = True
    seed = 47
    plot = True
    adjacency_matrix, cluster_idxs, clusters = create_graph(N, a, b, A_alpha, log, seed)

    A_random, idxs, cluster_random, cluster_num = scramble_graph(adjacency_matrix, clustering_list=cluster_idxs,
                                                                 seed=42)

    clusters_all = allPermutations(N)
    cluster_post = allPosteriors(A_random, a, b, A_alpha, log, joint=False)
    sort_idx = np.argsort(cluster_post)

    # Sample Gibbs
    N_samples = 2000
    A_random_numpy = A_random.numpy()
    clusters_sampled = gibbsSampler(N_samples, A_random_numpy, a, b, A_alpha)
    clusters_unique, clusters_count = np.unique(clusters_sampled, axis=0, return_counts=True)
    gibbs_post = np.log(clusters_count / N_samples)

    plt.plot(cluster_post[sort_idx], "bo")
    plt.plot(gibbs_post[sort_idx], "rx")
    plt.show()





