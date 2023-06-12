from GraphClustering.Core.Core import GraphNet, torch_posterior
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch


def return_0():
    return 0


if __name__ == '__main__':
    net = GraphNet(n_nodes=34)
    fname = 'Data/KarateResults_100_500_10000_o_Samples_400.pt'
    state_tensor = net.load_samples(fname)

    cluster_lists = defaultdict(return_0)
    IRM_list = []
    for state in state_tensor:
        adj_mat, cluster_mat = net.get_matrices_from_state(state)
        cluster_list, _ = net.get_clustering_list(cluster_mat)
        IRM_value = int(torch_posterior(adj_mat, cluster_list - 1))
        IRM_list.append(IRM_value)
        cluster_lists[tuple(cluster_list)] += 1

    plt.plot(IRM_list, label='IRM Values')
    plt.plot(list(cluster_lists.values()), label='Cluster Count')
    plt.legend()
    plt.show()
