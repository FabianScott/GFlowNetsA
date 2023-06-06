from GraphClustering.Core.Core import compare_results_small_graphs

from GraphClustering.Core.Core import GraphNet, torch_posterior, check_gpu
from copy import deepcopy
import networkx as nx
import pandas as pd
import numpy as np
import torch
if __name__ == '__main__':
    N = 3
    compare_results_small_graphs(filename=f'Comparison{N}.txt', min_N=N, max_N=N)


def train_and_save(net, X1, Adj_karate, epoch_interval, n_samples, array1, array2, df1, df2, filename1, filename2, header, i):
    net.train(X1, epochs=epoch_interval)  # Train an extra epoch interval
    X2 = net.sample_forward(Adj_karate, n_samples=n_samples, timer=True)

    net_values1, IRM_values1 = [], []

    for state in X1:
        net_values1.append(net.predict(state))
        adjacency_matrix, clustering_matrix = net.get_matrices_from_state(state)
        clustering_list, _ = net.get_clustering_list(clustering_matrix)
        IRM_values1.append(torch_posterior(adjacency_matrix, clustering_list - 1))

    difference1 = sum(abs(torch.tensor(net_values1) - torch.tensor(IRM_values1)))

    net_values2, IRM_values2 = [], []
    for state in X2:
        # net_values2.append(net.predict(state))
        adjacency_matrix, clustering_matrix = net.get_matrices_from_state(state)
        clustering_list, _ = net.get_clustering_list(clustering_matrix)
        IRM_values2.append(torch_posterior(adjacency_matrix, clustering_list - 1))

    IRM_sum = sum(IRM_values2)  # Tracking the amount of flow the network samples

    # Save the numbers every iteration
    array1[i] = difference1
    array2[i] = IRM_sum

    df1 = pd.concat((pd.DataFrame(header), pd.DataFrame(array1)), axis=1)
    df2 = pd.concat((pd.DataFrame(header), pd.DataFrame(array2)), axis=1)

    df1.to_csv(filename1, index=False, header=False)
    df2.to_csv(filename2, index=False, header=False)

    # Use the previous samples for the next iteration
    return X2
