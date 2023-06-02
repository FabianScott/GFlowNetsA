import os
print(os.getcwd()) # You should be running this from the GFlowNetsA directory.
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
try:
    from GraphClustering import GraphNet # I created a launch.json to make this work for debugging sessions. It is infuriating that it doesn't work regularly. 
except: # Do not change this if it is unnecessary for you. Directly picking the cwd for jupyter notebooks can be a massive hassle in VSCode.
    import sys
    print("Appending to sys path")
    sys.path.append(os.getcwd()) # This is really ugly
    from GraphClustering import GraphNet
from GraphClustering import IRM_graph, clusterIndex
from GraphClustering import Cmatrix_to_array, torch_posterior
from GraphClustering.IRM_post import Gibbs_likelihood

def gibbsSampler(N, graph, alpha):
    n_nodes = graph.shape[0]
    clusters = []
    for i in range(N):
        new_cluster = np.array([])
        permutation = np.random.permutation(n_nodes)
        perm_graph = graph[permutation][:, permutation]
        for node in permutation:
            # Eq 33
            _, counts = np.unique(new_cluster, return_counts=True)
            prior = np.concatenate((counts, np.array([alpha])))
            # Eq 34
            likelihood = Gibbs_likelihood(perm_graph, new_cluster, log = False)
            likelihood = np.array(likelihood)[0]

            post = prior * likelihood
            post = post / sum(post)
            cum_post = np.cumsum(post)
            new_assign = int(sum(np.random.rand() > cum_post))
            new_cluster = np.append(new_cluster, new_assign)

        clusters.append(np.zeros(n_nodes))
        for i, c in zip(permutation, new_cluster):
            clusters[-1][i] = c

    return clusters

if __name__ == '__main__':
    G = nx.karate_club_graph()
    Adj_karate = nx.adjacency_matrix(G).todense()
    Adj_karate = Adj_karate > 0
    graph = Adj_karate
    clusters_sampled = gibbsSampler(100, graph, 5)
