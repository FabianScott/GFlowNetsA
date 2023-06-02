import os
print(os.getcwd()) # You should be running this from the GFlowNetsA directory.
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
try:
    # from GraphClustering import GraphNet
    print("All my friends hate hate Core GraphNet")
    from GraphClustering.Network.Network import GraphNet # This worries me. There is an outdated version of Graphnet in Core.
except: # Do not change this if it is unnecessary for you. Directly picking the cwd for jupyter notebooks can be a massive hassle in VSCode.
    print("Previous import statement didn't work. Changing cwd to parent directory.") #
    for _ in range(4):
        print("Stepping one directory up.")
        try:
            os.chdir("..")
            print(os.getcwd())
            from GraphClustering import GraphNet
            print("Successful import.")
            break
        except:
            pass
from GraphClustering import IRM_graph, clusterIndex
from GraphClustering import Cmatrix_to_array, torch_posterior
from GraphClustering.IRM_post import *

def gibbsSampler(N, graph, alpha):
    n_nodes = graph.shape[0]
    clusters = []
    for i in range(N):
        new_cluster = []
        permutation = np.random.permutation(n_nodes)
        for j, node in enumerate(permutation):
            # Eq 33
            prior = np.array([new_cluster.count(cluster) for cluster in set(new_cluster)]+[alpha])
            # Eq 34
            likelihood = Gibbs_likelihood(log = False)

            post = prior * likelihood
            post = post / sum(post)
            cum_post = np.cumsum(post)
            new_assign = sum(np.random.rand > cum_post)
            new_cluster.append(new_assign)

        clusters.append(np.zeros(n_nodes))
        for i, c in zip(permutation, new_cluster):
            clusters[-1][i] = c

if __name__ == '__main__':
    G = nx.karate_club_graph()
    Adj_karate = nx.adjacency_matrix(G).todense()
    Adj_karate = Adj_karate > 0
    graph = Adj_karate
    gibbsSampler(1, graph, 2)
