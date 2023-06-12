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
from GraphClustering.IRM_post import Gibbs_sample_np

if __name__ == '__main__':
    G = nx.karate_club_graph()
    Adj_karate = nx.adjacency_matrix(G).todense()
    Adj_karate = Adj_karate > 0
    graph = np.array(Adj_karate)
    T = 2000
    clusters_sampled = Gibbs_sample_np(graph, T, burn_in_buffer = None, sample_interval = None, seed = 42, a = 1, b = 1, A_alpha = 1, return_clustering_matrix = True)
    print(clusters_sampled)
