import os
print(os.getcwd()) # You should be running this from the GFlowNetsA directory.
import numpy as np
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

def count_links(node_idx, adjacency_matrix, cluster_idxs):

    return m_bar, r_bar

def gibbsSampler(N, graph, alpha):
    n_nodes = len(graph[0])
    clusters = []
    for i in range(N):
        clusters.append([0])
        for j, node in enumerate(np.permutation(n_nodes)):
            p_zk = np.array([clusters[-1].count(cluster) for cluster in set(clusters[-1])]+[alpha])
            p_zk /= sum(p_zk)

            []







if __name__ == '__main__':

