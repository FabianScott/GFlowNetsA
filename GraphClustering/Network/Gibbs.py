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

def gibbsSampler(N, graph, alpha):

    clusters = [0]
    for i in range(N)
        p_zk = np.array([clusters.count(cluster) for cluster in set(clusters)]+[alpha])
        p_zk /= sum(p_zk)

        

if __name__ == '__main__':

