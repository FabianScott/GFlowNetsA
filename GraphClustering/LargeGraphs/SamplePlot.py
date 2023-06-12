from GraphClustering.Core.Core import GraphNet, torch_posterior, Gibbs_sample_torch
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


def compareIRMSamples(tensors: list, nbins=100, names=None, filenameSave='', title=''):
    if names is None:
        names = [str(el) for el in range(1, 1 + len(tensors))]
    # tensorsFlat = torch.concat(tensors, dim=0)
    IRM_lists = []
    sort_idxs = []
    for name, state_tensor in zip(names, tensors):
        IRM_list = []
        for state in state_tensor:
            adj_mat, cluster_mat = net.get_matrices_from_state(state)
            cluster_list, _ = net.get_clustering_list(cluster_mat)
            IRM_value = int(torch_posterior(adj_mat, cluster_list - 1))
            IRM_list.append(IRM_value)
        sort_idxs.append(np.argsort(IRM_list))
        IRM_list = sorted(IRM_list)
        plt.hist(IRM_list, label=name, bins=nbins)
        IRM_lists.append(IRM_list)

    plt.ylabel('Count')
    plt.xlabel('Log IRM Values')
    plt.title(title)
    plt.legend()
    if filenameSave:
        plt.savefig(filenameSave)
    plt.show()

    return IRM_lists, sort_idxs


if __name__ == '__main__':
    run_Gibbs = False   # There is a saved run for 10_000 samples in this folder

    net = GraphNet(n_nodes=34)
    fname = 'Data/KarateResults_100_500_10000_o_Samples_400.pt'
    netSamples = net.load_samples(fname)

    # Load the adjacency matrix to create the state vector, used by the plotting function
    Adj_karate = torch.tensor(pd.read_csv("Adj_karate.csv", header=None, dtype=int).to_numpy())

    if run_Gibbs:
        gibbsSamples = Gibbs_sample_torch(torch.tensor(Adj_karate, dtype=torch.float32), T=len(netSamples) * 2, return_clustering_matrix=True)
    else:
        gibbsSamples = net.load_samples('GibbsSamples_10000.pt')
    gibbsSamplesPlotable = [torch.concat((Adj_karate.flatten(), gibbsSample.flatten())) for gibbsSample in gibbsSamples]
    torch.save(gibbsSamplesPlotable, 'GibbsSamples_10000.pt')

    def return_0():
        return 0

    cluster_lists = defaultdict(return_0)
    IRM_list = []
    for state in netSamples:
        adj_mat, cluster_mat = net.get_matrices_from_state(state)
        cluster_list, _ = net.get_clustering_list(cluster_mat)
        IRM_value = int(torch_posterior(adj_mat, cluster_list - 1))
        IRM_list.append(IRM_value)
        cluster_lists[tuple(list(cluster_list))] += 1

    plt.plot(IRM_list, label='IRM Values')
    plt.plot(list(cluster_lists.values()), label='Cluster Count')
    plt.legend()
    plt.show()


    I = compareIRMSamples([netSamples, gibbsSamplesPlotable],
                          names=['GFlowNet', 'Gibbs Sampler'],
                          title='Histogram of log IRM values for GFlowNet vs GibbsSampler\non Zachary Karate Club graph',
                          filenameSave='comparisonGraph10000Samples.png')
