from GraphClustering.Core.Core import GraphNet, torch_posterior, Gibbs_sample_torch
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from scipy import stats
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "..", "Data"))


def compareIRMSamples(tensors: list, nbins=100, names=None, filenameSave='', title=''):
    """
    Plot a histogram of clusterings per IRM values.
    Can also save the plots.
    :param tensors:
    :param nbins:
    :param names:
    :param filenameSave:
    :param title:
    :return:
    """
    if names is None:
        names = [str(el) for el in range(1, 1 + len(tensors))]
    # tensorsFlat = torch.concat(tensors, dim=0)
    IRM_lists = []
    sort_idxs = []
    for name, tensors in zip(names, tensors):
        IRM_list = []
        for state in tqdm(tensors, desc=f'Calculating IRM for {name} States'):
            adj_mat, cluster_mat = net.get_matrices_from_state(state)
            cluster_list, _ = net.get_clustering_list(cluster_mat)
            if not sum(cluster_list == 0):
                cluster_list -= 1
            IRM_value = int(torch_posterior(adj_mat, cluster_list))
            IRM_list.append(IRM_value)
        IRM_list = np.array(IRM_list)
        sort_idxs.append(np.argsort(IRM_list))
        IRM_list = sorted(IRM_list)
        plt.hist(IRM_list, label=name, bins=nbins, alpha=0.6)
        IRM_lists.append(IRM_list)
        print(f'For {name} we find \tMean: {np.mean(IRM_list)}\tMode: {stats.mode(IRM_list)}\tMax: {np.max(IRM_list)}')
    plt.ylabel('Count')
    plt.xlabel('Log IRM Values')
    plt.title(title)
    plt.legend()
    if filenameSave:
        plt.savefig(filenameSave)
    plt.show()

    return IRM_lists, sort_idxs


def getTopClusterings(states, n=10, csvFilename=''):  # , plot=False, saveFilename='', title='', nbins=100)
    """
    Given an iterable of tensors that represent states,
    return the top 10 most common states and potentially
    save them as csv files
    :param states:
    :param n:
    :param csvFilename:
    :return:
    """
    def return_0():
        return 0

    # Use a defaultdict for ease
    cluster_lists = defaultdict(return_0)
    IRM_list = []
    for state in tqdm(states, desc='Counting most common states'):
        adj_mat, cluster_mat = net.get_matrices_from_state(state)
        cluster_list, _ = net.get_clustering_list(cluster_mat)
        if not sum(cluster_list == 0):
            cluster_list -= 1
        IRM_value = int(torch_posterior(adj_mat, cluster_list))
        IRM_list.append(IRM_value)
        cluster_lists[tuple(cluster_list.detach().numpy())] += 1

    # Put the dictionary values into lists so numpy can sort them
    clusters = []
    counts = []
    for cluster, count in cluster_lists.items():
        clusters.append(cluster)
        counts.append(count)
    sort_idx = np.argsort(counts)
    top_n = np.array(clusters)[sort_idx[-n:]]
    if csvFilename: pd.DataFrame(top_n).to_csv(csvFilename)
    # if plot:
    #     plt.hist(np.array(counts)[sort_idx], label='Cluster Count', bins=nbins)
    #     # plt.xlabel('Cluster order')
    #     plt.ylabel('Counts')
    #     plt.title(title)
    #     plt.legend()
    #     plt.show()
    #     if saveFilename:
    #         plt.savefig(saveFilename)

    return top_n


if __name__ == '__main__':
    run_Gibbs = False  # There is a saved run for 10_000 samples in this folder
    epochs = 0
    topClustersList = []
    for epochs in range(0, 100, 100):
        net = GraphNet(n_nodes=34)
        fname = f'Data/NewKarateResults_0_500_10000_o_Samples_{epochs}.pt'
        netSamples = net.load_samples(fname)

        # Load the adjacency matrix to create the state vector, used by the plotting function
        Adj_karate = torch.tensor(pd.read_csv("Adj_karate.csv", header=None, dtype=int).to_numpy())

        if run_Gibbs:
            gibbsSamples = Gibbs_sample_torch(torch.tensor(Adj_karate, dtype=torch.float32), T=len(netSamples) * 2,
                                              return_clustering_matrix=True)
            gibbsSamplesPlotable = [torch.concat((Adj_karate.flatten(), gibbsSample.flatten())) for gibbsSample in gibbsSamples]
        else:
            gibbsSamplesPlotable = net.load_samples('Data/GibbsSamples_10000.pt')

        torch.save(gibbsSamplesPlotable, 'Data/GibbsSamples_10000.pt')
        I = compareIRMSamples([netSamples, gibbsSamplesPlotable],
                              names=['GFlowNet', 'Gibbs Sampler'],
                              title=f'Histogram of log IRM values for GFlowNet vs GibbsSampler\non Zachary Karate Club graph after {epochs} epochs',
                              )#filenameSave=f'Plots/comparisonGraph10000Samples_{epochs}.png')

        topClustersNet = getTopClusterings(netSamples, n=5, csvFilename=f'Data/TopClusteringsGFlowNet_{epochs}.csv')
        topClustersGibbs = getTopClusterings(gibbsSamplesPlotable, n=5, csvFilename=f'Data/TopClusteringsGibbs_{epochs}.csv')
        topClustersList.append([topClustersNet, topClustersGibbs])
