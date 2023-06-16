from GraphClustering import GraphNet, torch_posterior, Gibbs_sample_torch
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


def return_0():
    return 0


def returnSet():
    return set()


def compareIRMSamples(tensors: list, nbins=100, names=None, filenameSave='', title='', n=5, topNFilename='', alpha=.6, runGibbs=False):
    """
    Plot a histogram of clusterings per IRM values.
    Can also save the plots.
    :param tensors:
    :param nbins:
    :param names:
    :param filenameSave:
    :param title:
    :param topNFilename: Will be postfixed using the name provided in names
    :return:
    """
    if names is None:
        names = [str(el) for el in range(1, 1 + len(tensors))]
    # tensorsFlat = torch.concat(tensors, dim=0)
    IRM_lists = []

    for name, tensors in zip(names, tensors):
        cluster_lists = defaultdict(return_0)

        if name == 'Gibbs Sampler' and not runGibbs:
            try:
                IRM_list = pd.read_csv('Data/IRMGibbs.csv', index_col=0).values.T[0]
                uniqueCount = pd.read_csv('Data/uniqueCountGibbs.csv', index_col=0).values.T[0]
                uniqueIRM = pd.read_csv('Data/uniqueIRMGibbs.csv', index_col=0).values.T[0]
                plt.hist(IRM_list, label=name, bins=nbins, alpha=0.6)
                plt.plot(uniqueIRM, uniqueCount, label=f'Unique Clusterings for {name}', alpha=alpha)

                continue
            except FileNotFoundError:
                pass
        IRM_list = []
        IRM_dict = defaultdict(returnSet)
        for state in tqdm(tensors, desc=f'Calculating IRM for {name} States'):
            adj_mat, cluster_mat = net.get_matrices_from_state(state)
            cluster_list, _ = net.get_clustering_list(cluster_mat)
            if not sum(cluster_list == 0):
                cluster_list -= 1
            IRMValue = int(torch_posterior(adj_mat, cluster_list))
            tempTuple = tuple(cluster_list.detach().numpy())
            IRM_list.append(IRMValue)
            cluster_lists[tempTuple] += 1
            IRM_dict[IRMValue].add(tempTuple)

        uniqueCount, uniqueIRM = [], []
        for IRMValue, clusterSet in IRM_dict.items():
            uniqueCount.append(len(clusterSet))
            uniqueIRM.append(IRMValue)
        uniqueSortIdx = np.argsort(uniqueIRM)
        uniqueCount, uniqueIRM = np.array(uniqueCount)[uniqueSortIdx], np.array(uniqueIRM)[uniqueSortIdx]

        IRM_list = np.array(IRM_list)
        if name == 'Gibbs Sampler':
            pd.DataFrame(uniqueCount).to_csv('Data/uniqueCountGibbs.csv')
            pd.DataFrame(uniqueIRM).to_csv('Data/uniqueIRMGibbs.csv')
            pd.DataFrame(IRM_list).to_csv('Data/IRMGibbs.csv')

        plt.hist(IRM_list, label=name, bins=nbins, alpha=alpha)
        plt.plot(uniqueIRM, uniqueCount, label=f'Unique Clusterings for {name}', alpha=alpha)
        clusters = [cluster for cluster in cluster_lists.keys()]
        counts = [count for count in cluster_lists.values()]
        sort_idx = np.argsort(counts)

        top_n = np.array(clusters)[sort_idx[-n:]]
        if topNFilename: pd.DataFrame(top_n).to_csv(topNFilename + name + '.csv')
        print(f'\n{name}\tMean: {np.mean(IRM_list)}\tMode: {stats.mode(IRM_list)}\tMax: {np.max(IRM_list)}')

    plt.ylabel('Count')
    plt.xlabel('Log IRM Values')
    plt.title(title)
    plt.legend()
    if filenameSave:
        plt.savefig(filenameSave)
    plt.show()

    return IRM_lists


if __name__ == '__main__':
    run_Gibbs = False  # There is a saved run for 10_000 samples in this folder
    prefixString = 'New'
    epochs = 0
    for epochs in range(0, 100, 100):
        net = GraphNet(n_nodes=34)
        fname = f'Data/{prefixString}KarateResults_0_500_10000_o_Samples_{epochs}.pt'
        netSamples = net.load_samples(fname)

        # Load the adjacency matrix to create the state vector, used by the plotting function
        Adj_karate = torch.tensor(pd.read_csv("Data/Adj_karate.csv", header=None, dtype=int).to_numpy())

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
                              filenameSave=f'Plots/{prefixString}ComparisonGraph10000Samples_{epochs}.png',
                              topNFilename=f'Data/{prefixString}TopClusterings_{epochs}_',
                              n=5,
                              runGibbs=False)
