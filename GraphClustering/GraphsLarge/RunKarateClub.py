try:
    from Core import GraphNet, GraphNetNodeOrder, torch_posterior, check_gpu, Gibbs_sample_torch
except ModuleNotFoundError:
    from GraphClustering.Core.Core import GraphNet, GraphNetNodeOrder, torch_posterior, check_gpu, Gibbs_sample_torch
from copy import deepcopy
import networkx as nx
import pandas as pd
import numpy as np
import torch


def GibbsSampleStates(adjacency_matrix, n_samples, N):
    tempSamples = Gibbs_sample_torch(torch.tensor(adjacency_matrix, dtype=torch.float32), T=n_samples * 2)
    X1 = torch.zeros((n_samples, N ** 2 * 2))

    for i, sample in enumerate(tempSamples):
        X1[i] = torch.concat((adjacency_matrix.flatten(), torch.tensor(sample.flatten())))
    return X1


if __name__ == '__main__':
    """
    Define the interval and step for epochs you want to test
    on the karate club graph. The script saves two different
    values, firstly, the difference in the network's prediction 
    of each sampled state and the IRM value, secondly the sum
    of the IRM values for the sampled states.
    """
    check_gpu()
    # Parameters to set:
    n_samples = 1
    epoch_interval = 1
    min_epochs = 0
    max_epochs = 1
    node_order = True
    GibbsStart = True
    prefix = ''

    # Filepaths created:
    node_order_string = '_o' if node_order else ''
    filepathSamples = f'Data/{prefix}Karate{min_epochs}_{max_epochs}_{n_samples}{node_order_string}'
    filepathWeights = f'Weights/{prefix}Karate{min_epochs}_{max_epochs}_{n_samples}{node_order_string}'

    # Load graph and network:
    Adj_karate = torch.tensor(pd.read_csv("Data/Adj_karate.csv", header=None, dtype=int).to_numpy())
    net = GraphNetNodeOrder(Adj_karate.shape[0], n_layers=5, n_hidden=64) if node_order else GraphNet(
        Adj_karate.shape[0])
    net.save(prefix=filepathWeights, postfix=str(0))

    # Initial sample:
    X1 = GibbsSampleStates(Adj_karate, n_samples=n_samples, N=net.n_nodes) if GibbsStart \
        else net.sample_forward(Adj_karate, n_samples=n_samples, timer=True)
    torch.save(X1, filepathSamples + f'_Samples_{0}' + '.pt')

    # Sampling loop:
    for i in range(1, ((max_epochs - min_epochs) // epoch_interval) + 1):
        net.train(X1, epochs=epoch_interval // 2)  # Train an extra epoch interval

        X1 = net.sample_forward(Adj_karate,
                                n_samples=n_samples,
                                timer=True,
                                saveFilename=filepathSamples + f'_Samples_{i * epoch_interval}')
        net.save(prefix=filepathWeights, postfix=str(epoch_interval * i))
