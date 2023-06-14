try:
    from Core import GraphNet, GraphNetNodeOrder, torch_posterior, check_gpu, Gibbs_sample_torch
except ModuleNotFoundError:
    from GraphClustering.Core.Core import GraphNet, GraphNetNodeOrder, torch_posterior, check_gpu, Gibbs_sample_torch
from copy import deepcopy
import networkx as nx
import pandas as pd
import numpy as np
import torch

if __name__ == '__main__':
    """
    Define the interval and step for epochs you want to test
    on the karate club graph. The script saves two different
    values, firstly, the difference in the network's prediction 
    of each sampled state and the IRM value, secondly the sum
    of the IRM values for the sampled states.
    """
    check_gpu()
    # 9000 for the n_layers=5, n_hidden=64
    n_samples = 10000
    epoch_interval = 100
    min_epochs = 0
    max_epochs = 500
    node_order = True
    GibbsStart = True
    folder_and_forward_slash = 'Data/Gibbs'
    filename1 = f'{folder_and_forward_slash}KarateResults_{min_epochs}_{max_epochs}_{n_samples}_o.csv' if node_order else f'{folder_and_forward_slash}KarateResults_{min_epochs}_{max_epochs}_{n_samples}.csv'

    Adj_karate = torch.tensor(pd.read_csv("Adj_karate.csv", header=None, dtype=int).to_numpy())
    net = GraphNetNodeOrder(Adj_karate.shape[0], n_layers=5, n_hidden=64) if node_order else GraphNet(Adj_karate.shape[0])
    net.save(prefix=f'{folder_and_forward_slash}Karate_{min_epochs}_{max_epochs}_{n_samples}_o_' if node_order else f'{folder_and_forward_slash}Karate_{min_epochs}_{max_epochs}_{n_samples}_', postfix=str(0))

    if GibbsStart:
        tempSamples = Gibbs_sample_torch(torch.tensor(Adj_karate, dtype=torch.float32), T=n_samples * 2)
        X1 = torch.zeros((n_samples, net.n_nodes ** 2 * 2))
        for i, sample in enumerate(tempSamples):
            X1[i] = torch.concat((Adj_karate.flatten(), torch.tensor(sample.flatten())))
        torch.save(X1, filename1[:-4] + f'_Samples_{0}' + '.pt')
    else:
        X1 = net.sample_forward(Adj_karate, n_samples=n_samples, timer=True)

    for i in range(1, ((max_epochs - min_epochs) // epoch_interval) + 1):
        net.train(X1, epochs=epoch_interval)  # Train an extra epoch interval
        X1 = net.sample_forward(Adj_karate,
                                n_samples=n_samples,
                                timer=True,
                                saveFilename=filename1[:-4] + f'_Samples_{i * epoch_interval}')
        net.save(prefix=f'{folder_and_forward_slash}Karate_{min_epochs}_{max_epochs}_{n_samples}_o_' if node_order else f'{folder_and_forward_slash}Karate_{min_epochs}_{max_epochs}_{n_samples}_', postfix=str(epoch_interval * i))
