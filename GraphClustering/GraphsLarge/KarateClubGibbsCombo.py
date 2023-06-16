try:
    from Core import GraphNet, GraphNetNodeOrder, torch_posterior, check_gpu, Gibbs_sample_torch, GibbsSampleStates
except ModuleNotFoundError:
    from GraphClustering.Core.Core import GraphNet, GraphNetNodeOrder, torch_posterior, check_gpu, Gibbs_sample_torch, GibbsSampleStates
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
    n_samples = 10000   # Must be greater than 1
    epoch_interval = 500
    min_epochs = 0
    max_epochs = 1000
    node_order = True
    GibbsStart = False
    folder_and_forward_slash = 'Data/GibbsHalf'
    prefix = 'WomboCombo'

    # Filepaths created:
    node_order_string = '_o' if node_order else ''
    filepathSamples = f'Data/{prefix}Karate{min_epochs}_{max_epochs}_{n_samples}{node_order_string}_Samples_'
    filepathWeights = f'Weights/{prefix}Karate{min_epochs}_{max_epochs}_{n_samples}{node_order_string}'

    # Load graph and network:
    Adj_karate = torch.tensor(pd.read_csv("Data/Adj_karate.csv", header=None, dtype=int).to_numpy())
    n = Adj_karate.shape[0]
    net = GraphNetNodeOrder(n_nodes=n, n_layers=5, n_hidden=64) if node_order else GraphNet(
        Adj_karate.shape[0])
    net.save(prefix=filepathWeights, postfix=str(0))

    # Initial sample:
    X1 = GibbsSampleStates(Adj_karate, n_samples=n_samples, N=n) if GibbsStart \
        else net.sample_forward(Adj_karate, n_samples=n_samples, timer=True)
    torch.save(X1, filepathSamples + f'{0}.pt')

    for i in range(1, ((max_epochs - min_epochs) // epoch_interval) + 1):
        net.train(X1, epochs=epoch_interval)  # Train an extra epoch interval
        # Take a sample from the GFlowNet part of the previous samples:
        z = X1[n_samples//2:][torch.randint(n_samples//2, (1,))][0][net.n_nodes**2:].reshape((net.n_nodes, net.n_nodes))
        z, _ = net.get_clustering_list(z)
        z = z.reshape((-1, 1))
        # Sample again:
        gibbsSamples = GibbsSampleStates(Adj_karate, n_samples=n_samples // 2, N=net.n_nodes, z=z)
        gflowSamples = net.sample_forward(Adj_karate,
                                          n_samples=n_samples // 2,
                                          timer=True,
                                          saveFilename=filepathSamples + f'{i * epoch_interval}')
        X1 = torch.concat((gibbsSamples, gflowSamples), dim=0)
        net.save(prefix=filepathWeights, postfix=str(epoch_interval * i))
