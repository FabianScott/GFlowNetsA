try:
    from Core import GraphNet, GraphNetNodeOrder, torch_posterior, check_gpu, Gibbs_sample_torch, GibbsSampleStates
except ModuleNotFoundError:
    from GraphClustering import GraphNet, GraphNetNodeOrder, torch_posterior, check_gpu, Gibbs_sample_torch, GibbsSampleStates
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
    nSamples = 10   # Must be greater than 1
    epochInterval = 1
    minEpochs = 0
    maxEpochs = 2
    nodeOrder = True
    GibbsStart = False
    GibbsProportion = .6
    # folder_and_forward_slash = 'Data/GibbsHalf'
    prefix = 'WomboCombo'

    # Filepaths created:
    nodeOrderString = '_o' if nodeOrder else ''
    filepathSamples = f'Data/{prefix}Karate{minEpochs}_{maxEpochs}_{nSamples}{nodeOrderString}_Samples_'
    filepathWeights = f'Weights/{prefix}Karate{minEpochs}_{maxEpochs}_{nSamples}{nodeOrderString}'

    # Load graph and network:
    Adj_karate = torch.tensor(pd.read_csv("Data/Adj_karate.csv", header=None, dtype=int).to_numpy())
    n = Adj_karate.shape[0]
    net = GraphNetNodeOrder(n_nodes=n, nLayers=5, nHidden=64) if nodeOrder else GraphNet(
        Adj_karate.shape[0])
    net.save(prefix=filepathWeights, postfix=str(0))

    # Initial sample:
    X1 = GibbsSampleStates(Adj_karate, nSamples=nSamples, N=n) if GibbsStart \
        else net.sample_forward(Adj_karate, nSamples=nSamples, timer=True)
    torch.save(X1, filepathSamples + f'{0}.pt')
    nGibbs = int(nSamples * GibbsProportion)

    for i in range(1, ((maxEpochs - minEpochs) // epochInterval) + 1):
        net.train(X1, epochs=epochInterval)  # Train an extra epoch interval
        # Take a sample from the GFlowNet part of the previous samples:
        z = X1[nGibbs:][torch.randint(nSamples - nGibbs, (1,))][0][net.n_nodes ** 2:].reshape((net.n_nodes, net.n_nodes))
        z = net.get_clustering_list(z)[0].reshape((-1, 1))
        # Sample again:
        gibbsSamples = GibbsSampleStates(Adj_karate, nSamples=nGibbs, N=net.n_nodes, z=z)
        gflowSamples = net.sample_forward(Adj_karate,
                                          nSamples=nSamples - nGibbs,
                                          timer=True,
                                          saveFilename=filepathSamples + f'{i * epochInterval}')
        X1 = torch.concat((gibbsSamples, gflowSamples), dim=0)
        net.save(prefix=filepathWeights, postfix=str(epochInterval * i))
