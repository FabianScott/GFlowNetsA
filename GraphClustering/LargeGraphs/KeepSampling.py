from GraphClustering.Core.Core import GraphNet, torch_posterior
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

if __name__ == '__main__':
    Adj_karate = torch.tensor(pd.read_csv("Adj_karate.csv", header=None, dtype=int).to_numpy())

    net = GraphNet(n_nodes=34)
    net.load_forward(prefix='Data/Karate_100_500_10000_o_', postfix='400')
    net.sample_forward(Adj_karate, n_samples=250_000, saveFilename='Data/KarateResults_100000_Sample_o', timer=True)
