from GraphClustering import GraphNetNodeOrder, torch_posterior
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

if __name__ == '__main__':
    n_samples = 1000
    Adj_karate = torch.tensor(pd.read_csv("Data/Adj_karate.csv", header=None, dtype=int).to_numpy())

    net = GraphNetNodeOrder(n_nodes=34)
    net.load_forward(prefix='Data/Karate_100_500_10000_o_', postfix='400')
    net.sample_forward(Adj_karate, nSamples=n_samples, saveFilename='Data/KarateResults_100000_Sample_o', timer=True)
