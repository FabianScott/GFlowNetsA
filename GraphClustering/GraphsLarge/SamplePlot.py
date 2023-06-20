from GraphClustering import GraphNet, torch_posterior, Gibbs_sample_torch, compareIRMSamples
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


if __name__ == '__main__':
    run_Gibbs = False  # There is a saved run for 10_000 samples in this folder
    prefixString = 'Gibbs'
    epochs = 0
    n_samples = 10000

    for epochs in range(100, 500, 100):
        net = GraphNet(n_nodes=34)
        fname = f'Data/{prefixString}KarateResults_0_500_{n_samples}_o_Samples_{epochs}.pt'
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
        I = compareIRMSamples([netSamples, gibbsSamplesPlotable], net=net,
                              names=['GFlowNet', 'Gibbs Sampler'],
                              title=f'Histogram of log IRM values for GFlowNet vs GibbsSampler\non Zachary Karate Club graph after {epochs} epochs',
                              filenameSave=f'Plots/{prefixString}ComparisonGraph10000Samples_{epochs}.png',
                              topNFilename=f'Data/{prefixString}TopClusterings_{epochs}_',
                              n=5,
                              runGibbs=False)
