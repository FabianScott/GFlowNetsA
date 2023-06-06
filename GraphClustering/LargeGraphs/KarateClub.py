from GraphClustering.Core.Core import GraphNet
import networkx as nx
import torch

if __name__ == '__main__':
    G = nx.karate_club_graph()
    Adj_karate = nx.adjacency_matrix(G).todense()
    Adj_karate = torch.tensor(Adj_karate>0)

    net = GraphNet(len(G.nodes), n_hidden=64)
    X = net.sample_forward(Adj_karate, n_samples=1000, timer=True)
    net.train(X, epochs=100)
