import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import os

# Karate club graph
G = nx.karate_club_graph()
Adj_karate = nx.adjacency_matrix(G).todense()
Adj_karate = Adj_karate>0

# C. elegans graph
Adj_celegans = pd.read_csv("../celegans277/celegans277matrix.csv").to_numpy()
