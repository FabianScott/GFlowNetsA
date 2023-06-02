import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

G = nx.karate_club_graph()
Adj_karate = nx.adjacency_matrix(G).todense()
Adj_karate = Adj_karate>0

