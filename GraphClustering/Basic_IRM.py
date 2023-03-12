import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from irmlearn import IRM


# Takes 4 parameters, l for nr. of clusters,
# k for nodes in clsuters, p and q for connective
# probabilities inside and outside groups.
def ClusterGraph(l, k, p, q):
    n = l * k
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            prob = np.random.rand(2)

            if i // k == j // k and prob[0] < p:
                adjacency[i, j] = 1

            elif prob[1] < q:
                adjacency[i, j] = 1

    return adjacency


def block_diag(A, B):
    c1 = np.zeros((A.shape[0], B.shape[1]))

    c2 = np.zeros((B.shape[0], A.shape[1]))

    return np.block([[A, c1],
                     [c2, B]])


l = 25
k = 6

A_adj = ClusterGraph(l, k, 0.9, 0.01)

idxs = np.random.permutation(np.arange(l * k))
A_random = A_adj[idxs][:, idxs] # It makes sense to permute both axes in the same way.
# Otherwise, you change the edges and their directionality. 

# A = nx.from_numpy_array(A_random)
# nx.draw(A, node_size=30)

plt.figure()
plt.imshow(A_adj, cmap='gray')

plt.figure()
plt.imshow(A_random, cmap='gray')

alpha = 1.5
a = 0.1
b = 0.1
max_iter = 20

model = IRM(alpha, a, b, max_iter, verbose=True, use_best_iter=True, random_state = 42) 

model.fit(A_random)

model._calc_posterior(A_random) # We are down the rabit hole of private methods here. 
print(model._logv_cur)

plt.plot(model.history_)




