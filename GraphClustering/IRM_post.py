import torch
# from torch.distributions.beta import Beta
# from torch.distributions.gamma import Gamma
import numpy as np
from scipy.special import betaln, gammaln


def p_x_giv_z(A, C, a=1, b=1, log=True):
    """Calculate P(X|z): the probability of the graph given a particular clustering structure.
    # This is calculated by integrating out all the internal cluster connection parameters.

    Parameters
    ----------
    A : Adjacency matrix (2D ndarray)
    C : clustering index array (1D ndarray) (n long with the cluster c of each node ordered by the Adjacency matrix at each index)
    a and b: float
        Parameters for the beta distribution prior for the cluster connectivities. 
        a = b = 1 yields a uniform distribution.
    log : Bool
        Whether or not to return log of the probability

    Return
    ----------
    Probability of data given clustering: float
    """
    np.einsum("ii->i", A)[...] = 0  # This function assumes that nodes aren't connected to themselves. This should be irrelevant for the clustering.
    # Product over all pairs of components.
    values, nk = np.unique(C, return_counts=True)
    # I just need to create m_kl and m_bar_kl matrices. Then I can vectorize the whole thing

    # create node-cluster adjacency matrix
    n_C = np.identity(C.max() + 1, int)[C]
    m_kl = n_C.T @ A @ n_C

    np.einsum("ii->i", m_kl)[
        ...] //= 2  # np.diag(m_kl)[...] //= 2 , but this allows for an in-place update.
    m_bar_kl = np.outer(nk, nk) - np.diag(nk * (nk + 1) / 2) - m_kl  # The diagonal simplifies to the sum up to nk-1.

    # Alternative to the matrix multiplication. This is a little faster.
    # Sort A according to the clustering.
    # boundaries = np.diff(C, prepend=-1).nonzero()[0] # tells me at what index each cluster begins. 
    # out =  np.add.reduceat(np.add.reduceat(A,boundaries,1),boundaries,0) # Basically just the matrix-algebra above.

    logP_x_giv_z = np.sum(betaln(m_kl + a, m_bar_kl + b) - betaln(a, b))

    return logP_x_giv_z if log else np.exp(logP_x_giv_z)


def p_z(A, C, alpha=1, log=True):
    """Probability of clustering.

    Parameters
    ----------
    A : Adjacency matrix (2D ndarray)
    C : clustering index array (ndarray)
    alpha : float
        Concentration of clusters.
    log : Bool
        Whether or not to return log of the probability

    Return
    ----------
    probability of cluster: float
    """

    # Alpha is the concentration parameter. In theory, this could be different for the different clusters.
    # A constant concentration corrosponds to the chinese restaurant process. 
    K = np.amax(C)
    N = len(A)
    values, nk = np.unique(C,
                           return_counts=True)  # nk is an array of counts, so the number of elements in each cluster.
    K_bar = len(values) - K  # number of empty clusters.

    log_labellings = gammaln(K + 1) - gammaln(K - K_bar + 1)
    A = alpha * K

    # nk (array of number of nodes in each cluster)
    log_p_z = log_labellings * (gammaln(A) - gammaln(A + N)) * np.sum(gammaln(alpha + nk) - gammaln(alpha))

    return log_p_z if log else np.exp(log_p_z)



def torch_posterior(A_in, C_in, a=torch.ones(1), b=torch.ones(1), alpha = 1, log=True):
    # Likelyhood part
    A = torch.t_copy(A_in)
    C = torch.t_copy(torch.tensor(C_in, dtype=torch.int32))
    torch.einsum("ii->i", A)[...] = 0   # Fills the diagonal with zeros.
    values, nk = torch.unique(C, return_counts=True)
    n_C = torch.eye(int(C.max()) + 1)[C]

    m_kl = n_C.T @ A @ n_C
    torch.einsum("ii->i", m_kl)[...] //= 2  # m_kl[np.diag_indices_form(m_kl)] //= 2 should do the same thing.

    m_bar_kl = torch.outer(nk, nk) - torch.diag(nk * (nk + 1) / 2) - m_kl

    logP_x_giv_z = torch.sum(betaln(m_kl + a, m_bar_kl + b) - betaln(a, b))

    # Prior part
    K = torch.amax(C)
    N = torch.len(A)
    values, nk = torch.unique(C, return_counts=True)
    K_bar = len(values) - K  # number of empty clusters.

    log_labellings = gammaln(K + 1) - gammaln(K - K_bar + 1)
    A = alpha * K

    # nk (array of number of nodes in each cluster)
    log_p_z = log_labellings * (gammaln(A) - gammaln(A + N)) * torch.sum(gammaln(alpha + nk) - gammaln(alpha))

    # Return posterior
    return logP_x_giv_z + log_p_z if log else torch.exp(logP_x_giv_z + log_p_z)


def crp(n, alpha): # Not complete; just taken from the IRM package for inspiration.
    """Chinese restaurant process.

    Parameters
    ----------
    n : int
        num of people to seat.
    alpha : float
        Concentration.

    Return
    ----------
    assignments : list
        table_id for each people.
    n_assignments : list
        partition.
    """
    n = int(n)
    alpha = float(alpha)
    assert n >= 1
    assert alpha > 0

    assignments = [0]  # First person sits at table 0.
    n_assignments = [1]  # One person at table 0.
    for _ in range(2, n + 1):
        table_id = pick_discrete(n_assignments) - 1
        if table_id == -1:
            n_assignments.append(1)
            assignments.append(len(n_assignments) - 1)
        else:
            n_assignments[table_id] = n_assignments[table_id] + 1
            assignments.append(table_id)

    return assignments, n_assignments

def Cmatrix_to_array(Cmat):
    C = np.zeros(len(Cmat))
    cluster = 0
    for i, row in enumerate(Cmat):  # row = Cmat[i]
        if np.any(Cmat[i]):
            C[Cmat[i].astype(bool)] = cluster 
            Cmat[Cmat[i].astype(bool)] = 0 # Remove these clusters
            cluster += 1
    return C





def ClusterGraph(l, k, p, q):
    n = l * k
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(n):

            prob = np.random.rand(2)

            if i // k == j // k and prob[0] < p:
                adjacency[(i, j), (j, i)] = 1

            elif prob[1] < q:
                adjacency[(i, j), (j, i)] = 1

    return adjacency


if __name__ == "__main__":
    # from Basic_IRM import ClusterGraph
    # A = np.array([[1,0,1,0,0], [0,1,1,1,1], [1,1,1,1,1], [0,1,1,1,1], [0,1,1,1,1]])
    # C = np.array([1,1,2,3,0])
    # p_x_giv_z(A, C, a=1, b=1)

    import matplotlib.pyplot as plt
    l = 5 # nr. of clusters,
    k = 10 # k for nodes in clsuters l*k becomes nodes in total

    A_adj = ClusterGraph(l, k, 0.9, 0.01)

    idxs = np.random.permutation(np.arange(l * k))
    A_random = A_adj[idxs][:, idxs]  # It makes sense to permute both axes in the same way.
    # Otherwise, you change the edges and their directionality. 

    K = 10
    # Create random clusterings
    iterations = 100
    probs_C_log = np.zeros(iterations)
    random_state = 42
    for i in range(iterations):
        np.random.seed(random_state)
        C = np.zeros(l * k).astype(int)
        for n in range(l * k):
            c = np.random.randint(0, high=K)
            C[n] = c  # Assign clusters at random.
        probs_C_log[i] = p_x_giv_z(A_random, C, a=1 / 2, b=1 / 2) + p_z(A_random, C, alpha=1)
        if i == 0: best_clustering, best_P = C, probs_C_log[i]
        if i > 0 and probs_C_log[i] > best_P: best_clustering, best_P = C, probs_C_log[i]

    c_idxs = np.argsort(best_clustering)
    print(best_clustering)
    A_C = A_adj[c_idxs][:, c_idxs]
    plt.figure()
    plt.imshow(A_adj, cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(A_C, cmap='gray')
    plt.show()

    a = np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])
    C1 = Cmatrix_to_array(a)
    print(C1)
    # print(A_C)
