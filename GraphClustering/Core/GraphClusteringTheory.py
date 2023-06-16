import pandas as pd
import torch
import itertools
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from scipy import stats
from scipy.special import betaln, gammaln, logsumexp
from torch.special import gammaln as torch_gammaln
from torch.distributions import Beta
import matplotlib.pyplot as plt
from collections import defaultdict


# %% Graph Theory functions
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
    np.einsum("ii->i", A)[
        ...] = 0  # This function assumes that nodes aren't connected to themselves. This should be irrelevant for the clustering.
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

    logP_x_giv_z = np.sum(np.triu(betaln(m_kl + a, m_bar_kl + b) - betaln(a, b)))

    return logP_x_giv_z if log else np.exp(logP_x_giv_z)


def p_z(A, C, A_alpha=1, log=True):
    """Probability of clustering.

    Parameters
    ----------
    A : Adjacency matrix (2D ndarray)
    C : clustering index array (ndarray)
    A_alpha : float
        Total concentration of clusters.
    log : Bool
        Whether or not to return log of the probability

    Return
    ----------
    probability of cluster: float
    """

    # A_alpha is the total concentration parameter.
    # A constant concentration corrosponds to the chinese restaurant process.
    N = len(A)
    values, nk = np.unique(C,
                           return_counts=True)  # nk is an array of counts, so the number of elements in each cluster.
    K_bar = len(values)  # number of non empty clusters.

    # nk (array of number of nodes in each cluster)
    log_p_z = (gammaln(A_alpha) + K_bar * (np.log(A_alpha)) - gammaln(A_alpha + N)) + np.sum(gammaln(nk))

    return log_p_z if log else np.exp(log_p_z)


def torch_posterior(A_in, C_in, a=None, b=None, A_alpha=None, log=True, verbose=False):
    """Calculate P(X,z): the joint probability of the graph and a particular clustering structure. This is proportional to the posterior.
    # This is calculated by integrating out all the internal cluster connection parameters.

    Parameters
    ----------
    A : Adjacency matrix (2D ndarray)
    C : clustering index array (1D ndarray) (n long with the cluster c of each node ordered by the Adjacency matrix at each index)
        Importantly it is 0 indexed! (Write to me if I should change this. We should be consistent on this, and it is probably best to keep 0 as unclustered)
    a and b: float
        Parameters for the beta distribution prior for the cluster connectivities.
        a = b = 1 yields a uniform distribution.
    A_alpha : float
        Total concentration of clusters.
    log : Bool
        Whether or not to return log of the probability
    verbose: Bool
        Whether or not to return the part of the computer this computation is computed on.

    Return
    ----------
    Probability of data and clustering: float
    """
    assert 0 in C_in  # All nodes should be clustered and the clusters should be 0-indexed. 0 must be in C_in. # We really should decide on one standard here.

    # Likelihood part
    if a is None:
        a = torch.ones(1)
    if b is None:
        b = torch.ones(1)
    if A_alpha is None:
        A_alpha = torch.ones(1)

    A = torch.t_copy(A_in)
    C = torch.t_copy(torch.tensor(C_in, dtype=torch.int64))
    torch.einsum("ii->i", A)[...] = 0  # Fills the diagonal with zeros.
    values, nk = torch.unique(C, return_counts=True)
    n_C = torch.eye(int(C.max()) + 1)[C]

    m_kl = n_C.T @ A @ n_C
    torch.einsum("ii->i", m_kl)[
        ...] /= 2  # m_kl[np.diag_indices_form(m_kl)] //= 2 should do the same thing. Will always be an integer.

    m_bar_kl = torch.outer(nk, nk) - torch.diag(nk * (nk + 1) / 2) - m_kl

    if verbose: print(m_bar_kl.device, m_kl.device)

    if str(a.device)[:4] == 'cuda':
        a_ = a.detach().cpu().numpy()
        b_ = b.detach().cpu().numpy()
        m_kl_ = m_kl.detach().cpu().numpy()
        m_bar_kl_ = m_bar_kl.detach().cpu().numpy()
        logP_x_giv_z = torch.tensor(np.sum(torch.triu(betaln(m_kl_ + a_, m_bar_kl_ + b_) - betaln(a_, b_))))
    else:
        logP_x_giv_z = torch.sum(torch.triu(betaln(m_kl + a, m_bar_kl + b) - betaln(a, b)))

    # Prior part. P(z|K), så given K possible labellings.
    N = len(A)
    # values, nk = torch.unique(C, return_counts=True)
    K_bar = len(values)  # number of non empty clusters.

    log_p_z = (torch_gammaln(A_alpha) + K_bar * (torch.log(A_alpha)) - torch_gammaln(A_alpha + N)) + torch.sum(
        torch_gammaln(nk))
    # Return joint probability, which is proportional to the posterior
    return logP_x_giv_z + log_p_z if log else torch.exp(logP_x_giv_z + log_p_z)


def Cmatrix_to_array(Cmat):
    C = np.zeros(len(Cmat))
    cluster = 0
    for i, row in enumerate(Cmat):  # row = Cmat[i]
        if np.any(Cmat[i]):
            C[Cmat[i].astype(bool)] = cluster
            Cmat[Cmat[i].astype(bool)] = 0  # Remove these clusters
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


# Collected function
def IRM_graph(A_alpha, a, b, N):
    clusters = CRP(A_alpha, N)
    phis = Phi(clusters, a, b)
    Adj = Adj_matrix(phis, clusters)
    return Adj, clusters


# Perform Chinese Restaurant Process
def CRP(A_alpha, N):
    # First seating
    clusters = [[1]]
    for i in range(2, N + 1):
        # Calculate cluster assignment as index to the list clusters.
        p = torch.rand(1)
        probs = torch.tensor([len(cluster) / (i - 1 + A_alpha) for cluster in clusters])
        cluster_assignment = sum(torch.cumsum(probs, dim=0) < p)

        # Make new table or assign to current
        if cluster_assignment == len(clusters):
            clusters.append([i])
        else:
            clusters[cluster_assignment].append(i)

    # Return the cluster sizes
    return torch.tensor([len(cluster) for cluster in clusters])


# Return a symmetric matrix of cluster probabilities,
# defined by a beta distribution.
def Phi(clusters, a, b):
    n = len(clusters)
    phis = Beta(a, b).rsample((n, n))
    # Symmetrize
    for i in range(n - 1, -1, -1):
        for j in range(n):
            phis[i, j] = phis[j, i]

    return phis


# Helper function to construct block matrix of cluster probabilities.
def make_block_phis(phis, clusters):
    for i, ii in enumerate(clusters):
        for j, jj in enumerate(clusters):
            if j == 0:
                A = torch.full((ii, jj), phis[i, j])
            else:
                A = torch.hstack((A, torch.full((ii, jj), phis[i, j])))

        if i == 0:
            block_phis = A
        else:
            block_phis = torch.vstack((block_phis, A))

    return block_phis


# Construct adjacency matrix.
def Adj_matrix(phis, clusters):
    n = sum(clusters)
    Adj_matrix = torch.zeros((n, n))

    block_phis = make_block_phis(phis, clusters)

    # Iterate over all nodes and cluster probabilities.
    for i in range(n):
        for j in range(n):
            p = torch.rand(1)
            if p < block_phis[i, j]:
                Adj_matrix[i, j] = 1
                Adj_matrix[j, i] = 1
            else:
                Adj_matrix[i, j] = 0
                Adj_matrix[j, i] = 0

    return Adj_matrix


def clusterIndex(clusters):
    idxs = torch.tensor([])
    for i, k in enumerate(clusters):
        idxs = torch.cat((idxs, torch.tensor([i] * k)))
    return idxs


def allPermutations(n):
    """
    Return a list of all possible permutations of clustering
    lists for a graph with n nodes
    :param n: int
    :return: numpy array
    """
    perm = [[[1]]]
    for i in range(n - 1):
        perm.append([])
        for partial in perm[i]:
            for j in range(1, max(partial) + 2):
                perm[i + 1].append(partial + [j])

    return np.array(perm[-1]) - 1


def allPosteriors(A_random, a, b, A_alpha, log, joint = False):
    # Computing posteriors for all clusters.
    N = len(A_random)
    clusters_all = allPermutations(N)
    Bell = len(clusters_all)
    clusters_all_post = np.zeros(Bell)
    for i, cluster in enumerate(clusters_all):
        posterior = torch_posterior(A_random, cluster, a=torch.tensor(a), b=torch.tensor(b), A_alpha = torch.tensor(A_alpha), log= True)
        clusters_all_post[i] = posterior
    if joint: return clusters_all_post # Return the joint probability instead of normalizing.
    cluster_post = clusters_all_post - logsumexp(clusters_all_post) # Normalize them into proper log probabilities
    if not log: cluster_post = np.exp(cluster_post)
    return cluster_post


