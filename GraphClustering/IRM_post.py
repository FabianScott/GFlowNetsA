import numpy as np
from scipy.special import betaln, gammaln

A = np.zeros((8,8))

A[0:4,0:4] = np.ones((4,4))
A[4:9,4:9] = np.ones((4,4))

K = 3
N = 10
test_P = np.zeros((K,N))
random_state = 42
np.random.seed(random_state)
for n in range(N):
    k = np.random.randint(0, high=K)
    test_P[k,n] = 1

# This is just an idea.
# I expect to recieve clusterings as an adjacency matrix
# and an index key with the entries as the cluster index.


def p_x_giv_z(K):
    """Calculate P(X|z): the probability of the graph given a particular clustering structure.
    # This is calculated by integrating out all the internal cluster connection parameters.
    # This is non parametric in terms of clusters.

    # Alphak is the concentration parameter of cluster (component) k. 
    # nk are the number of nodes in cluster k. 

    # z is a parameter that stores the partitioning of the network.
    """
    # 

    pass

def p_z(A, C, alpha = 1):

    """Probability of clustering.

    Parameters
    ----------
    A : Adjacency list
    C : clustering index list
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
    

    # P is a partitioning of the graph as a boolean matrix of dimension (clusters, nodes). Short and fat. 
    # Alpha is the concentration parameter. In theory, this could be different for the different clusters.
    # A constant concentration corrosponds to the chinese restaurant process. 
    
    log_labellings = gammaln(K+1) - gammaln(K-K_bar+1)
    A = alpha*K

    # nk (array of number of nodes in each cluster)
    log_p_z = log_labellings* (gammaln(A)-gammaln(A+N)) * np.sum(gammaln(alpha+nk)-gammaln(alpha))

    return np.exp(log_p_z)

    """Probability of clustering.

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

def crp(n, alpha):
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

    assignments = [0] # First person sits at table 0.
    n_assignments = [1] # One person at table 0. 
    for _ in range(2, n + 1):
        table_id = pick_discrete(n_assignments) - 1
        if table_id == -1:
            n_assignments.append(1)
            assignments.append(len(n_assignments) - 1)
        else:
            n_assignments[table_id] = n_assignments[table_id] + 1
            assignments.append(table_id)

    return assignments, n_assignments