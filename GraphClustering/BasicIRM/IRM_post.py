import torch
import numpy as np
from scipy.special import betaln, gammaln
from torch.special import gammaln as torch_gammaln


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

def m_kl(A, C, return_m_bar_kl =  True):
    np.einsum("ii->i", A)[...] = 0
    values, nk = np.unique(C, return_counts=True)

    n_C = np.identity(C.max() + 1, int)[C] # create node-cluster adjacency matrix
    m_kl = n_C.T @ A @ n_C

    np.einsum("ii->i", m_kl)[...] //= 2
    if not return_m_bar_kl: return m_kl
    else:
        m_bar_kl = np.outer(nk, nk) - np.diag(nk * (nk + 1) / 2) - m_kl
        return m_kl, m_bar_kl

def r_nl(A, C, n, l):
    np.einsum("ii->i", A)[...] = 0
    values, nk = np.unique(C, return_counts=True)

    n_C = np.identity(C.max() + 1, int)[C] # create node-cluster adjacency matrix
    r_nl = (A @ n_C)[n,l]

    return r_nl

def Gibbs_sample_np(A, T, burn_in_buffer = None, sample_interval = None, seed = 42, a = 1, b = 1, A_alpha = 1, return_clustering_matrix = False):
    """Perform one Gibbs sweep with a specified node order to yield one sampled clustering.

    Parameters
    ----------
    A : (2D ndarray) Adjacency matrix.
    T : (int) Number of Gibbs Sweeps.
    burn_in_buffer : (int) Number of initial samples to discard. Default is T//2.
    sample_interval : (int) The sweep interval with which to save clusterings.
    seed : (int) Seed used for random node permutations.
    a, b and A_alpha: float
        Parameters for the beta distribution prior for the cluster connectivities and concentration. 
        a = b = 1 yields a uniform distribution.
        A_alpha is the total cluster concentration parameter.
    return_clustering_matrix: (bool) a list of full clustering adjacency matrices instead of partial one-hot encoded clustering matrices. 
        
    Return
    ----------
    Z: (list) list of clusterings z sampled through iterations of Gibbs sweeps. 
    """ 
    assert isinstance(A, np.ndarray)
    A[np.diag_indices_from(A)] = 0 # Assume nodes aren't connected to themselves. Equivalent to np.einsum("ii->i", A)[...] = 0

    N = A.shape[0]
    # z_f = np.zeros((N,N)) # z_full. Take all the memory I need, which is not that much.
    # z_f[:, 0] = 1 # Initialize everthing to be in the first cluster.
    z = np.ones((N,1))
    np.random.seed(seed=seed) # torch.manual_seed(seed) torch.cuda.manual_seed(seed)
    Z = []
    for t in range(T):
        # node_order = np.random.permutation(N) # I could also make the full (N,T) permutations matrix to exchange speed for memory
        node_order = np.arange(N)
        for i, n in enumerate(node_order):
            # nn = np.delete(node_order, i, axis = 0)
            nn_ = np.arange(N) # Another option that is a bit simpler and doesn't permute quite as hard.
            nn = nn_[nn_ != n]
            m = np.sum(z[nn,:], axis = 0) # Could also use my old cluster representation, which is probably faster.
            K_ind = np.nonzero(m)[0]
            K = len(K_ind) # Number of clusters
            z, m = z[:,K_ind], m[K_ind]  # Fix potential empty clusters

            m_kl = z[nn,:].T @ A[nn][:,nn] @ z[nn,:]
            m_kl[np.diag_indices_from(m_kl)] //= 2 # The edges are undirected and the previous line counts edges within a cluster twice as if directed. Compensate for that here.

            m_bar_kl = np.outer(m, m) - np.diag(m * (m + 1) / 2) - m_kl
            r_nl = (z[nn,:].T @ A[nn,n]) # node n connections to each cluster (l). 

            n_C = z
            r_nl_matrix = (A @ n_C)
            assert np.all(r_nl == r_nl_matrix[n])

            # Calculate the big log posterior for the cluster allocation for node n. (k+1 possible cluster allocations)
            logP = (np.sum(np.vstack((betaln(m_kl + r_nl + a, m_bar_kl + m - r_nl + b) - betaln(m_kl + a, m_bar_kl + b), \
                                     betaln(r_nl+a, m-r_nl+b)-betaln(a,b))), axis=1) + np.log(np.append(m,A_alpha))) # k are rows and l are columns. Sum = log product over l.
            
            P = np.exp(logP-np.max(logP)) # Avoid underflow and turn into probabilities
            cum_post = np.cumsum(P/np.sum(P))
            new_assign = int(np.sum(np.random.rand() > cum_post)) # Calculate which cluster by finding where in the probability distribution rand() lands.
            z[n,:] = 0
            if new_assign == z.shape[1]:
                z = np.hstack((z, np.zeros((N,1))))
            z[n,new_assign] = 1

            # z = z[np.nonzero(np.sum(z, axis = 0))] # But this new clustering can't have fewer clusters, since we compensated at the beginning.
            assert np.all(np.sum(z, axis = 0) > 0)
        print(K)
        Z.append(z if not return_clustering_matrix else z @ z.T) # for each z, the full clustering adjacency matrix is now just z @ z.T
    
    assert len(Z) == T
    Z = (Z[burn_in_buffer:] if burn_in_buffer is not None else Z[T//2:])
    if sample_interval is not None: Z = Z[::sample_interval]
    return Z

def Gibbs_sample_torch(A, T, burn_in_buffer = None, sample_interval = None, seed = 42, a = 1, b = 1, A_alpha = 1, return_clustering_matrix = False):
    """Perform one Gibbs sweep with a specified node order to yield one sampled clustering.

    Parameters
    ----------
    A : (2D ndarray) Adjacency matrix.
    T : (int) Number of Gibbs Sweeps.
    burn_in_buffer : (int) Number of initial samples to discard. Default is T//2.
    sample_interval : (int) The sweep interval with which to save clusterings.
    seed : (int) Seed used for random node permutations.
    a, b and A_alpha: float
        Parameters for the beta distribution prior for the cluster connectivities and concentration. 
        a = b = 1 yields a uniform distribution.
        A_alpha is the total cluster concentration parameter.
    return_clustering_matrix: (bool) a list of full clustering adjacency matrices instead of partial one-hot encoded clustering matrices. 
        
    Return
    ----------
    Z: (list) list of clusterings z sampled through iterations of Gibbs sweeps. 
    """ 


    N = A.shape[0]
    A[torch.arange(N),torch.arange(N)] //= 2 # Assume nodes aren't connected to themselves.
    # z_f = np.zeros((N,N)) # z_full. Take all the memory I need, which is not that much.
    # z_f[:, 0] = 1 # Initialize everthing to be in the first cluster.
    z = torch.ones((N,1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    Z = []
    for t in range(T):
        # node_order = torch.randperm(N) # Vary the node order to avoid committing to one specific arbitrary node order. 
            # I could also make the full (N,T) permutations matrix to exchange speed for memory
        node_order = torch.arange(N) # Mikkel's version. Just same the same initial arbitrary node order every time
        for i, n in enumerate(node_order):
            # nn = node_order[node_order != n] # Permute hard in here as well because we can.
            nn_ = torch.arange(N) # Another option that is a bit simpler and doesn't permute quite as hard.
            nn = nn_[nn_ != n]
            m = torch.sum(z[nn,:], axis = 0) # Could also use my old cluster representation, which is probably faster.
            K_ind = m.nonzero(as_tuple=True)[0] # K_ind = m.nonzero(as_tuple=True)[0]
            K = len(K_ind) # Number of clusters
            z, m = z[:,K_ind], m[K_ind]  # Fix potential empty clusters

            m_kl = z[nn,:].T @ A[nn][:,nn] @ z[nn,:]
            m_kl[torch.arange(K),torch.arange(K)] //= 2 # The edges are undirected and the previous line counts edges within a cluster twice as if directed. Compensate for that here.

            m_bar_kl = torch.outer(m, m) - torch.diag(m * (m + 1) / 2) - m_kl
            r_nl = (z[nn,:].T @ A[nn,n]) # node n connections to each cluster (l). 

            # Calculate the big log posterior for the cluster allocation for node n. (k+1 possible cluster allocations)
            logP = torch.sum(torch.vstack((betaln(m_kl + r_nl + a, m_bar_kl + m - r_nl + b) - betaln(m_kl + a, m_bar_kl + b), \
                                     betaln(r_nl+a, m-r_nl+b)-betaln(a,b))), axis=1) + torch.log(torch.hstack((m,torch.tensor([A_alpha])))) # k are rows and l are columns. Sum = log product over l.
            
            P = torch.exp(logP-torch.max(logP)) # Avoid underflow and turn into probabilities
            cum_post = torch.cumsum(P/torch.sum(P), dim=0)
            new_assign = int(torch.sum(torch.rand(1) > cum_post)) # Calculate which cluster by finding where in the probability distribution rand() lands.
            z[n,:] = 0
            if new_assign == z.shape[1]:
                z = torch.hstack((z, torch.zeros((N,1))))
            z[n,new_assign] = 1

            # z = z[np.nonzero(np.sum(z, axis = 0))] # But this new clustering can't have fewer clusters, since we compensated at the beginning.
            assert torch.all(torch.sum(z, axis = 0) > 0)
        Z.append(z if not return_clustering_matrix else z @ z.T) # for each z, the full clustering adjacency matrix is now just z @ z.T

    assert len(Z) == T
    Z = (Z[burn_in_buffer:] if burn_in_buffer is not None else Z[T//2:])
    if sample_interval is not None: Z = Z[::sample_interval]
    return Z



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

    # A_alpha is the concentration parameter.
    # A constant concentration corrosponds to the chinese restaurant process. 
    N = len(A)
    values, nk = np.unique(C,
                           return_counts=True)  # nk is an array of counts, so the number of elements in each cluster.
    K_bar = len(values) # number of non empty clusters.

    # nk (array of number of nodes in each cluster)
    log_p_z = (gammaln(A_alpha) + K_bar*(np.log(A_alpha)) - gammaln(A_alpha + N)) + np.sum(gammaln(nk))

    return log_p_z if log else np.exp(log_p_z)


def torch_posterior(A_in, C_in, a=None, b=None, A_alpha=None, log=True):
    # Likelihood part
    if a is None:
        a = torch.ones(1)
    if b is None:
        b = torch.ones(1)
    if A_alpha is None:
        A_alpha = torch.ones(1)

    A = torch.t_copy(A_in)
    C = torch.t_copy(torch.tensor(C_in, dtype=torch.int32))
    torch.einsum("ii->i", A)[...] = 0   # Fills the diagonal with zeros.
    values, nk = torch.unique(C, return_counts=True)
    n_C = torch.eye(int(C.max()) + 1)[C]

    m_kl = n_C.T @ A @ n_C
    torch.einsum("ii->i", m_kl)[...] //= 2  # m_kl[np.diag_indices_form(m_kl)] //= 2 should do the same thing.

    m_bar_kl = torch.outer(nk, nk) - torch.diag(nk * (nk + 1) / 2) - m_kl

    print(m_bar_kl.device, m_kl.device)

    if str(a.device)[:4] == 'cuda':
        a_ = a.detach().cpu().numpy()
        b_ = b.detach().cpu().numpy()
        m_kl_ = m_kl.detach().cpu().numpy()
        m_bar_kl_ = m_bar_kl.detach().cpu().numpy()
        logP_x_giv_z = torch.tensor(np.sum(betaln(m_kl_ + a_, m_bar_kl_ + b_) - betaln(a_, b_)))
    else:
        logP_x_giv_z = torch.sum(betaln(m_kl + a, m_bar_kl + b) - betaln(a, b))

    # Prior part
    N = len(A)
    values, nk = torch.unique(C, return_counts=True)
    K_bar = len(values) # number of empty clusters.
    
    log_p_z = (torch_gammaln(A_alpha) + K_bar*(torch.log(A_alpha)) - torch_gammaln(A_alpha + N)) + torch.sum(torch_gammaln(nk))

    # Return joint probability, which is proportional to the posterior
    return logP_x_giv_z + log_p_z if log else torch.exp(logP_x_giv_z + log_p_z)



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
    
    A_adj = np.array([[1,0,1,1],[0,0,1,0],[1,1,1,0],[1,0,0,1]])
    Gibbs_sample_np(A_adj, T = 100, burn_in_buffer = None, sample_interval = 10, seed = 42, a = 1, b = 1, A_alpha = 1)

    idxs = np.random.permutation(np.arange(l * k))
    A_random = A_adj[idxs][:, idxs]  # It makes sense to permute both axes in the same way.
    # Otherwise, you change the edges and their directionality. 

    # Gibbs_likelihood(A_random, np.array([0,1,1,0,0,2,2,0,3,3]), a = 0.5, b = 0.5, log = True)

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
        probs_C_log[i] = p_x_giv_z(A_random, C, a=1 / 2, b=1 / 2) + p_z(A_random, C, A_alpha=1)
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
