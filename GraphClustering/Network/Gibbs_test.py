import os
print(os.getcwd())  # You should be running this from the GFlowNetsA directory.
import numpy as np
import torch
from scipy.special import logsumexp
import matplotlib.pyplot as plt

try:
    from GraphClustering import GraphNet  # I created a launch.json to make this work for debugging sessions. It is infuriating that it doesn't work regularly.
except:  # Do not change this if it is unnecessary for you. Directly picking the cwd for jupyter notebooks can be a massive hassle in VSCode.
    import sys

    print("Appending to sys path")
    sys.path.append(os.getcwd())  # This is really ugly
    from GraphClustering import GraphNet

from GraphClustering import IRM_graph, clusterIndex
from GraphClustering import Cmatrix_to_array, torch_posterior
from GraphClustering.Network.Gibbs import gibbsSampler


# These two are net methods, but are useful to have as stand alone functions. See net for details.
def get_clustering_matrix(clustering_list, number_of_clusters):
    N = len(clustering_list)
    clustering_matrix = torch.zeros((N, N))
    for cluster_no in range(1, number_of_clusters + 1):
        cluster_positions = torch.argwhere(clustering_list == cluster_no).flatten()
        indices = torch.unique(torch.combinations(torch.concat((cluster_positions, cluster_positions)), r=2), dim=0)
        for ind in indices:
            clustering_matrix[(ind[0], ind[1])] = 1
    return clustering_matrix


def get_clustering_list(clustering_matrix):
    current_clustering_copy = torch.clone(clustering_matrix)
    clustering_list = torch.zeros(current_clustering_copy.size()[0])
    number_of_clusters = 1  # starting at the empty cluster
    node_no = 0
    while torch.sum(current_clustering_copy):
        row = current_clustering_copy[node_no]
        if torch.sum(row):
            indices = torch.argwhere(row)
            clustering_list[indices] = number_of_clusters
            current_clustering_copy[indices] = 0
            current_clustering_copy[:, indices] = 0
            number_of_clusters += 1
        node_no += 1

    return clustering_list, number_of_clusters


def allPermutations(n):
    perm = [[[1]]]
    for i in range(n - 1):
        perm.append([])
        for partial in perm[i]:
            for j in range(1, max(partial) + 2):
                perm[i + 1].append(partial + [j])

    return np.array(perm[-1]) - 1


def allPosteriors(A_random, a, b, alpha, log, joint=False):
    # Computing posteriors for all clusters.
    N = len(A_random)
    clusters_all = allPermutations(N)
    Bell = len(clusters_all)
    clusters_all_post = np.zeros(Bell)
    for i, cluster in enumerate(clusters_all):
        posterior = torch_posterior(A_random, cluster, a=torch.tensor(a), b=torch.tensor(b), alpha=torch.tensor(alpha),
                                    log=True)
        clusters_all_post[i] = posterior
    if joint: return clusters_all_post  # Return the joint probability instead of normalizing.
    cluster_post = clusters_all_post - logsumexp(clusters_all_post)  # Normalize them into proper log probabilities
    if not log: cluster_post = np.exp(cluster_post)
    return cluster_post


def create_graph(N, a, b, alpha, log=True, seed=42):
    # Making the graph, and outputting cluster indexes.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    adjacency_matrix, clusters = IRM_graph(alpha=alpha, a=a, b=b, N=N)
    cluster_idxs = clusterIndex(clusters)
    clusters = len(clusters)
    return adjacency_matrix, cluster_idxs, clusters


def scramble_graph(adjacency_matrix, clustering_list=None, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    N = adjacency_matrix.size()[0]
    idxs = torch.randperm(N, dtype=torch.int64)
    A_random = adjacency_matrix[idxs][:, idxs]
    if clustering_list is None:
        return A_random, idxs
    else:
        # This makes sure to represent the cluster in the correct format despite the clustering array being permuted.
        N_clusters = len(torch.unique(clustering_list))
        clustering_matrix = get_clustering_matrix(clustering_list.clone().detach() + 1, torch.tensor(N_clusters))
        cluster_random, cluster_num = get_clustering_list(clustering_matrix[idxs][:, idxs])
        return A_random, idxs, cluster_random, cluster_num


def fix_net_clusters(cluster_prob_dict, clusters_all, log=True):
    Bell, N = clusters_all.shape
    net_posteriors = torch.zeros(Bell)
    clusters_all_tensor = torch.tensor(clusters_all + 1)
    assert -0.1 < torch.logsumexp(torch.tensor(list(cluster_prob_dict[N].values())),
                                  (0)) < 0.1  # Make sure that the probabilities sum to 1.
    for net_c, post in cluster_prob_dict[N].items():
        # Vectorize this because I can.
        cluster_ind = torch.argwhere(torch.all(torch.eq(clusters_all_tensor, net_c), dim=1) == 1)[0][
            0]  # Find the correct cluster_ind from any net_c
        if not log:
            net_posteriors[cluster_ind] += post
        else:
            if net_posteriors[cluster_ind] == 0:
                net_posteriors[cluster_ind] = post
            else:
                net_posteriors[cluster_ind] = torch.logaddexp(net_posteriors[cluster_ind], post)
    assert -0.1 < torch.logsumexp(net_posteriors, (0)) < 0.1
    return net_posteriors


def clusters_all_index(clusters_all_tensor, specific_cluster_list):
    cluster_ind = torch.argwhere(torch.all(torch.eq(clusters_all_tensor, specific_cluster_list), dim=1) == 1)[0][0]
    return cluster_ind

if __name__ == '__main__':
    N = 4
    a, b, alpha = 1, 1, 3  # 10000
    log = True
    seed = 46
    plot = True
    adjacency_matrix, cluster_idxs, clusters = create_graph(N, a, b, alpha, log, seed)

    A_random, idxs, cluster_random, cluster_num = scramble_graph(adjacency_matrix, clustering_list=cluster_idxs,
                                                                 seed=42)

    clusters_all = allPermutations(N)
    cluster_post = allPosteriors(A_random, a, b, alpha, log, joint=False)
    sort_idx = np.argsort(cluster_post)

    # Sample Gibbs
    N_samples = 2000
    A_random_numpy = A_random.numpy()
    clusters_sampled = gibbsSampler(N_samples, A_random_numpy, a, b, alpha)
    clusters_unique, clusters_count = np.unique(clusters_sampled, axis=0, return_counts=True)
    gibbs_post = np.log(clusters_count / N_samples)

    plt.plot(cluster_post[sort_idx], "bo")
    plt.plot(gibbs_post[sort_idx], "rx")
    plt.show()





