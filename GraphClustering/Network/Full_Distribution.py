
import os
print(os.getcwd()) # You should be running this from the GFlowNetsA directory. 
import numpy as np
import torch
from scipy.special import logsumexp
import matplotlib.pyplot as plt
try:
    from GraphClustering import GraphNet # I created a launch.json to make this work for debugging sessions. It is infuriating that it doesn't work regularly. 
except: # Do not change this if it is unnecessary for you. Directly picking the cwd for jupyter notebooks can be a massive hassle in VSCode.
    import sys
    print("Appending to sys path")
    sys.path.append(os.getcwd()) # This is really ugly
    from GraphClustering import GraphNet

    # print("Previous import statement didn't work. Changing cwd to parent directory.") # 
    # for _ in range(4):
    #     print("Stepping one directory up.")
    #     try:
    #         os.chdir("..")
    #         print(os.getcwd())
    #         from GraphClustering import GraphNet
    #         print("Successful import.")
    #         break
    #     except:
    #         pass
from GraphClustering import IRM_graph, clusterIndex
from GraphClustering import Cmatrix_to_array, torch_posterior

# These two are net methods, but are useful to have as stand alone functions. See net for details.
def get_clustering_matrix(clustering_list, number_of_clusters):
    N = len(clustering_list)
    clustering_matrix = torch.zeros((N,N))
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
    for i in range(n-1):
        perm.append([])
        for partial in perm[i]:
            for j in range(1, max(partial) + 2):
                perm[i + 1].append(partial + [j])

    return np.array(perm[-1])-1

def allPosteriors(N, a, b, alpha, log, joint = False):
    # Computing posteriors for all clusters.
    clusters_all = allPermutations(N)
    Bell = len(clusters_all)
    clusters_all_post = np.zeros(Bell)
    for i, cluster in enumerate(clusters_all):
        posterior = torch_posterior(A_random, cluster, a=torch.tensor(a), b=torch.tensor(b), alpha = torch.tensor(alpha), log= True)
        clusters_all_post[i] = posterior
    if joint: return clusters_all_post # Return the joint probability instead of normalizing.
    cluster_post = clusters_all_post - logsumexp(clusters_all_post) # Normalize them into proper log probabilities
    if not log: cluster_post = np.exp(cluster_post)
    return cluster_post

def create_graph(N, a, b, alpha, log = True, seed = 42):
    # Making the graph, and outputting cluster indexes.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    adjacency_matrix, clusters = IRM_graph(alpha = alpha, a = a, b = b, N = N)
    cluster_idxs = clusterIndex(clusters)
    clusters = len(clusters)
    return adjacency_matrix, cluster_idxs, clusters

def scramble_graph(adjacency_matrix, clustering_list = None, seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    N = adjacency_matrix.size()[0]    
    idxs = torch.randperm(N, dtype=torch.int64)
    A_random = adjacency_matrix[idxs][:, idxs]
    if clustering_list is None: return A_random, idxs
    else:
        # This makes sure to represent the cluster in the correct format despite the clustering array being permuted. 
        N_clusters = len(torch.unique(clustering_list))
        clustering_matrix = get_clustering_matrix(clustering_list.clone().detach()+1, torch.tensor(N_clusters))
        cluster_random, cluster_num = get_clustering_list(clustering_matrix[idxs][:, idxs]) 
        return A_random, idxs, cluster_random, cluster_num

def fix_net_clusters(cluster_prob_dict, clusters_all, log = True):
    Bell, N = clusters_all.shape
    net_posteriors = torch.zeros(Bell)
    clusters_all_tensor = torch.tensor(clusters_all+1)
    assert -0.1 < torch.logsumexp(torch.tensor(list(cluster_prob_dict[N].values())), (0)) < 0.1 # Make sure that the probabilities sum to 1. 
    for net_c, post in cluster_prob_dict[N].items():
        # Vectorize this because I can.
        cluster_ind = torch.argwhere(torch.all(torch.eq(clusters_all_tensor,net_c), dim=1) == 1)[0][0] 
        if not log: net_posteriors[cluster_ind] += post
        else: 
            if net_posteriors[cluster_ind] == 0: net_posteriors[cluster_ind] = post
            else: net_posteriors[cluster_ind] = torch.logaddexp(net_posteriors[cluster_ind], post)
    assert -0.1 < torch.logsumexp(net_posteriors, (0)) < 0.1
    return net_posteriors

def clusters_all_index(clusters_all_tensor, specific_cluster_list):
    cluster_ind = torch.argwhere(torch.all(torch.eq(clusters_all_tensor, specific_cluster_list), dim=1) == 1)[0][0]
    return cluster_ind

if __name__ == '__main__':
    N =  3
    a, b, alpha = 0.5, 0.5, 3
    log = True
    seed = 46
    adjacency_matrix, cluster_idxs, clusters = create_graph(N, a, b, alpha, log, seed)

    print(cluster_idxs)
    plt.figure()
    plt.imshow(adjacency_matrix)
    plt.show()
    # sys.exit()
    
    A_random, idxs, cluster_random, cluster_num = scramble_graph(adjacency_matrix, clustering_list = cluster_idxs, seed = 42)
    
    print(cluster_random-1)
    plt.figure()
    plt.imshow(A_random)
    plt.show()
    

    clusters_all = allPermutations(N)
    cluster_post = allPosteriors(N, a, b, alpha, log, joint = False)
    print("Log Probabilities: ", cluster_post)
    print("Probabilities: ", allPosteriors(N, a, b, alpha, log = False, joint = False))
    print(clusters_all)
    
    f = plt.figure()
    plt.title('Cluster Posterior Probabilites by Index')
    plt.plot(cluster_post, "o")
    plt.xlabel("Cluster Index")
    plt.ylabel("Posterior Probability")
    plt.show()


    sort_idx = np.argsort(cluster_post)
    # Results
    top = 10
    print("Total possible clusters: "+str(len(sort_idx)))
    print("Ground truth: "+str((cluster_random-1).tolist()))
    print("Top clusterings:")
    for i, idx in enumerate(np.flip(sort_idx)[:top]):
        print(str(i+1)+": "+str(clusters_all[idx]))

    f = plt.figure()
    plt.title('Cluster Posterior Probabilites by Magnitude')
    plt.plot(cluster_post[sort_idx], "o")
    plt.xlabel("Sorted Cluster Index")
    plt.ylabel("Posterior Probability")
    plt.show()


    net = GraphNet(n_nodes=adjacency_matrix.size()[0], a = a, b = b, alpha = alpha)
    X = net.sample_forward(adjacency_matrix=A_random, epochs=100)
    net.train(X, epochs=1000) # This is the time consuming part. 
    
    exact = True
    if exact:
        cluster_prob_dict = net.full_sample_distribution_G(adjacency_matrix = A_random, log = log)
        # print(cluster_prob_dict) # Here there is the significant problem of cleaning up the dictionary, since tensors are mutable and constitue unique keys.

        net_posteriors = fix_net_clusters(cluster_prob_dict, clusters_all, log = log)
        net_posteriors_numpy = net_posteriors.detach().numpy()

    N_samples = 1000
    if N_samples:
        clusters_all_tensor = torch.tensor(clusters_all+1)
        X1 = net.sample_forward(adjacency_matrix = A_random, epochs= N_samples)

        sample_posterior_counts = torch.zeros(len(clusters_all))

        for x in X1:
            x_c_list = get_clustering_list(net.get_matrices_from_state(x)[1])[0]
        
            cluster_ind = clusters_all_index(clusters_all_tensor, specific_cluster_list = x_c_list)
            sample_posterior_counts[cluster_ind] += 1

        sample_posterior_probs = sample_posterior_counts/torch.sum(sample_posterior_counts)
        if log:
            sample_posterior_probs = torch.log(sample_posterior_probs)
            assert -0.1 < torch.logsumexp(sample_posterior_probs, (0)) < 0.1
        sample_posteriors_numpy = sample_posterior_probs.detach().numpy()

    
    f = plt.figure()
    plt.title('Cluster Posterior Log Probabilites by Magnitude:\nExact and extracted from network')
    plt.plot(cluster_post[sort_idx], "bo")
    if exact: plt.plot(net_posteriors_numpy[sort_idx], "ro", markersize=4)
    if N_samples: plt.plot(sample_posteriors_numpy[sort_idx], "gx")
    plt.xlabel("Sorted Cluster Index")
    plt.ylabel("Posterior Probability")
    plt.legend(["Exact values", "From Network", "Sampled Empirically"])
    # plt.ylim(0, -5)
    plt.tight_layout()
    plt.show()

    f = plt.figure()
    plt.title('Cluster Posterior Probabilites by Magnitude:\nExact and extracted from network')
    plt.plot(np.exp(cluster_post[sort_idx]), "bo")
    if exact: plt.plot(np.exp(net_posteriors_numpy[sort_idx]), "ro", markersize=4)
    if N_samples: plt.plot(np.exp(sample_posteriors_numpy[sort_idx]), "gx")
    plt.xlabel("Sorted Cluster Index")
    plt.ylabel("Posterior Probability")
    plt.legend(["Exact values", "From Network", "Sampled Empirically"])
    # plt.ylim(0, -5)
    plt.tight_layout()
    plt.show()





