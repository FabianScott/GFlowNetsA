
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

def allPosteriors(A_random, a, b, alpha, log, joint = False):
    # Computing posteriors for all clusters.
    N = len(A_random)
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
        cluster_ind = torch.argwhere(torch.all(torch.eq(clusters_all_tensor,net_c), dim=1) == 1)[0][0] # Find the correct cluster_ind from any net_c
        if not log: net_posteriors[cluster_ind] += post
        else: 
            if net_posteriors[cluster_ind] == 0: net_posteriors[cluster_ind] = post
            else: net_posteriors[cluster_ind] = torch.logaddexp(net_posteriors[cluster_ind], post)
    assert -0.1 < torch.logsumexp(net_posteriors, (0)) < 0.1
    return net_posteriors

def clusters_all_index(clusters_all_tensor, specific_cluster_list):
    cluster_ind = torch.argwhere(torch.all(torch.eq(clusters_all_tensor, specific_cluster_list), dim=1) == 1)[0][0]
    return cluster_ind

def plot_posterior(cluster_post, sort_idx = None, net_posteriors_numpy = None, sample_posteriors_numpy = None, log = True):
    log_string = " Log " if log else " "
    order = "index" if (sort_idx is None) else "Magnitude"
    xlab = "Cluster Index" if (sort_idx is None) else "Sorted Cluster Index"
    if sort_idx is None: sort_idx = torch.arange(len(cluster_post))

    from_network = '' if ((net_posteriors_numpy is None) or (sample_posteriors_numpy is None)) else ':\nExact and extracted from network'
    f = plt.figure()
    plt.title('Cluster Posterior' + log_string + 'Probabilites by ' + order + from_network)
    if not log: cluster_post = np.exp(cluster_post)
    plt.plot(cluster_post[sort_idx], "bo")
    if net_posteriors_numpy is not None:
        if not log: net_posteriors_numpy = np.exp(net_posteriors_numpy)
        plt.plot(net_posteriors_numpy[sort_idx], "ro", markersize=4)
    if sample_posteriors_numpy is not None:
        if not log: sample_posteriors_numpy = np.exp(sample_posteriors_numpy)
        plt.plot(sample_posteriors_numpy[sort_idx], "gx")
    plt.xlabel(xlab)
    plt.ylabel("Posterior Probability")
    plt.legend(["Exact values", "From Network", "Sampled Empirically"])
    # plt.ylim(0, -5)
    plt.tight_layout()
    return




if __name__ == '__main__':
    N =  3
    a, b, alpha = 1, 1, 3 # 10000
    log = True
    seed = 46
    plot = True
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
    cluster_post = allPosteriors(A_random, a, b, alpha, log, joint = False)
    print("Log Probabilities: ", cluster_post)
    print("Probabilities: ", allPosteriors(A_random, a, b, alpha, log = False, joint = False))
    print(clusters_all)
    
    plot_posterior(cluster_post, sort_idx = None, net_posteriors_numpy = None, sample_posteriors_numpy = None, log = log)
    plt.show()

    sort_idx = np.argsort(cluster_post)
    # Results
    top = 10
    print("Total possible clusters: "+str(len(sort_idx)))
    print("Ground truth: "+str((cluster_random-1).tolist()))
    print("Top clusterings:")
    for i, idx in enumerate(np.flip(sort_idx)[:top]):
        print(str(i+1)+": "+str(clusters_all[idx]))

    plot_posterior(cluster_post, sort_idx = sort_idx, net_posteriors_numpy = None, sample_posteriors_numpy = None, log = log)
    plt.show()
    # sys.exit()

    net = GraphNet(n_nodes=adjacency_matrix.size()[0], a = a, b = b, alpha = alpha, lr = 1)
    X = net.sample_forward(adjacency_matrix=A_random, epochs=100)
    # Sample once before and after training
    for i in range(2):
        exact = True
        train_epochs = 100

        if exact:
            cluster_prob_dict = net.full_sample_distribution_G(adjacency_matrix = A_random, log = log, fix=False) # Could also use fix.
            net_posteriors = fix_net_clusters(cluster_prob_dict, clusters_all, log = log)
            net_posteriors_numpy = net_posteriors.detach().numpy()
        else: net_posteriors_numpy = None

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

        if i == 0:
            plot_posterior(cluster_post, sort_idx, net_posteriors_numpy, sample_posteriors_numpy, log = True)
            plt.show()

            plot_posterior(cluster_post, sort_idx, net_posteriors_numpy, sample_posteriors_numpy, log = False)
            plt.show() 

            if train_epochs: net.train(X, epochs=train_epochs) # This is the time consuming part. 

    
    plot_posterior(cluster_post, sort_idx, net_posteriors_numpy, sample_posteriors_numpy, log = True)
    plt.show()

    plot_posterior(cluster_post, sort_idx, net_posteriors_numpy, sample_posteriors_numpy, log = False)
    plt.show() 





