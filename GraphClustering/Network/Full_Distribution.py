
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
    sys.path.append(os.getcwd()) # This is really ugly. Will fix and make it pip installable when the rest works.
    sys.path.append(os.path.join(os.getcwd(), "GraphClustering", "Core"))
    from GraphClustering import GraphNet

from GraphClustering import IRM_graph, clusterIndex
from GraphClustering import Cmatrix_to_array, torch_posterior
import time

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

def save_img(N, log, i, train_epochs):
    # This function has to be used from the GFlowNetsA directory. 
    stage = "pretrain" if i == 0 else "posttrain"
    filename = 'N{N}_{log}posteriors_{stage}{train_epochs}'.format(N=N, log= ("log_" if log else ""), stage=stage, train_epochs = (train_epochs if i == 1 else ""))
    completeName = os.path.join(os.getcwd(), "Plots", "full_distribution" , filename+".png")
    plt.savefig(completeName, bbox_inches='tight')

def time_func(func, n = 1, *kwargs):
    t0 = time.process_time()
    for i in range(n):
        func(*kwargs)
    t1 = time.process_time()
    t_total = t1-t0
    return t_total

def full_distribution_test(N, a=1, b=1, alpha=3, log = True, seed = 42, plot_adj = False, check_adj = False, plot_results = False, save_results = False,
                        _print_clusterings = False, top = 10, exact = False, train_samples = 100, N_samples = None, train_epochs = 100):
    """
    A combined test script to test the exact IRM posterior values against those learned by the GFlowNet.
        It is flexible and can ignore test methods according to parameter values.
    :param
        N: (int) Number of nodes in the graph. The main scaling factor and time sink.
        a, b, alpha: (float) Priors for the IRM. Determine both how the graph is generated from IRM and how IRM posteriors are calculated.
            Node links are beta distributed according to a and b.
            The cluster concentration is distributed as a chinese restaurant process (CRP) with alpha as concentration parameter. 
        log: (bool) Whether or not to compute the function using log-probabilities.
        seed: (int) Seed to ensure result consistency
        plot_adj: (bool) Whether or not to plot the original and scrambled adjacency matrices.
        check_adj: (bool) Terminate after generating adjacency matrix.
        
        plot_results: (bool) Whether or not to display plots of the results.
        save_results: (bool) Whether or not to save plots of the results in the plots folder.

        _print_clusterings: (bool) Whether or not to print all possible clusterings and posteriors. Grows fast with N, mainly for debugging. 

        top: (int) Print the "top" clusterings.
        exact: (bool) Calculate the exact probability distribution of the GFlowNet using the forward policy.
        train_samples: (int) Number of samples used to train the network.
        N_samples: (int) Calculate the empirical probability distribution of the GFlowNet by sampling N_samples. If none, N_samples = 10*np.power(4,N).
        train_epochs: (int) Train the network for "train_epochs". This is the major time sink.

    :soft return: If plots: returns plots comparing IRM posterior values for the different clusterings.
    """
    t0 = time.process_time()

    adjacency_matrix, cluster_idxs, clusters = create_graph(N, a, b, alpha, log, seed)

    if plot_adj:
        print(cluster_idxs)
        plt.figure()
        plt.imshow(adjacency_matrix)
        plt.show()
        if check_adj: sys.exit()
    
    A_random, idxs, cluster_random, cluster_num = scramble_graph(adjacency_matrix, clustering_list = cluster_idxs, seed = 42)
    
    if plot_adj:
        print(cluster_random-1)
        plt.figure()
        plt.imshow(A_random)
        plt.show()
    

    clusters_all = allPermutations(N)
    cluster_post = allPosteriors(A_random, a, b, alpha, log, joint = False)
    if _print_clusterings:
        print("Log Probabilities: ", cluster_post)
        print("Probabilities: ", allPosteriors(A_random, a, b, alpha, log = False, joint = False))
        print(clusters_all)
    
    if plot_results: #IRM posterior values by index
        plot_posterior(cluster_post, sort_idx = None, net_posteriors_numpy = None, sample_posteriors_numpy = None, log = log)
        plt.show()

    sort_idx = np.argsort(cluster_post)
    # Results
    if top:
        print("Total possible clusterings: "+str(len(sort_idx)))
        print("Ground truth: "+str((cluster_random-1).tolist()))
        print("Top clusterings:")
        for i, idx in enumerate(np.flip(sort_idx)[:top]):
            print(str(i+1)+": "+str(clusters_all[idx]))

    if plot_results: # Sorted IRM posterior values
        plot_posterior(cluster_post, sort_idx = sort_idx, net_posteriors_numpy = None, sample_posteriors_numpy = None, log = log)
        plt.show()
        # sys.exit()

    net = GraphNet(n_nodes=adjacency_matrix.size()[0], a = a, b = b, alpha = alpha)
    X = net.sample_forward(adjacency_matrix=A_random, epochs=100)

    # Sample once before and after training
    for i in range(2):

        if exact:
            cluster_prob_dict = net.full_sample_distribution_G(adjacency_matrix = A_random, log = log, fix=False) # Could also use fix.
            net_posteriors = fix_net_clusters(cluster_prob_dict, clusters_all, log = log)
            net_posteriors_numpy = net_posteriors.detach().numpy()
        else: net_posteriors_numpy = None

        if N_samples is None: N_samples = 10*np.power(4,N)
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
        
        if plot_results or save_results: # Plot results before and after training.
            plot_posterior(cluster_post, sort_idx, net_posteriors_numpy, sample_posteriors_numpy, log = True)
            if save_results: save_img(N, log = True, i = i, train_epochs = train_epochs)
            if plot_results: plt.show()
            plt.close()

            plot_posterior(cluster_post, sort_idx, net_posteriors_numpy, sample_posteriors_numpy, log = False)
            if save_results: save_img(N, log = False, i = i, train_epochs = train_epochs)
            if plot_results: plt.show()
            plt.close()

        if i == 0:

            if train_epochs: net.train(X, epochs=train_epochs) # This is the time consuming part. 
            
    t1 = time.process_time()
    t_total = t1-t0
    print("Total elapsed time for ",N, "nodes: ", t_total)



if __name__ == '__main__':
    """
    A combined test script to test the exact IRM posterior values against those learned by the GFlowNet.
        It is flexible and can ignore test methods according to parameter values.
    :param
        N: (int) Number of nodes in the graph. The main scaling factor and time sink.
        a, b, alpha: (float) Priors for the IRM. Determine both how the graph is generated from IRM and how IRM posteriors are calculated.
            Node links are beta distributed according to a and b.
            The cluster concentration is distributed as a chinese restaurant process (CRP) with alpha as concentration parameter. 
        log: (bool) Whether or not to compute the function using log-probabilities.
        seed: (int) Seed to ensure result consistency
        plot_adj: (bool) Whether or not to plot the original and scrambled adjacency matrices.
        check_adj: (bool) Terminate after generating adjacency matrix.

        plot_results: (bool) Whether or not to display plots of the results.
        save_results: (bool) Whether or not to save plots of the results in the plots folder.

        _print_clusterings: (bool) Whether or not to print all possible clusterings and posteriors. Grows fast with N, mainly for debugging. 

        top: (int) Print the "top" clusterings.
        exact: (bool) Calculate the exact probability distribution of the GFlowNet using the forward policy.
        train_samples: (int) Number of samples used to train the network. 
        N_samples: (int) Calculate the empirical probability distribution of the GFlowNet by sampling N_samples. If none, N_samples = 10*np.power(4,N).
        train_epochs: (int) Train the network for "train_epochs". This is the major time sink.

    :soft return: If plots: returns plots comparing IRM posterior values for the different clusterings.
    """
    t0 = time.process_time()
    N =  4
    a, b, alpha = 1, 1, 3 # 10000
    log = True
    seed = 49
    plot_adj = True
    check_adj = False
    plot_results = True
    save_results = False
    _print_clusterings = False

    top = 10
    exact = False
    train_samples = 100
    N_samples = None
    train_epochs = 200
    adjacency_matrix, cluster_idxs, clusters = create_graph(N, a, b, alpha, log, seed)

    if plot_adj:
        print(cluster_idxs)
        plt.figure()
        plt.imshow(adjacency_matrix)
        plt.show()
        if check_adj: sys.exit()
    
    A_random, idxs, cluster_random, cluster_num = scramble_graph(adjacency_matrix, clustering_list = cluster_idxs, seed = 42)
    
    if plot_adj:
        print(cluster_random-1)
        plt.figure()
        plt.imshow(A_random)
        plt.show()
    

    clusters_all = allPermutations(N)
    cluster_post = allPosteriors(A_random, a, b, alpha, log, joint = False)
    if _print_clusterings:
        print("Log Probabilities: ", cluster_post)
        print("Probabilities: ", allPosteriors(A_random, a, b, alpha, log = False, joint = False))
        print(clusters_all)
    
    if plot_results: #IRM posterior values by index
        plot_posterior(cluster_post, sort_idx = None, net_posteriors_numpy = None, sample_posteriors_numpy = None, log = log)
        plt.show()

    sort_idx = np.argsort(cluster_post)
    # Results
    top = 10
    if top:
        print("Total possible clusterings: "+str(len(sort_idx)))
        print("Ground truth: "+str((cluster_random-1).tolist()))
        print("Top clusterings:")
        for i, idx in enumerate(np.flip(sort_idx)[:top]):
            print(str(i+1)+": "+str(clusters_all[idx]))

    if plot_results: # Sorted IRM posterior values
        plot_posterior(cluster_post, sort_idx = sort_idx, net_posteriors_numpy = None, sample_posteriors_numpy = None, log = log)
        plt.show()
        # sys.exit()

    net = GraphNet(n_nodes=adjacency_matrix.size()[0], a = a, b = b, alpha = alpha)
    X = net.sample_forward(adjacency_matrix=A_random, epochs=train_samples)

    # Sample once before and after training
    for i in range(2):

        if exact:
            cluster_prob_dict = net.full_sample_distribution_G(adjacency_matrix = A_random, log = log, fix=False) # Could also use fix.
            net_posteriors = fix_net_clusters(cluster_prob_dict, clusters_all, log = log)
            net_posteriors_numpy = net_posteriors.detach().numpy()
        else: net_posteriors_numpy = None

        if N_samples is None: N_samples = 10*np.power(4,N)
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
        
        if plot_results or save_results: # Plot results before and after training.
            plot_posterior(cluster_post, sort_idx, net_posteriors_numpy, sample_posteriors_numpy, log = True)
            if save_results: save_img(N, log = True, i = i, train_epochs = train_epochs)
            if plot_results: plt.show()
            plt.close()

            plot_posterior(cluster_post, sort_idx, net_posteriors_numpy, sample_posteriors_numpy, log = False)
            if save_results: save_img(N, log = False, i = i, train_epochs = train_epochs)
            if plot_results: plt.show()
            plt.close()

        if i == 0:

            if train_epochs: net.train(X, epochs=train_epochs) # This is the time consuming part. 

    t1 = time.process_time()
    t_total = t1-t0
    print("Total elapsed time for ",N, "nodes: ", t_total)
    





