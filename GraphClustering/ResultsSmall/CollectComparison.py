from GraphClustering.Core.Core import *
import pandas as pd


def transferLearning(fully_trained_networks, max_epochs, epoch_interval, n_samples_distribution, saveFilename):
    test_results = []
    for net in tqdm(fully_trained_networks):
        N = net.n_nodes
        test_temp = []
        adjacency_matrix_test, clusters_test = IRM_graph(A_alpha=A_alpha, a=a, b=b, N=N)
        cluster_post = allPosteriors(adjacency_matrix_test, a, b, A_alpha, log=True, joint=False)
        X1 = net.sample_forward(adjacency_matrix_test, n_samples=n_samples_distribution)
        sample_posteriors_numpy = empiricalSampleDistribution(X1, N, net, log=True, numpy=True)
        inf_mask = sample_posteriors_numpy == -np.inf
        sample_posteriors_numpy[inf_mask] = np.min(sample_posteriors_numpy[np.logical_not(inf_mask)])
        sort_idx = np.argsort(cluster_post)
        difference = sum(abs(cluster_post[sort_idx] - sample_posteriors_numpy[sort_idx]))
        test_temp.append(difference)
        for epochs in tqdm(range(0, max_epochs + epoch_interval, epoch_interval), desc='Epoch Iteration'):
            losses = net.train(X1, epochs=epoch_interval, verbose=True)
            # cluster_prob_dict = net.full_sample_distribution_G(adjacency_matrix=adjacency_matrix,
            #                                                                 log=True,
            #                                                                 fix=False)
            # fixed_probs = net.fix_net_clusters(cluster_prob_dict, log=True)
            X1 = net.sample_forward(adjacency_matrix_test, n_samples=n_samples_distribution, timer=True)
            sample_posteriors_numpy = empiricalSampleDistribution(X1, N, net, log=True, numpy=True)
            inf_mask = sample_posteriors_numpy == -np.inf
            sample_posteriors_numpy[inf_mask] = np.min(sample_posteriors_numpy[np.logical_not(inf_mask)])
            sort_idx = np.argsort(cluster_post)
            difference = sum(abs(cluster_post[sort_idx] - sample_posteriors_numpy[sort_idx]))
            test_temp.append(difference)

        net.save(f'Data/SmallNet_{N}_{max_epochs}')
        gibbsSamples = Gibbs_sample_torch(adjacency_matrix_test, n_samples_distribution * 2,
                                          return_clustering_matrix=True)
        tempSamples = torch.zeros((n_samples_distribution, N ** 2 * 2))
        for i, sample in enumerate(gibbsSamples):
            tempSamples[i] = torch.concat((adjacency_matrix_test.flatten(), torch.tensor(sample.flatten())))
        # gibbsSamples = torch.tensor([ for sample in gibbsSamples])
        gibbsDistribution = empiricalSampleDistribution(tempSamples, N, net, numpy=True, log=True)
        inf_mask = gibbsDistribution == -np.inf
        gibbsDistribution[inf_mask] = np.min(gibbsDistribution[np.logical_not(inf_mask)])

        plot_posterior(cluster_post,
                       sort_idx=sort_idx,
                       net_posteriors_numpy=None,
                       gibbs_sample_posteriors=gibbsDistribution,
                       sample_posteriors_numpy=sample_posteriors_numpy,
                       saveFilename=f'Plots/TransferPosteriorPlot_{N}_{max_epochs}_{n_samples_distribution}.png',
                       title=f'Posterior values from GflowNet, Gibbs and IRM\nOn graph of {N} nodes')
        test_results.append(test_temp)

    pd.DataFrame(test_results).to_csv(saveFilename)
    return


if __name__ == '__main__':

    networks = []
    for i in range(2, 7):
        net_temp = GraphNetNodeOrder(i)
        net_temp.load_forward(prefix=f'SmallNet_{i}_99')
        networks.append(net_temp)
    transferLearning(networks,
                     max_epochs=99,
                     epoch_interval=1,
                     n_samples_distribution=1000,
                     saveFilename='TransferData.csv')

