from GraphClustering import *

if __name__ == '__main__':
    n_min = 3
    n_max = 6
    epochs = 100

    for n in range(n_min, n_max+1):
        n_samples = 1000
        a, b, A_alpha = (1,1,1)
        net = GraphNetNodeOrder(n_nodes=n)
        net.load_forward(prefix=f'Data/SmallNet_{n}_99')
        adjacency_matrix, _ = IRM_graph(a, b, A_alpha, N=n)
        X = net.sample_forward(adjacency_matrix, n_samples=n_samples, timer=True)
        net.train(X, epochs=epochs)
        X1 = net.sample_forward(adjacency_matrix, n_samples=n_samples, timer=True)

        # sample_posteriors_numpy = empiricalSampleDistribution(X1, n_nodes=n, log=False)
        # cluster_post = allPosteriors(adjacency_matrix, a, b, A_alpha, log=True, joint=False)
        gibbsSamples = Gibbs_sample_torch(adjacency_matrix, n_samples * 2,
                                          return_clustering_matrix=True)
        tempSamples = torch.zeros((n_samples, n ** 2 * 2))
        for i, sample in enumerate(gibbsSamples):
            tempSamples[i] = torch.concat((adjacency_matrix.flatten(), torch.tensor(sample.flatten())))
        # gibbsSamples = torch.tensor([ for sample in gibbsSamples])
        # gibbsDistribution = empiricalSampleDistribution(tempSamples, n, numpy=True, log=True)
        # inf_mask = gibbsDistribution == -np.inf
        # gibbsDistribution[inf_mask] = np.min(gibbsDistribution[np.logical_not(inf_mask)])
        # sort_idx = np.argsort(cluster_post)
        #
        # cluster_post = np.exp(cluster_post)
        # gibbsDistribution = np.exp(gibbsDistribution)

        compareIRMSamples(tensors=[X1, tempSamples], net=net,
                          names=['GFlowNet', 'Gibbs Sampler'],
                          title=f'Histogram of log IRM values for GFlowNet vs GibbsSampler'
                                f'\non graph of {n} nodes after {epochs} epochs of training',
                          filenameSave=f'Plots/SmallComparisonPlot_{n}_{epochs}.png',
                          # topNFilename=f'Data/{prefixString}TopClusterings_{epochs}_',
                          n=5,
                          runGibbs=True,
                          postfixSaveGibbs=f'small_{n}',
                          roundTo=2
                          )

