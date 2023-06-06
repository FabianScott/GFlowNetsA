from GraphClustering.Core.Core import GraphNet, torch_posterior, check_gpu
import networkx as nx
import torch

if __name__ == '__main__':
    check_gpu()
    n_samples = 1
    epoch_interval = 1
    min_epochs = 0
    max_epochs = 50

    G = nx.karate_club_graph()
    Adj_karate = nx.adjacency_matrix(G).todense()
    Adj_karate = torch.tensor(Adj_karate > 0)

    with open(f'Data/KarateResults_{min_epochs}_{max_epochs}_{n_samples}.txt', 'w') as file:
        with open(f'Data/KarateResultsIRM_{min_epochs}_{max_epochs}_{n_samples}.txt', 'w') as file2:
            for epochs in range(0, max_epochs + 1, epoch_interval):
                file.write(f'{epochs},')
            file.write('\n')

            for epochs in range(0, max_epochs + 1, epoch_interval):
                file2.write(f'{epochs},')
            file2.write('\n')

            for epochs in range(min_epochs, max_epochs + 1, epoch_interval):
                net = GraphNet(len(G.nodes))
                X1 = net.sample_forward(Adj_karate, n_samples=n_samples, timer=True)
                net.train(X1, epochs=epochs)
                X2 = net.sample_forward(Adj_karate, n_samples=n_samples, timer=True)

                net_values1, IRM_values1 = [], []

                for state in X1:
                    net_values1.append(net.predict(state))
                    adjacency_matrix, clustering_matrix = net.get_matrices_from_state(state)
                    clustering_list, _ = net.get_clustering_list(clustering_matrix)
                    IRM_values1.append(torch_posterior(adjacency_matrix, clustering_list - 1))

                difference1 = sum(abs(torch.tensor(net_values1) - torch.tensor(IRM_values1)))
                file.write(f'{difference1},')

                net_values2, IRM_values2 = [], []
                for state in X2:
                    net_values2.append(net.predict(state))
                    adjacency_matrix, clustering_matrix = net.get_matrices_from_state(state)
                    clustering_list, _ = net.get_clustering_list(clustering_matrix)
                    IRM_values2.append(torch_posterior(adjacency_matrix, clustering_list - 1))

                # Should hopefully be positive and large, as the network samples more valuable clusters
                difference_IRM = sum(torch.tensor(IRM_values2) - torch.tensor(IRM_values1))
                file2.write(f'{difference_IRM},')



