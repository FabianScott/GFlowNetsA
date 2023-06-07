try:
    from Core import GraphNet, torch_posterior, check_gpu
except ModuleNotFoundError:
    from GraphClustering.Core.Core import GraphNet, torch_posterior, check_gpu
from copy import deepcopy
import networkx as nx
import pandas as pd
import numpy as np
import torch


def train_and_save(net, X1, adjacency_matrix, epochs, n_samples, array1, array2, filename1, filename2, header, i):
    """
    Helper function to test the network on larger graphs.
    Trains the network on the given sample, epochs
    number of epochs. Then samples n_samples number of
    new samples. On the X1 sample we calculate the network's
    output values and find the difference between them and
    the IRM values. For the new samples the IRM values alone
    are calculated. Both values are then inserted into the
    given arrays which are then converted into DataFrames
    and saved to csv files.
    :param net:
    :param X1:
    :param adjacency_matrix:
    :param epochs:
    :param n_samples:
    :param array1:
    :param array2:
    :param filename1:
    :param filename2:
    :param header:
    :param i: index at which to insert values
    :return:
    """
    net.train(X1, epochs=epochs)  # Train an extra epoch interval
    X2 = net.sample_forward(adjacency_matrix, n_samples=n_samples, timer=True)

    net_values1, IRM_values1 = [], []

    for state in X1:
        net_values1.append(net.predict(state))
        adjacency_matrix, clustering_matrix = net.get_matrices_from_state(state)
        clustering_list, _ = net.get_clustering_list(clustering_matrix)
        IRM_values1.append(torch_posterior(adjacency_matrix, clustering_list - 1))

    difference1 = sum(abs(torch.tensor(net_values1) - torch.tensor(IRM_values1)))

    net_values2, IRM_values2 = [], []
    for state in X2:
        # net_values2.append(net.predict(state))
        adjacency_matrix, clustering_matrix = net.get_matrices_from_state(state)
        clustering_list, _ = net.get_clustering_list(clustering_matrix)
        IRM_values2.append(torch_posterior(adjacency_matrix, clustering_list - 1))

    IRM_sum = sum(IRM_values2)  # Tracking the amount of flow the network samples

    # Save the numbers every iteration
    array1[i] = difference1
    array2[i] = IRM_sum

    df1 = pd.concat((pd.DataFrame(header), pd.DataFrame(array1)), axis=1)
    df2 = pd.concat((pd.DataFrame(header), pd.DataFrame(array2)), axis=1)

    df1.to_csv(filename1, index=False, header=False)
    df2.to_csv(filename2, index=False, header=False)

    # Use the previous samples for the next iteration
    return X2


if __name__ == '__main__':
    """
    Define the interval and step for epochs you want to test
    on the karate club graph. The script saves two different
    values, firstly, the difference in the network's prediction 
    of each sampled state and the IRM value, secondly the sum
    of the IRM values for the sampled states.
    """
    check_gpu()
    n_samples = 10
    epoch_interval = 10
    min_epochs = 0
    max_epochs = 30

    Adj_karate = torch.tensor(pd.read_csv("Adj_karate.csv", header=None, dtype=int).to_numpy())
    net = GraphNet(Adj_karate.shape[0])
    X1 = net.sample_forward(Adj_karate, n_samples=n_samples, timer=True)

    header = np.array([epochs for epochs in range(0, max_epochs + 1, epoch_interval)])
    filename1 = f'Data/KarateResults_{min_epochs}_{max_epochs}_{n_samples}.csv'
    filename2 = f'Data/KarateResultsIRM_{min_epochs}_{max_epochs}_{n_samples}.csv'

    array1 = np.zeros(len(header))
    array2 = np.zeros(len(header))

    train_and_save(net, X1, Adj_karate, min_epochs, n_samples, array1, array2, filename1, filename2, header, 0)
    net.save(prefix=f'Karate_{min_epochs}_{max_epochs}_{n_samples}_', postfix=str(0))

    for i in range(1, (max_epochs // epoch_interval) + 1):
        X1 = train_and_save(net, X1, Adj_karate, epoch_interval, n_samples, array1, array2, filename1, filename2, header, i)
        net.save(prefix=f'Karate_{min_epochs}_{max_epochs}_{n_samples}_', postfix=str(epoch_interval * i))
